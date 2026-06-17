"""
Russian NLP sidecar.

Stack:
  razdel      — Russian tokenizer
  pymorphy3   — morphological analysis + lemmatization
  natasha     — NER (Money/Date/Person/Location/Org) + dependency parsing
  GLiNER      — zero-shot domain NER driven by config labels
  sentence-transformers (multilingual) — implicit intent detection
"""

import os
import re
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pymorphy3
import uvicorn
from fastapi import FastAPI, HTTPException
from gliner import GLiNER
from natasha import (
    Doc,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsDependencyParser,
    NewsNERTagger,
    Segmenter,
)
from pydantic import BaseModel
from razdel import tokenize as razdel_tokenize
from sentence_transformers import SentenceTransformer


# ── startup: load all models once ─────────────────────────────────────────────

print("Loading models…")

# natasha pipeline
_emb  = NewsEmbedding()
_segm = Segmenter()
_morph_tagger = NewsMorphTagger(_emb)
_dep_parser   = NewsDependencyParser(_emb)
_ner_tagger   = NewsNERTagger(_emb)
_morph_vocab  = MorphVocab()

# pymorphy3 — morphological analyser (handles every inflected form)
_morphy = pymorphy3.MorphAnalyzer()

# GLiNER — zero-shot NER with labels coming from config at runtime
_gliner = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

# Sentence-transformers — multilingual, good Russian support
_st_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

print("Models ready.")

app = FastAPI(title="Russian NLP Sidecar")


# ── config: load GLiNER domain labels from filter configs ─────────────────────

def _load_gliner_labels(config_dir: str = "/config/filters") -> list[str]:
    """
    Read all YAML/JSON filter configs and collect gliner_labels fields.
    Falls back to a sensible default set if configs not mounted.
    """
    labels: list[str] = []
    p = Path(config_dir)
    if p.exists():
        import yaml  # optional dep only needed here
        for f in p.glob("*.yaml"):
            data = yaml.safe_load(f.read_text())
            labels.extend(data.get("gliner_labels", []))
    if not labels:
        labels = [
            "цена", "максимальная цена", "минимальная цена",
            "цвет", "размер", "бренд", "материал", "состояние",
            "категория", "вес", "объём", "гарантия",
        ]
    return list(dict.fromkeys(labels))  # deduplicate, preserve order

_GLINER_LABELS = _load_gliner_labels()


# ── intent anchors (Russian seed words) ───────────────────────────────────────

_INTENT_ANCHORS_RU = [
    ("PRICE_LOW",      ["дешёвый", "недорогой", "бюджетный", "экономичный", "доступный", "выгодный"]),
    ("PRICE_HIGH",     ["дорогой", "премиальный", "люксовый", "элитный", "эксклюзивный"]),
    ("SORT_DATE_DESC", ["новый", "новинка", "свежий", "актуальный", "последний"]),
    ("SORT_POPULAR",   ["популярный", "трендовый", "хит", "бестселлер", "топ"]),
    ("CONDITION_NEW",  ["новый", "новьё", "запечатанный", "нераспакованный"]),
    ("CONDITION_USED", ["б/у", "бу", "подержанный", "секонд", "восстановленный"]),
    ("GEO_NEARBY",     ["рядом", "поблизости", "недалеко", "близко", "рядышком"]),
    ("FREE_DELIVERY",  ["бесплатная доставка", "с доставкой", "доставка бесплатно"]),
    ("IN_STOCK",       ["в наличии", "есть в наличии", "в наличие"]),
]

# precompute normalised anchor vectors
_anchor_vecs: list[tuple[str, np.ndarray, float]] = []
for _name, _seeds, *_rest in _INTENT_ANCHORS_RU:
    _threshold = _rest[0] if _rest else 0.60
    _vecs = _st_model.encode(_seeds, normalize_embeddings=True)
    _avg  = _vecs.mean(axis=0)
    _avg  = _avg / (np.linalg.norm(_avg) + 1e-9)
    _anchor_vecs.append((_name, _avg, _threshold))


# ── pydantic models ────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str
    include_embeddings: bool = False
    gliner_labels: Optional[list[str]] = None  # override config labels at request time


class MorphInfo(BaseModel):
    lemma: str
    pos: str                    # pymorphy3 POS: NOUN, ADJF, VERB, NUMR …
    case: Optional[str] = None  # Nom Gent Datv Accs Ablt Loct
    number: Optional[str] = None
    gender: Optional[str] = None
    grammemes: list[str] = []


class Token(BaseModel):
    i: int
    text: str
    lemma: str
    pos: str         # pymorphy3 POS
    dep: str         # natasha dependency relation
    head_i: int
    start: int
    end: int
    morph: MorphInfo


class Entity(BaseModel):
    text: str
    lemma: str       # lemmatised form of entity text
    label: str       # MONEY DATE PER LOC ORG (natasha) or domain label (GLiNER)
    source: str      # "natasha" | "gliner"
    start: int
    end: int
    confidence: float


class NegationScope(BaseModel):
    neg_token_i: int
    neg_text: str
    scope_token_indices: list[int]
    scope_lemmas: list[str]


class IntentSignal(BaseModel):
    name: str
    confidence: float
    source_tokens: list[int]


class AnalyzeResponse(BaseModel):
    text: str
    tokens: list[Token]
    entities: list[Entity]
    negation_scopes: list[NegationScope]
    intent_signals: list[IntentSignal]
    morphology_map: dict[str, MorphInfo]  # original_form → morph info
    embedding: Optional[list[float]] = None


# ── helpers ────────────────────────────────────────────────────────────────────

def _pymorphy_info(word: str) -> MorphInfo:
    parsed = _morphy.parse(word)
    if not parsed:
        return MorphInfo(lemma=word.lower(), pos="UNKN", grammemes=[])
    best = parsed[0]
    tag  = best.tag
    grammemes = [str(g) for g in tag.grammemes]
    return MorphInfo(
        lemma=best.normal_form,
        pos=str(tag.POS) if tag.POS else "UNKN",
        case=str(tag.case) if tag.case else None,
        number=str(tag.number) if tag.number else None,
        gender=str(tag.gender) if tag.gender else None,
        grammemes=grammemes,
    )


def _lemmatise_span(text: str) -> str:
    """Lemmatise each token of a multi-word span."""
    return " ".join(_morphy.parse(w)[0].normal_form for w in text.split())


def _resolve_negations(doc: Doc, tokens: list[Token]) -> list[NegationScope]:
    """
    Walk natasha dependency tree.
    Russian negation markers: не, нет, ни, без, кроме, нельзя.
    """
    NEG_LEMMAS = {"не", "нет", "ни", "без", "кроме", "нельзя", "никакой"}

    # build index: token_id → Token
    tok_by_id = {t.i: t for t in tokens}

    # build children map from dep tree
    children: dict[int, list[int]] = {t.i: [] for t in tokens}
    for t in tokens:
        if t.i != t.head_i:
            children[t.head_i].append(t.i)

    def subtree(i: int) -> list[int]:
        result = [i]
        for c in children.get(i, []):
            result.extend(subtree(c))
        return result

    scopes: list[NegationScope] = []
    for t in tokens:
        morph = _pymorphy_info(t.text)
        if morph.lemma in NEG_LEMMAS:
            head_i = t.head_i
            scope_indices = [i for i in subtree(head_i) if i != t.i]
            scope_lemmas  = [tok_by_id[i].lemma for i in scope_indices if i in tok_by_id]
            scopes.append(NegationScope(
                neg_token_i=t.i,
                neg_text=t.text,
                scope_token_indices=sorted(scope_indices),
                scope_lemmas=scope_lemmas,
            ))
    return scopes


def _detect_intents(text: str, tokens: list[Token]) -> list[IntentSignal]:
    signals: list[IntentSignal] = []

    for t in tokens:
        if t.pos in ("PUNCT", "CONJ"):
            continue
        vec = _st_model.encode([t.text], normalize_embeddings=True)[0]
        for name, anchor, threshold in _anchor_vecs:
            score = float(np.dot(vec, anchor))
            if score >= threshold:
                signals.append(IntentSignal(
                    name=name,
                    confidence=round(score, 4),
                    source_tokens=[t.i],
                ))

    # also run multi-word intent phrases (e.g. "в наличии", "бесплатная доставка")
    phrase_vec = _st_model.encode([text], normalize_embeddings=True)[0]
    for name, anchor, threshold in _anchor_vecs:
        score = float(np.dot(phrase_vec, anchor))
        if score >= threshold:
            signals.append(IntentSignal(name=name, confidence=round(score, 4), source_tokens=[]))

    # deduplicate — keep highest confidence per intent
    best: dict[str, IntentSignal] = {}
    for s in signals:
        if s.name not in best or s.confidence > best[s.name].confidence:
            best[s.name] = s
    return sorted(best.values(), key=lambda x: -x.confidence)


# ── main endpoint ──────────────────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    text = req.text.strip()

    # ── 1. tokenise with razdel ─────────────────────────────────────────
    raw_tokens = list(razdel_tokenize(text))

    # ── 2. morphological analysis with pymorphy3 ────────────────────────
    tokens: list[Token] = []
    morph_map: dict[str, MorphInfo] = {}
    for i, rt in enumerate(raw_tokens):
        m = _pymorphy_info(rt.text)
        morph_map[rt.text] = m
        tokens.append(Token(
            i=i,
            text=rt.text,
            lemma=m.lemma,
            pos=m.pos,
            dep="",      # filled below by natasha
            head_i=i,    # filled below
            start=rt.start,
            end=rt.stop,
            morph=m,
        ))

    # ── 3. natasha: NER + dependency parsing ────────────────────────────
    doc = Doc(text)
    doc.segment(_segm)
    doc.tag_morph(_morph_tagger)
    doc.parse_syntax(_dep_parser)
    doc.tag_ner(_ner_tagger)

    for span in doc.spans:
        span.normalize(_morph_vocab)

    # apply natasha dep relations back to our token list
    # natasha tokens align by char offsets
    char_to_tok: dict[int, int] = {t.start: t.i for t in tokens}
    for nt in doc.tokens:
        t_i = char_to_tok.get(nt.start)
        if t_i is not None:
            tokens[t_i].dep    = nt.rel or ""
            head_start = nt.head_start if hasattr(nt, "head_start") else nt.start
            tokens[t_i].head_i = char_to_tok.get(head_start, t_i)

    # collect natasha entities
    entities: list[Entity] = []
    for span in doc.spans:
        entities.append(Entity(
            text=span.text,
            lemma=span.normal or _lemmatise_span(span.text),
            label=span.type,          # PER LOC ORG MONEY DATE
            source="natasha",
            start=span.start,
            end=span.stop,
            confidence=0.88,          # natasha doesn't expose per-span score
        ))

    # ── 4. GLiNER: domain entity extraction ─────────────────────────────
    labels = req.gliner_labels or _GLINER_LABELS
    gliner_entities = _gliner.predict_entities(text, labels, threshold=0.50)
    for ge in gliner_entities:
        # skip if already covered by natasha span
        overlap = any(
            e.start <= ge["start"] < e.end or ge["start"] <= e.start < ge["end"]
            for e in entities
        )
        if not overlap:
            entities.append(Entity(
                text=ge["text"],
                lemma=_lemmatise_span(ge["text"]),
                label=ge["label"].upper().replace(" ", "_"),
                source="gliner",
                start=ge["start"],
                end=ge["end"],
                confidence=round(ge["score"], 4),
            ))

    # ── 5. negation scopes ───────────────────────────────────────────────
    negation_scopes = _resolve_negations(doc, tokens)

    # ── 6. intent signals ────────────────────────────────────────────────
    intent_signals = _detect_intents(text, tokens)

    # ── 7. sentence embedding (optional) ────────────────────────────────
    embedding: Optional[list[float]] = None
    if req.include_embeddings:
        embedding = _st_model.encode([text], normalize_embeddings=True)[0].tolist()

    return AnalyzeResponse(
        text=text,
        tokens=tokens,
        entities=sorted(entities, key=lambda e: e.start),
        negation_scopes=negation_scopes,
        intent_signals=intent_signals,
        morphology_map=morph_map,
        embedding=embedding,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "gliner_labels": _GLINER_LABELS}


@app.post("/reload-labels")
def reload_labels() -> dict:
    """Hot-reload GLiNER labels from config without restarting."""
    global _GLINER_LABELS
    _GLINER_LABELS = _load_gliner_labels()
    return {"status": "ok", "labels": _GLINER_LABELS}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=1)
