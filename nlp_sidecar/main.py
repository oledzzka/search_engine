import spacy
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

nlp = spacy.load("en_core_web_sm")

app = FastAPI(title="NLP Sidecar")


# ── request / response models ────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str
    include_embeddings: bool = False  # sentence-level embedding (slow)


class Token(BaseModel):
    i: int           # token index in doc
    text: str
    lemma: str
    pos: str         # NOUN, VERB, ADJ, NUM, PUNCT …
    tag: str         # fine-grained POS
    dep: str         # dependency relation to head
    head_i: int      # index of syntactic head token
    is_stop: bool
    is_punct: bool
    start: int       # char offset start
    end: int         # char offset end


class Entity(BaseModel):
    text: str
    label: str       # spaCy built-in: MONEY, DATE, CARDINAL, ORG, GPE …
                     # + custom: PRICE_MAX, PRICE_MIN, COLOR, BRAND …
    start: int       # char offset
    end: int
    token_start: int # token index start
    token_end: int   # token index end (exclusive)
    confidence: float


class NegationScope(BaseModel):
    neg_token_i: int        # index of "not/no/never/-" token
    scope_token_indices: list[int]  # tokens under negation scope


class IntentSignal(BaseModel):
    name: str        # PRICE_LOW, PRICE_HIGH, SORT_DATE_DESC, GEO_NEARBY …
    confidence: float
    source_tokens: list[int]


class AnalyzeResponse(BaseModel):
    text: str
    tokens: list[Token]
    entities: list[Entity]
    negation_scopes: list[NegationScope]
    intent_signals: list[IntentSignal]
    embedding: Optional[list[float]] = None  # sentence vector if requested


# ── intent anchors ───────────────────────────────────────────────────────────
# each entry: (intent_name, seed_words, threshold)
INTENT_ANCHORS = [
    ("PRICE_LOW",       ["affordable", "cheap", "budget", "inexpensive", "low-cost", "economical"], 0.55),
    ("PRICE_HIGH",      ["premium", "luxury", "expensive", "high-end", "exclusive", "professional"], 0.55),
    ("SORT_DATE_DESC",  ["latest", "newest", "recent", "new", "fresh"], 0.60),
    ("SORT_POPULAR",    ["popular", "trending", "bestseller", "top-rated", "best"], 0.60),
    ("GEO_NEARBY",      ["nearby", "local", "near me", "close"], 0.65),
    ("CONDITION_NEW",   ["brand-new", "new", "sealed", "unopened"], 0.70),
    ("CONDITION_USED",  ["used", "second-hand", "pre-owned", "refurbished"], 0.65),
]

# precompute average vectors for each anchor group
_anchor_vectors: list[tuple[str, np.ndarray, float]] = []

def _build_anchors() -> None:
    for name, seeds, threshold in INTENT_ANCHORS:
        vecs = [nlp(w).vector for w in seeds if nlp(w).has_vector]
        if vecs:
            avg = np.mean(vecs, axis=0)
            norm = np.linalg.norm(avg)
            if norm > 0:
                avg = avg / norm
            _anchor_vectors.append((name, avg, threshold))

_build_anchors()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ── negation scope resolver ──────────────────────────────────────────────────

def _resolve_negation_scopes(doc: spacy.tokens.Doc) -> list[NegationScope]:
    """
    Walk the dependency tree to find what each negation token governs.
    neg → head → collect head + all its right children (the scope).
    Also handles: "-word" prefix patterns.
    """
    scopes: list[NegationScope] = []

    for token in doc:
        # syntactic negation: dep == "neg"
        if token.dep_ == "neg":
            head = token.head
            scope_indices = [head.i] + [t.i for t in head.subtree if t.i != token.i]
            scopes.append(NegationScope(
                neg_token_i=token.i,
                scope_token_indices=sorted(scope_indices),
            ))

        # prefix negation: token text starts with "-" and is followed by a word
        if token.text.startswith("-") and len(token.text) > 1:
            scopes.append(NegationScope(
                neg_token_i=token.i,
                scope_token_indices=[token.i],
            ))

    return scopes


# ── intent detection ─────────────────────────────────────────────────────────

def _detect_intents(doc: spacy.tokens.Doc) -> list[IntentSignal]:
    signals: list[IntentSignal] = []

    for token in doc:
        if not token.has_vector or token.is_stop or token.is_punct:
            continue

        tv = token.vector / (np.linalg.norm(token.vector) + 1e-9)
        for name, anchor_vec, threshold in _anchor_vectors:
            score = _cosine(tv, anchor_vec)
            if score >= threshold:
                signals.append(IntentSignal(
                    name=name,
                    confidence=round(score, 4),
                    source_tokens=[token.i],
                ))

    # deduplicate: keep highest confidence per intent name
    best: dict[str, IntentSignal] = {}
    for s in signals:
        if s.name not in best or s.confidence > best[s.name].confidence:
            best[s.name] = s

    return sorted(best.values(), key=lambda x: -x.confidence)


# ── entity confidence heuristic ──────────────────────────────────────────────
# spaCy NER doesn't expose per-entity confidence in the default pipeline.
# We use a simple heuristic: entities with longer spans and known labels
# get higher base confidence.
_LABEL_BASE_CONFIDENCE = {
    "MONEY":    0.92,
    "DATE":     0.88,
    "CARDINAL": 0.75,
    "ORG":      0.80,
    "GPE":      0.82,
    "PERSON":   0.78,
    "PRODUCT":  0.70,
    "QUANTITY": 0.80,
    "PERCENT":  0.85,
    "TIME":     0.85,
}

def _entity_confidence(ent: spacy.tokens.Span) -> float:
    base = _LABEL_BASE_CONFIDENCE.get(ent.label_, 0.60)
    length_bonus = min(0.05 * (len(ent) - 1), 0.10)
    return round(min(base + length_bonus, 0.99), 4)


# ── main endpoint ─────────────────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    doc = nlp(req.text)

    tokens = [
        Token(
            i=t.i,
            text=t.text,
            lemma=t.lemma_,
            pos=t.pos_,
            tag=t.tag_,
            dep=t.dep_,
            head_i=t.head.i,
            is_stop=t.is_stop,
            is_punct=t.is_punct,
            start=t.idx,
            end=t.idx + len(t.text),
        )
        for t in doc
    ]

    entities = [
        Entity(
            text=ent.text,
            label=ent.label_,
            start=ent.start_char,
            end=ent.end_char,
            token_start=ent.start,
            token_end=ent.end,
            confidence=_entity_confidence(ent),
        )
        for ent in doc.ents
    ]

    negation_scopes = _resolve_negation_scopes(doc)
    intent_signals  = _detect_intents(doc)

    embedding: Optional[list[float]] = None
    if req.include_embeddings:
        embedding = doc.vector.tolist()

    return AnalyzeResponse(
        text=req.text,
        tokens=tokens,
        entities=entities,
        negation_scopes=negation_scopes,
        intent_signals=intent_signals,
        embedding=embedding,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": nlp.meta["name"]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
