# Vibe Coding Prompt — Hybrid Russian Search Engine

## What We Are Building

A production-grade hybrid search engine where:
- **OpenSearch** handles lexical BM25 search
- **Qdrant** handles dense vector semantic search
- **RRF fusion** merges results from both
- **Primary language is Russian**
- Everything is written in **Go** (backend) + **Python** (NLP sidecar)

The core insight: neither BM25 nor vector search alone is best. We run both in parallel and fuse results using Reciprocal Rank Fusion: `score = Σ 1/(60 + rank_i)`. Documents appearing in both result sets beat documents appearing in only one.

---

## Current State of the Codebase

```
search_engine/
├── SYSTEM_PROMPT.md              ← full architecture reference
├── config/filters/               ← YAML filter definitions
│   ├── price.yaml
│   ├── color.yaml
│   ├── availability.yaml
│   └── date.yaml
├── nlp_sidecar/                  ← Python FastAPI NLP service
│   ├── main.py                   ← DONE: full Russian NLP pipeline
│   ├── requirements.txt
│   └── Dockerfile
└── internal/
    └── queryprep/
        ├── nlp/
        │   ├── client.go         ← DONE: Go HTTP client for sidecar
        │   └── types.go          ← DONE: AnalyzeResponse, Token, Entity, ...
        └── filters/
            ├── types.go          ← DONE: FilterSet, Condition, Op, ...
            ├── config.go         ← DONE: YAML registry loader
            ├── recognizer.go     ← DONE: NLP output → FilterSet
            └── postprocess.go    ← DONE: OR merge, dedup, conflict resolve
```

---

## The NLP Pipeline (Python sidecar, port 8001)

Stack: `razdel` → `pymorphy3` → `natasha` → `GLiNER` → `sentence-transformers`

```
POST /analyze
{ "text": "красные кроссовки Nike до 5000 рублей не кожаные размер 42" }

→ returns:
  tokens[]          razdel tokens + pymorphy3 morphology (lemma, POS, case, gender)
  entities[]        natasha NER (MONEY/DATE/PER/LOC/ORG) + GLiNER domain NER (ЦВЕТ/БРЕНД/РАЗМЕР)
  negation_scopes[] what "не/без/кроме" governs in the dep tree
  intent_signals[]  implicit intents (PRICE_LOW, SORT_DATE_DESC, IN_STOCK, ...)
  morphology_map{}  every word form → lemma
```

Key Russian rule: **always use lemma, never raw text** for lookup.
`"красных"` `"красные"` `"красной"` → all lemmatise to `"красный"`

---

## The Filter System (Go)

Config-driven. Each filter type = one YAML file in `/config/filters/`.
To add a new filter: drop a YAML file, call `POST /reload-labels`. Zero code changes.

```yaml
# example: config/filters/price.yaml
name: price
field: price
type: money
min_confidence: 0.70
hard_threshold: 0.80
low_confidence_action: boost
natasha_labels: ["MONEY"]
gliner_labels: ["цена", "стоимость", "максимальная цена"]
direction_words:
  lte: ["до", "не более", "максимум", "не дороже"]
  gte: ["от", "не менее", "минимум", "не дешевле"]
currency_aliases:
  "руб": "RUB"
  "₽":   "RUB"
intent_mapping:
  PRICE_LOW:
    action: boost
    boost_field: price_tier_budget
```

Recognizer output:
```go
FilterSet{
  Numeric:      [{field:price, op:lte, value:5000, unit:RUB}],
  Categorical:  [{field:color, values:["красный"], op:eq}],
  MustNot:      [{field:material, value:"кожаный"}],   // "не кожаные"
  Boolean:      [{field:available, value:true}],
  Intents:      [{name:PRICE_LOW, action:boost}],
  SemanticText: "кроссовка",   // lemmatised, all filter tokens removed
}
```

Confidence rule: **boost rather than hard-filter when unsure**. Over-filtering (zero results) is always worse than under-filtering.

---

## What Is NOT Built Yet — Your Task

Everything below needs to be written. Work in this order, it's the natural dependency chain:

### 1. `go.mod` + project bootstrap
```
module github.com/oledzzka/search_engine
go 1.22
```
Dependencies needed: `opensearch-go`, `qdrant-go-client`, `chi` (router), `viper` (config), `yaml.v3`, `zap` (logging)

### 2. `internal/config/config.go`
Typed config struct loaded from env vars + YAML. Fields:
- OpenSearch: address, index name, username/password
- Qdrant: address, collection name, vector size
- NLP sidecar: address, timeout
- Server: port, read/write timeouts
- Filter config dir path

### 3. `internal/queryprep/embedder/client.go`
HTTP client that calls the NLP sidecar's `/analyze` with `include_embeddings: true` to get a sentence vector (`[]float32`) for the SemanticText. This vector goes into Qdrant.

### 4. `internal/search/opensearch/client.go`
Takes `FilterSet` → builds OpenSearch BM25 query DSL → returns `[]ScoredDoc`.

OpenSearch query shape:
```json
{
  "query": {
    "bool": {
      "must": [{
        "multi_match": {
          "query": "<SemanticText>",
          "fields": ["title^3", "description^1", "tags^2"],
          "fuzziness": "AUTO"
        }
      }],
      "filter": [
        { "range":  { "price": { "lte": 5000 } } },
        { "term":   { "color": "красный" } },
        { "term":   { "available": true } }
      ],
      "must_not": [
        { "term": { "material": "кожаный" } }
      ]
    }
  }
}
```
Map `FilterSet.Numeric` → range clauses, `FilterSet.Categorical` → term clauses (OpOr → terms array), `FilterSet.Boolean` → term clauses, `FilterSet.MustNot` → must_not.

### 5. `internal/search/qdrant/client.go`
Takes `FilterSet` + `[]float32` vector → builds Qdrant SearchPoints → returns `[]ScoredDoc`.

Qdrant filter mirrors OpenSearch filter exactly — same `FilterSet`, different syntax:
```go
// Numeric lte → qdrant.Range{Lte: &val}
// Categorical eq → qdrant.FieldCondition with Match
// MustNot → qdrant.Filter{MustNot: [...]}
```

### 6. `internal/search/fusion/rrf.go`
```go
func Fuse(results ...[]ScoredDoc) []ScoredDoc {
    // for each doc across all result lists:
    //   score += 1.0 / (60.0 + float64(rank))
    // sort descending by fused score
    // deduplicate by doc ID
}
```
`ScoredDoc` needs: `ID string`, `Score float64`, `Payload map[string]any`

### 7. `internal/search/orchestrator.go`
Fan-out to OpenSearch + Qdrant in parallel using goroutines + `sync.WaitGroup`, collect results, call RRF, return fused list.

```go
func (o *Orchestrator) Search(ctx context.Context, req SearchRequest) ([]ScoredDoc, error) {
    // 1. call NLP sidecar → AnalyzeResponse
    // 2. call Recognizer → FilterSet
    // 3. call embedder → []float32 vector from FilterSet.SemanticText
    // 4. fan-out: OpenSearch(FilterSet) || Qdrant(FilterSet, vector)
    // 5. RRF fusion
    // 6. return
}
```

### 8. `internal/api/handler/search.go` + router
`POST /search` → JSON body `{query: string, limit: int}` → JSON response `{results: [...], took_ms: int}`
`GET /health` → `{status: ok}`

### 9. `cmd/server/main.go`
Wire everything together: load config, build clients, build orchestrator, start HTTP server.

### 10. `docker-compose.yml`
Services:
- `opensearch` (opensearchproject/opensearch:2)
- `qdrant` (qdrant/qdrant:latest)
- `nlp-sidecar` (built from `./nlp_sidecar/Dockerfile`)
- `app` (built from Go binary)

### 11. `internal/indexing/pipeline.go`
Index a document into both OpenSearch and Qdrant simultaneously:
- OpenSearch: index JSON document with all fields
- Qdrant: upsert point with vector (from embedder) + payload (all filterable fields)

### 12. `cmd/indexer/main.go`
CLI that reads JSON lines from stdin and runs them through the indexing pipeline.

---

## Shared Types to Define Early

```go
// internal/model/document.go
type Document struct {
    ID          string
    Title       string
    Description string
    Tags        []string
    Price       float64
    Currency    string
    Color       string
    Brand       string
    Material    string
    Size        float64
    Available   bool
    OnSale      bool
    FreeShip    bool
    CreatedAt   time.Time
    Payload     map[string]any  // arbitrary extra fields
}

type ScoredDoc struct {
    ID      string
    Score   float64
    Doc     Document
    Source  string  // "opensearch" | "qdrant" | "fused"
}

type SearchRequest struct {
    Query  string
    Limit  int
    Offset int
}
```

---

## Russian Gotchas to Keep in Mind

1. **Lemmatise before any string comparison.** `"красных" != "красный"` but they mean the same.
2. **Negation flips the operator.** `"не дороже 5000"` → `price.lte=5000` (not must_not price).
3. **Direction word detection uses lookback window** (20 chars before entity), not token distance.
4. **Implicit intents are boosts, not hard filters.** `"недорогой"` → boost cheap docs, don't filter by price.
5. **OR conjunctions.** `"красный или синий"` → `color IN [красный, синий]`, not two separate filters.
6. **Mixed Cyrillic/Latin.** Brand names often appear in Latin inside Russian queries. GLiNER handles this.
7. **Tысяч multiplier.** `"до 50 тысяч"` = 50,000, not 50.

---

## Vibe

- **Go is the backbone.** Fast, typed, concurrent. Use goroutines freely for fan-out.
- **Python sidecar is NLP-only.** It does no business logic. Keep it thin.
- **Config files, not code, define filters.** The recognizer is a generic engine.
- **Fuse first, re-rank later.** RRF is the baseline. Cross-encoder re-ranker is a future phase.
- **Boost over filter when uncertain.** Zero results is always the worst outcome.
- **Lemmas everywhere in Russian.** Raw text matching is wrong.

---

## Reference: Full Query Example

```
Input:  "красные кроссовки Nike до 5000 рублей не кожаные размер 42 в наличии"

NLP:
  entities:  ЦВЕТ=красный(0.89), БРЕНД=Nike(0.91), MONEY=5000руб(0.96),
             РАЗМЕР=42(0.84), НАЛИЧИЕ=в наличии(0.92)
  negation:  "не" governs → "кожаный"

FilterSet:
  Numeric:     price.lte=5000(RUB)
  Categorical: color=красный, brand=Nike
  Numeric:     size.eq=42
  Boolean:     available=true
  MustNot:     material=кожаный
  SemanticText: "кроссовка"

OpenSearch query:
  multi_match("кроссовка", fields=[title^3, description, tags^2])
  filter: price<=5000, color=красный, brand=Nike, size=42, available=true
  must_not: material=кожаный

Qdrant query:
  vector: embed("кроссовка")
  filter: price<=5000 AND color=красный AND brand=Nike AND size=42
          AND available=true AND NOT material=кожаный

RRF:
  merge OpenSearch top-20 + Qdrant top-20
  score = 1/(60+rank_os) + 1/(60+rank_qd)
  return top-10
```
