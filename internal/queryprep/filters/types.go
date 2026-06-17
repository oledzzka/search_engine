package filters

import "time"

// ─── operator ────────────────────────────────────────────────────────────────

type Op string

const (
	OpEq      Op = "eq"
	OpLt      Op = "lt"
	OpLte     Op = "lte"
	OpGt      Op = "gt"
	OpGte     Op = "gte"
	OpBetween Op = "between"
	OpMustNot Op = "must_not"
	OpOr      Op = "or" // multi-value OR (e.g. "красный или синий")
)

// ─── individual condition ─────────────────────────────────────────────────────

// Condition is one resolved filter condition ready for query builders.
type Condition struct {
	Field      string
	Op         Op
	Value      any        // string | float64 | bool | time.Time
	ValueTo    any        // populated for OpBetween
	Confidence float64
	Source     string     // "entity:natasha" | "entity:gliner" | "intent" | "negation"
}

// ─── intent action ───────────────────────────────────────────────────────────

type IntentAction string

const (
	ActionFilter IntentAction = "filter" // hard filter
	ActionBoost  IntentAction = "boost"  // soft boost, don't filter
	ActionSort   IntentAction = "sort"   // change sort order
)

type IntentCondition struct {
	Name       string
	Action     IntentAction
	SortField  string // populated when Action == ActionSort
	SortDesc   bool
	BoostField string // populated when Action == ActionBoost
	Confidence float64
}

// ─── filter set ──────────────────────────────────────────────────────────────

// FilterSet is the fully resolved output of the recognizer.
// Query builders (OpenSearch / Qdrant) consume this directly.
type FilterSet struct {
	// Hard filters — directly applied as query constraints
	Numeric    []NumericCondition
	Categorical []CategoricalCondition
	Date       []DateCondition
	Boolean    []BoolCondition
	MustNot    []Condition

	// Soft signals — applied as scoring hints, not hard constraints
	Intents    []IntentCondition

	// Semantic text remaining after filter extraction
	SemanticText string

	// Low-confidence extractions that were dropped
	Dropped []DroppedCondition
}

type NumericCondition struct {
	Field    string
	Op       Op
	Value    float64
	ValueTo  float64 // for OpBetween
	Unit     string  // "RUB" "USD" "kg" "cm" …
	Confidence float64
}

type CategoricalCondition struct {
	Field      string
	Values     []string // >1 when Op==OpOr
	Op         Op       // OpEq | OpOr
	Confidence float64
}

type DateCondition struct {
	Field      string
	Op         Op
	Value      time.Time
	ValueTo    time.Time // for OpBetween
	Confidence float64
}

type BoolCondition struct {
	Field      string
	Value      bool
	Confidence float64
}

type DroppedCondition struct {
	RawText    string
	Label      string
	Confidence float64
	Reason     string
}
