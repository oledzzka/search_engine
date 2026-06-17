package nlp

// AnalyzeRequest is sent to the NLP sidecar.
type AnalyzeRequest struct {
	Text               string `json:"text"`
	IncludeEmbeddings  bool   `json:"include_embeddings"`
}

// AnalyzeResponse is returned by POST /analyze.
type AnalyzeResponse struct {
	Text            string          `json:"text"`
	Tokens          []Token         `json:"tokens"`
	Entities        []Entity        `json:"entities"`
	NegationScopes  []NegationScope `json:"negation_scopes"`
	IntentSignals   []IntentSignal  `json:"intent_signals"`
	Embedding       []float32       `json:"embedding,omitempty"`
}

// Token is one spaCy token with full linguistic annotation.
type Token struct {
	I       int    `json:"i"`        // position in doc
	Text    string `json:"text"`
	Lemma   string `json:"lemma"`
	POS     string `json:"pos"`      // NOUN VERB ADJ NUM PUNCT …
	Tag     string `json:"tag"`      // fine-grained POS
	Dep     string `json:"dep"`      // dependency relation to head
	HeadI   int    `json:"head_i"`   // index of syntactic head
	IsStop  bool   `json:"is_stop"`
	IsPunct bool   `json:"is_punct"`
	Start   int    `json:"start"`    // char offset
	End     int    `json:"end"`
}

// Entity is a named entity recognised by spaCy.
// Label is the spaCy label: MONEY, DATE, CARDINAL, ORG, GPE, PRODUCT …
type Entity struct {
	Text        string  `json:"text"`
	Label       string  `json:"label"`
	Start       int     `json:"start"`       // char offset
	End         int     `json:"end"`
	TokenStart  int     `json:"token_start"` // token index
	TokenEnd    int     `json:"token_end"`   // exclusive
	Confidence  float64 `json:"confidence"`
}

// NegationScope describes what a negation token governs in the dep tree.
type NegationScope struct {
	NegTokenI          int   `json:"neg_token_i"`
	ScopeTokenIndices  []int `json:"scope_token_indices"`
}

// IntentSignal is an implicit search intent detected via embedding similarity.
// Name examples: PRICE_LOW, PRICE_HIGH, SORT_DATE_DESC, SORT_POPULAR, GEO_NEARBY.
type IntentSignal struct {
	Name         string  `json:"name"`
	Confidence   float64 `json:"confidence"`
	SourceTokens []int   `json:"source_tokens"`
}
