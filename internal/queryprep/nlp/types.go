package nlp

// AnalyzeRequest is sent to the Russian NLP sidecar.
type AnalyzeRequest struct {
	Text              string   `json:"text"`
	IncludeEmbeddings bool     `json:"include_embeddings"`
	// GlinerLabels overrides the sidecar's config-loaded labels for this request.
	GlinerLabels      []string `json:"gliner_labels,omitempty"`
}

// AnalyzeResponse is returned by POST /analyze.
type AnalyzeResponse struct {
	Text            string               `json:"text"`
	Tokens          []Token              `json:"tokens"`
	Entities        []Entity             `json:"entities"`
	NegationScopes  []NegationScope      `json:"negation_scopes"`
	IntentSignals   []IntentSignal       `json:"intent_signals"`
	MorphologyMap   map[string]MorphInfo `json:"morphology_map"` // original_form → morph
	Embedding       []float32            `json:"embedding,omitempty"`
}

// MorphInfo holds pymorphy3 morphological analysis for one word form.
type MorphInfo struct {
	Lemma     string   `json:"lemma"`
	POS       string   `json:"pos"`    // NOUN ADJF VERB NUMR ADVB PRTF …
	Case      string   `json:"case"`   // Nom Gent Datv Accs Ablt Loct
	Number    string   `json:"number"` // Sing Plur
	Gender    string   `json:"gender"` // Masc Femn Neut
	Grammemes []string `json:"grammemes"`
}

// Token is one razdel token with pymorphy3 + natasha dep annotation.
type Token struct {
	I     int       `json:"i"`
	Text  string    `json:"text"`
	Lemma string    `json:"lemma"`
	POS   string    `json:"pos"`
	Dep   string    `json:"dep"`    // natasha dependency relation
	HeadI int       `json:"head_i"`
	Start int       `json:"start"`  // char offset
	End   int       `json:"end"`
	Morph MorphInfo `json:"morph"`
}

// Entity is a named entity from natasha (PER/LOC/ORG/MONEY/DATE)
// or a domain entity from GLiNER (ЦВЕТ/БРЕНД/РАЗМЕР/…).
type Entity struct {
	Text       string  `json:"text"`
	Lemma      string  `json:"lemma"`      // lemmatised form
	Label      string  `json:"label"`      // entity type
	Source     string  `json:"source"`     // "natasha" | "gliner"
	Start      int     `json:"start"`      // char offset
	End        int     `json:"end"`
	Confidence float64 `json:"confidence"`
}

// NegationScope describes what a negation word (не/без/кроме/…) governs.
type NegationScope struct {
	NegTokenI          int      `json:"neg_token_i"`
	NegText            string   `json:"neg_text"`
	ScopeTokenIndices  []int    `json:"scope_token_indices"`
	ScopeLemmas        []string `json:"scope_lemmas"`
}

// IntentSignal is an implicit search intent detected via embedding similarity.
// Name examples: PRICE_LOW, PRICE_HIGH, SORT_DATE_DESC, IN_STOCK, GEO_NEARBY.
type IntentSignal struct {
	Name         string  `json:"name"`
	Confidence   float64 `json:"confidence"`
	SourceTokens []int   `json:"source_tokens"`
}
