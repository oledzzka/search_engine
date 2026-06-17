package filters

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

// ─── filter config schema (one YAML file per filter) ─────────────────────────

type FilterConfig struct {
	Name        string   `yaml:"name"`
	Field       string   `yaml:"field"`        // target field in OpenSearch/Qdrant
	Type        string   `yaml:"type"`         // numeric | categorical | date | boolean | money
	Labels      []string `yaml:"gliner_labels"` // NER labels that map to this filter
	NatashaLabels []string `yaml:"natasha_labels"` // natasha built-in labels (MONEY DATE PER …)

	// Confidence threshold below which the condition is dropped (not applied)
	MinConfidence float64 `yaml:"min_confidence"`

	// What to do when confidence is between min_confidence and hard_threshold:
	// "filter" (hard) | "boost" (soft) | "drop"
	LowConfidenceAction string `yaml:"low_confidence_action"`
	HardThreshold       float64 `yaml:"hard_threshold"`

	// For numeric / money filters: how to interpret direction words
	// "до X" → lte, "от X" → gte, "X-Y" → between
	DirectionWords DirectionWords `yaml:"direction_words"`

	// For money: currency normalisation
	CurrencyAliases map[string]string `yaml:"currency_aliases"`

	// For categorical: lemma → canonical value mapping
	ValueMapping map[string]string `yaml:"value_mapping"`

	// For boolean flags: phrase → field+value
	BoolPhrases map[string]BoolPhrase `yaml:"bool_phrases"`

	// For intents: name → action
	IntentMapping map[string]IntentMapping `yaml:"intent_mapping"`
}

type DirectionWords struct {
	Lte []string `yaml:"lte"` // "до", "не более", "максимум", "max"
	Gte []string `yaml:"gte"` // "от", "не менее", "минимум", "min"
	Lt  []string `yaml:"lt"`  // "меньше", "ниже"
	Gt  []string `yaml:"gt"`  // "больше", "выше", "свыше"
}

type BoolPhrase struct {
	Field string `yaml:"field"`
	Value bool   `yaml:"value"`
}

type IntentMapping struct {
	Action     string `yaml:"action"`      // filter | boost | sort
	SortField  string `yaml:"sort_field"`
	SortDesc   bool   `yaml:"sort_desc"`
	BoostField string `yaml:"boost_field"`
}

// ─── registry ────────────────────────────────────────────────────────────────

// Registry holds all compiled filter configs indexed by NER label.
type Registry struct {
	configs  []*FilterConfig
	byLabel  map[string]*FilterConfig // NER label → config
	byIntent map[string]*FilterConfig // intent name → config
}

func LoadRegistry(configDir string) (*Registry, error) {
	files, err := filepath.Glob(filepath.Join(configDir, "*.yaml"))
	if err != nil {
		return nil, err
	}

	r := &Registry{
		byLabel:  make(map[string]*FilterConfig),
		byIntent: make(map[string]*FilterConfig),
	}

	for _, f := range files {
		data, err := os.ReadFile(f)
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", f, err)
		}
		var cfg FilterConfig
		if err := yaml.Unmarshal(data, &cfg); err != nil {
			return nil, fmt.Errorf("parse %s: %w", f, err)
		}
		if cfg.MinConfidence == 0 {
			cfg.MinConfidence = 0.60
		}
		if cfg.HardThreshold == 0 {
			cfg.HardThreshold = 0.75
		}
		if cfg.LowConfidenceAction == "" {
			cfg.LowConfidenceAction = "boost"
		}

		r.configs = append(r.configs, &cfg)
		for _, lbl := range cfg.Labels {
			r.byLabel[lbl] = &cfg
		}
		for _, lbl := range cfg.NatashaLabels {
			r.byLabel[lbl] = &cfg
		}
		for intentName := range cfg.IntentMapping {
			r.byIntent[intentName] = &cfg
		}
	}
	return r, nil
}

func (r *Registry) ByLabel(label string) (*FilterConfig, bool) {
	c, ok := r.byLabel[label]
	return c, ok
}

func (r *Registry) ByIntent(name string) (*FilterConfig, bool) {
	c, ok := r.byIntent[name]
	return c, ok
}
