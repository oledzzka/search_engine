package filters

import (
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/oledzzka/search_engine/internal/queryprep/nlp"
)

// Recognizer converts an NLP AnalyzeResponse into a FilterSet.
type Recognizer struct {
	registry *Registry
}

func NewRecognizer(registry *Registry) *Recognizer {
	return &Recognizer{registry: registry}
}

// Recognize is the main entry point.
func (r *Recognizer) Recognize(resp *nlp.AnalyzeResponse) FilterSet {
	fs := FilterSet{}

	// build the set of token indices that are under negation scope
	negatedIndices := r.negatedIndexSet(resp.NegationScopes)

	// track which char spans are consumed by a filter (for semantic text)
	consumed := make([]bool, len(resp.Text))

	// ── 1. process entities ───────────────────────────────────────────────
	for _, ent := range resp.Entities {
		cfg, ok := r.registry.ByLabel(ent.Label)
		if !ok {
			continue
		}

		// check if entity is under negation scope
		inNegScope := r.entityInNegScope(ent, negatedIndices, resp.Tokens)

		// confidence gate
		if ent.Confidence < cfg.MinConfidence {
			fs.Dropped = append(fs.Dropped, DroppedCondition{
				RawText:    ent.Text,
				Label:      ent.Label,
				Confidence: ent.Confidence,
				Reason:     "below min_confidence",
			})
			continue
		}

		// dispatch by filter type
		switch cfg.Type {
		case "money", "numeric":
			r.resolveNumeric(cfg, ent, resp, inNegScope, &fs)
		case "categorical":
			r.resolveCategorical(cfg, ent, inNegScope, ent.Confidence >= cfg.HardThreshold, &fs)
		case "date":
			r.resolveDate(cfg, ent, inNegScope, &fs)
		case "boolean":
			r.resolveBoolean(cfg, ent, inNegScope, &fs)
		}

		// mark entity span as consumed
		markConsumed(consumed, ent.Start, ent.End)
	}

	// ── 2. process intent signals ─────────────────────────────────────────
	for _, sig := range resp.IntentSignals {
		cfg, ok := r.registry.ByIntent(sig.Name)
		if !ok {
			continue
		}
		im, ok := cfg.IntentMapping[sig.Name]
		if !ok {
			continue
		}
		if sig.Confidence < cfg.MinConfidence {
			continue
		}
		fs.Intents = append(fs.Intents, IntentCondition{
			Name:       sig.Name,
			Action:     IntentAction(im.Action),
			SortField:  im.SortField,
			SortDesc:   im.SortDesc,
			BoostField: im.BoostField,
			Confidence: sig.Confidence,
		})
	}

	// ── 3. build semantic text from unconsumed tokens ─────────────────────
	fs.SemanticText = r.buildSemanticText(resp.Text, resp.Tokens, consumed)

	return fs
}

// ── numeric / money ───────────────────────────────────────────────────────────

var (
	reAmount   = regexp.MustCompile(`[\d\s,.]+`)
	reCurrency = regexp.MustCompile(`(?i)(руб|рублей|рублёй|₽|usd|доллар|€|евро|тыс|тысяч|k)`)
	reRange    = regexp.MustCompile(`(?i)(от|from)`)
	reCeiling  = regexp.MustCompile(`(?i)(до|не более|максимум|max|не дороже)`)
	reFloor    = regexp.MustCompile(`(?i)(от|не менее|минимум|min|не дешевле|от)`)
)

func (r *Recognizer) resolveNumeric(
	cfg *FilterConfig, ent nlp.Entity,
	resp *nlp.AnalyzeResponse,
	inNegScope bool, fs *FilterSet,
) {
	// look at the token immediately before the entity to determine direction
	op := r.detectDirection(cfg, ent, resp.Tokens, resp.Text)
	amount, unit := parseAmount(ent.Text, cfg.CurrencyAliases)
	if amount == 0 {
		return
	}

	if inNegScope {
		// "не до 5000" → flip the operator
		op = flipOp(op)
	}

	// check if there's a range pattern — look for paired entity or "от X до Y"
	if paired, ok := r.findPairedNumeric(ent, resp, cfg); ok {
		lo, hi := amount, paired
		if lo > hi {
			lo, hi = hi, lo
		}
		fs.Numeric = append(fs.Numeric, NumericCondition{
			Field: cfg.Field, Op: OpBetween,
			Value: lo, ValueTo: hi,
			Unit: unit, Confidence: ent.Confidence,
		})
		return
	}

	fs.Numeric = append(fs.Numeric, NumericCondition{
		Field: cfg.Field, Op: op,
		Value: amount, Unit: unit,
		Confidence: ent.Confidence,
	})
}

// detectDirection reads the token(s) before the entity span to determine
// whether this is a ceiling (lte) or floor (gte) constraint.
// Falls back to OpLte for money (most queries mean "under X").
func (r *Recognizer) detectDirection(cfg *FilterConfig, ent nlp.Entity, tokens []nlp.Token, text string) Op {
	// look at up to 3 chars before entity start for a direction word
	lookback := ""
	if ent.Start >= 3 {
		lookback = strings.ToLower(text[max(0, ent.Start-20):ent.Start])
	}

	for _, w := range cfg.DirectionWords.Lte {
		if strings.Contains(lookback, strings.ToLower(w)) {
			return OpLte
		}
	}
	for _, w := range cfg.DirectionWords.Gte {
		if strings.Contains(lookback, strings.ToLower(w)) {
			return OpGte
		}
	}
	for _, w := range cfg.DirectionWords.Lt {
		if strings.Contains(lookback, strings.ToLower(w)) {
			return OpLt
		}
	}
	for _, w := range cfg.DirectionWords.Gt {
		if strings.Contains(lookback, strings.ToLower(w)) {
			return OpGt
		}
	}

	// default: money without direction word means ceiling ("5000 рублей кроссовки")
	if cfg.Type == "money" {
		return OpLte
	}
	return OpEq
}

// findPairedNumeric looks for a second MONEY/NUMERIC entity close to ent,
// indicating a range like "от 1000 до 5000".
func (r *Recognizer) findPairedNumeric(ent nlp.Entity, resp *nlp.AnalyzeResponse, cfg *FilterConfig) (float64, bool) {
	text := strings.ToLower(resp.Text[max(0, ent.Start-30):min(len(resp.Text), ent.End+30)])
	if !reRange.MatchString(text) {
		return 0, false
	}
	for _, other := range resp.Entities {
		if other.Start == ent.Start {
			continue
		}
		if otherCfg, ok := r.registry.ByLabel(other.Label); ok && otherCfg.Field == cfg.Field {
			v, _ := parseAmount(other.Text, cfg.CurrencyAliases)
			return v, v > 0
		}
	}
	return 0, false
}

// ── categorical ───────────────────────────────────────────────────────────────

func (r *Recognizer) resolveCategorical(
	cfg *FilterConfig, ent nlp.Entity,
	inNegScope bool, isHard bool, fs *FilterSet,
) {
	val := canonicalValue(ent.Lemma, cfg.ValueMapping)

	if inNegScope {
		fs.MustNot = append(fs.MustNot, Condition{
			Field: cfg.Field, Op: OpMustNot, Value: val,
			Confidence: ent.Confidence, Source: "negation:" + ent.Source,
		})
		return
	}

	// low confidence → boost only
	if !isHard {
		fs.Intents = append(fs.Intents, IntentCondition{
			Name:       "BOOST_" + strings.ToUpper(cfg.Field),
			Action:     ActionBoost,
			BoostField: cfg.Field,
			Confidence: ent.Confidence,
		})
		return
	}

	// check if OR conjunction precedes: "красный или синий"
	// (handled by the caller merging consecutive same-field categoricals)
	fs.Categorical = append(fs.Categorical, CategoricalCondition{
		Field:      cfg.Field,
		Values:     []string{val},
		Op:         OpEq,
		Confidence: ent.Confidence,
	})
}

// ── date ──────────────────────────────────────────────────────────────────────

var (
	reYear    = regexp.MustCompile(`\b(20\d{2}|19\d{2})\b`)
	reRelDate = regexp.MustCompile(`(?i)(последни[йеяхм]|прошл[аяое]?|за последни[ейх])\s+(\d+)?\s*(день|дня|дней|недел[юи]|недель|месяц|месяца|месяцев|год|года|лет)`)
)

func (r *Recognizer) resolveDate(cfg *FilterConfig, ent nlp.Entity, inNegScope bool, fs *FilterSet) {
	t, op := parseDate(ent.Text)
	if t.IsZero() {
		return
	}
	if inNegScope {
		op = flipOp(op)
	}
	fs.Date = append(fs.Date, DateCondition{
		Field: cfg.Field, Op: op, Value: t, Confidence: ent.Confidence,
	})
}

// ── boolean ───────────────────────────────────────────────────────────────────

func (r *Recognizer) resolveBoolean(cfg *FilterConfig, ent nlp.Entity, inNegScope bool, fs *FilterSet) {
	// match entity text against bool_phrases
	lower := strings.ToLower(strings.TrimSpace(ent.Text))
	for phrase, bp := range cfg.BoolPhrases {
		if strings.Contains(lower, phrase) {
			val := bp.Value
			if inNegScope {
				val = !val
			}
			fs.Boolean = append(fs.Boolean, BoolCondition{
				Field: bp.Field, Value: val, Confidence: ent.Confidence,
			})
			return
		}
	}
}

// ── negation helpers ──────────────────────────────────────────────────────────

func (r *Recognizer) negatedIndexSet(scopes []nlp.NegationScope) map[int]bool {
	out := make(map[int]bool)
	for _, s := range scopes {
		for _, i := range s.ScopeTokenIndices {
			out[i] = true
		}
	}
	return out
}

func (r *Recognizer) entityInNegScope(ent nlp.Entity, negated map[int]bool, tokens []nlp.Token) bool {
	for _, t := range tokens {
		if t.Start >= ent.Start && t.End <= ent.End {
			if negated[t.I] {
				return true
			}
		}
	}
	return false
}

// ── semantic text builder ─────────────────────────────────────────────────────

func (r *Recognizer) buildSemanticText(text string, tokens []nlp.Token, consumed []bool) string {
	// also strip negation markers from semantic text
	var parts []string
	for _, t := range tokens {
		if t.IsPunct {
			continue
		}
		if anyConsumed(consumed, t.Start, t.End) {
			continue
		}
		// strip standalone negation words — they are structurally handled
		if isNegationWord(t.Lemma) {
			continue
		}
		parts = append(parts, t.Lemma) // use lemma for better embedding quality
	}
	return strings.Join(parts, " ")
}

// ── parsing utilities ─────────────────────────────────────────────────────────

func parseAmount(text string, currencyAliases map[string]string) (float64, string) {
	// extract numeric part
	digits := strings.Map(func(r rune) rune {
		if unicode.IsDigit(r) || r == '.' || r == ',' {
			return r
		}
		return ' '
	}, text)
	digits = strings.TrimSpace(digits)
	digits = strings.ReplaceAll(digits, ",", ".")
	// take the first number found
	fields := strings.Fields(digits)
	if len(fields) == 0 {
		return 0, ""
	}
	amount, err := strconv.ParseFloat(fields[0], 64)
	if err != nil {
		return 0, ""
	}

	// detect multiplier (тыс/тысяч → ×1000)
	lower := strings.ToLower(text)
	if strings.Contains(lower, "тыс") || strings.Contains(lower, "тысяч") {
		amount *= 1000
	}

	// detect currency
	unit := "RUB"
	for alias, canonical := range currencyAliases {
		if strings.Contains(lower, strings.ToLower(alias)) {
			unit = canonical
			break
		}
	}
	return amount, unit
}

func parseDate(text string) (time.Time, Op) {
	now := time.Now()
	lower := strings.ToLower(text)

	// relative: "последние 7 дней"
	m := reRelDate.FindStringSubmatch(lower)
	if m != nil {
		n := 1
		if m[2] != "" {
			n, _ = strconv.Atoi(m[2])
		}
		unit := m[3]
		var d time.Duration
		switch {
		case strings.HasPrefix(unit, "день") || strings.HasPrefix(unit, "дн"):
			d = time.Duration(n) * 24 * time.Hour
		case strings.HasPrefix(unit, "недел"):
			d = time.Duration(n) * 7 * 24 * time.Hour
		case strings.HasPrefix(unit, "месяц"):
			d = time.Duration(n) * 30 * 24 * time.Hour
		case strings.HasPrefix(unit, "год") || unit == "лет":
			d = time.Duration(n) * 365 * 24 * time.Hour
		}
		return now.Add(-d), OpGte
	}

	// absolute year
	ym := reYear.FindString(lower)
	if ym != "" {
		year, _ := strconv.Atoi(ym)
		return time.Date(year, 1, 1, 0, 0, 0, 0, time.UTC), OpGte
	}

	return time.Time{}, OpGte
}

func canonicalValue(lemma string, mapping map[string]string) string {
	if mapping != nil {
		if v, ok := mapping[lemma]; ok {
			return v
		}
	}
	return lemma
}

func flipOp(op Op) Op {
	switch op {
	case OpLt:
		return OpGte
	case OpLte:
		return OpGt
	case OpGt:
		return OpLte
	case OpGte:
		return OpLt
	default:
		return op
	}
}

func markConsumed(consumed []bool, start, end int) {
	for i := start; i < end && i < len(consumed); i++ {
		consumed[i] = true
	}
}

func anyConsumed(consumed []bool, start, end int) bool {
	for i := start; i < end && i < len(consumed); i++ {
		if consumed[i] {
			return true
		}
	}
	return false
}

func isNegationWord(lemma string) bool {
	switch lemma {
	case "не", "нет", "ни", "без", "кроме", "нельзя":
		return true
	}
	return false
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
