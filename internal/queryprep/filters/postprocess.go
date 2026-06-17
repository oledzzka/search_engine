package filters

import "strings"

// PostProcess applies transformations to the raw FilterSet after recognition:
//  1. Merge same-field categorical conditions joined by OR conjunction
//  2. Deduplicate identical conditions
//  3. Resolve conflicting numeric ranges (keep tightest bounds)
func PostProcess(fs FilterSet, originalText string) FilterSet {
	fs.Categorical = mergeOrCategoricals(fs.Categorical, originalText)
	fs.Categorical = deduplicateCategoricals(fs.Categorical)
	fs.Numeric = resolveNumericConflicts(fs.Numeric)
	return fs
}

// mergeOrCategoricals detects "красный или синий" patterns in the original text
// and collapses same-field categoricals that appear within an OR window into
// a single OpOr condition.
func mergeOrCategoricals(conds []CategoricalCondition, text string) []CategoricalCondition {
	lower := strings.ToLower(text)
	orWords := []string{" или ", " or ", "/"}

	// group by field
	byField := make(map[string][]CategoricalCondition)
	for _, c := range conds {
		byField[c.Field] = append(byField[c.Field], c)
	}

	var out []CategoricalCondition
	for field, group := range byField {
		if len(group) == 1 {
			out = append(out, group[0])
			continue
		}

		// check if any OR word appears between consecutive values in the text
		mergedValues := []string{group[0].Values[0]}
		minConf := group[0].Confidence

		for i := 1; i < len(group); i++ {
			prev := group[i-1].Values[0]
			curr := group[i].Values[0]

			// find the window between prev value occurrence and curr
			prevIdx := strings.Index(lower, strings.ToLower(prev))
			currIdx := strings.Index(lower, strings.ToLower(curr))
			if prevIdx < 0 || currIdx < 0 {
				// can't locate — treat as separate
				out = append(out, group[i-1])
				continue
			}
			window := lower[prevIdx:currIdx]

			hasOr := false
			for _, ow := range orWords {
				if strings.Contains(window, ow) {
					hasOr = true
					break
				}
			}
			if hasOr {
				mergedValues = append(mergedValues, curr)
				if group[i].Confidence < minConf {
					minConf = group[i].Confidence
				}
			} else {
				// flush current group, start new
				out = append(out, CategoricalCondition{
					Field:      field,
					Values:     mergedValues,
					Op:         opForValues(mergedValues),
					Confidence: minConf,
				})
				mergedValues = []string{curr}
				minConf = group[i].Confidence
			}
		}
		out = append(out, CategoricalCondition{
			Field:      field,
			Values:     mergedValues,
			Op:         opForValues(mergedValues),
			Confidence: minConf,
		})
	}
	return out
}

func opForValues(vals []string) Op {
	if len(vals) > 1 {
		return OpOr
	}
	return OpEq
}

func deduplicateCategoricals(conds []CategoricalCondition) []CategoricalCondition {
	seen := make(map[string]bool)
	var out []CategoricalCondition
	for _, c := range conds {
		key := c.Field + ":" + strings.Join(c.Values, ",")
		if !seen[key] {
			seen[key] = true
			out = append(out, c)
		}
	}
	return out
}

// resolveNumericConflicts ensures a single field doesn't have contradictory
// bounds. If there are two Lte conditions on "price", keep the lower (tighter).
// If there are two Gte conditions, keep the higher (tighter).
func resolveNumericConflicts(conds []NumericCondition) []NumericCondition {
	type key struct {
		field string
		op    Op
	}
	best := make(map[key]NumericCondition)
	for _, c := range conds {
		k := key{c.Field, c.Op}
		prev, ok := best[k]
		if !ok {
			best[k] = c
			continue
		}
		switch c.Op {
		case OpLte, OpLt: // keep lower (tighter ceiling)
			if c.Value < prev.Value {
				best[k] = c
			}
		case OpGte, OpGt: // keep higher (tighter floor)
			if c.Value > prev.Value {
				best[k] = c
			}
		}
	}

	out := make([]NumericCondition, 0, len(best))
	for _, c := range best {
		out = append(out, c)
	}
	return out
}
