package server

import (
	"encoding/json"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"sparktalk/internal/llm"
	"sparktalk/internal/workflows"
)

var documentEntityKinds = []string{"person", "animal", "place", "organization", "work", "term", "unknown"}

func documentEntityTermSchema(ids []string) map[string]any {
	return map[string]any{"type": "array", "maxItems": 16, "items": map[string]any{"type": "object", "properties": map[string]any{
		"evidence": map[string]any{"type": "object", "properties": map[string]any{"id": map[string]any{"type": "string", "enum": ids}, "quote": map[string]any{"type": "string", "minLength": 1, "maxLength": 192}}, "required": []string{"id", "quote"}, "additionalProperties": false},
		"kind":     map[string]any{"type": "string", "enum": documentEntityKinds},
		"source":   map[string]any{"type": "string", "minLength": 1, "maxLength": 96},
		"target":   map[string]any{"type": "string", "minLength": 1, "maxLength": 96},
	}, "required": []string{"evidence", "kind", "source", "target"}, "additionalProperties": false}}
}

// Names remain proposals with source evidence, not user-pinned dictionary
// replacements. Same-spelling referents at different source locations are not
// automatically merged into a single identity.
func documentEntityCandidates(u workflows.DocumentUnit, result llm.StreamResult, prior []workflows.DocumentEntity) []workflows.DocumentEntity {
	var payload struct {
		Terms []struct {
			Source   string `json:"source"`
			Target   string `json:"target"`
			Kind     string `json:"kind"`
			Evidence struct {
				ID    string `json:"id"`
				Quote string `json:"quote"`
			} `json:"evidence"`
		} `json:"terms"`
	}
	if json.Unmarshal([]byte(result.Content), &payload) != nil {
		return nil
	}
	wire := map[string]documentWireEntry{}
	for _, entry := range documentWireEntries(u) {
		wire[entry.ID] = entry
	}
	existing := map[string]workflows.DocumentEntity{}
	for _, entity := range prior {
		existing[entity.ID] = entity
	}
	var entities []workflows.DocumentEntity
	seen := map[string]bool{}
	for i, term := range payload.Terms {
		if i >= 16 {
			break
		}
		kindOK := false
		for _, kind := range documentEntityKinds {
			kindOK = kindOK || term.Kind == kind
		}
		entry, ok := wire[term.Evidence.ID]
		if !ok || !kindOK || strings.TrimSpace(term.Source) == "" || strings.TrimSpace(term.Target) == "" || len([]rune(term.Source)) > 96 || len([]rune(term.Target)) > 96 {
			continue
		}
		quote, anchored := documentAnchoredQuote(entry.Text, term.Evidence.Quote)
		if !anchored || !strings.Contains(quote, term.Source) {
			continue
		}
		inputIDs := []string{}
		for _, index := range entry.Indexes {
			inputIDs = append(inputIDs, u.Inputs[index].ID)
		}
		if len(inputIDs) == 0 {
			continue
		}
		id := documentTextHash(inputIDs[0] + "\x00" + term.Source)[:24]
		if seen[id] {
			continue
		}
		seen[id] = true
		entity := workflows.DocumentEntity{ID: id, Version: 1, Source: term.Source, Target: term.Target, Kind: term.Kind, Origin: "model_candidate", Status: "candidate", Evidence: []workflows.DocumentEntityEvidence{{InputID: inputIDs[0], InputIDs: inputIDs, Quote: quote}}}
		if old, exists := existing[id]; exists {
			if old.Target == entity.Target && old.Kind == entity.Kind {
				continue
			}
			entity.Version = old.Version + 1
		}
		entities = append(entities, entity)
	}
	return entities
}

func documentRelevantEntities(u workflows.DocumentUnit, entities []workflows.DocumentEntity, tokenBudget int) ([]workflows.DocumentEntity, map[string]int) {
	var text strings.Builder
	for _, in := range u.Inputs {
		if !in.Annotation {
			text.WriteString(in.Text)
			text.WriteByte('\n')
		}
	}
	var candidates []workflows.DocumentEntity
	for _, entity := range entities {
		if (entity.Status == "reviewed" || entity.Origin == "user") && entity.Source != "" && documentEntityMention(text.String(), entity.Source) {
			candidates = append(candidates, entity)
		}
	}
	// Prefer stronger explicit provenance, then deterministic source order.
	sort.SliceStable(candidates, func(i, j int) bool {
		a, b := candidates[i], candidates[j]
		if (a.Origin == "user") != (b.Origin == "user") {
			return a.Origin == "user"
		}
		return a.ID < b.ID
	})
	selected := []workflows.DocumentEntity{}
	versions := map[string]int{}
	for _, entity := range candidates {
		next := append(append([]workflows.DocumentEntity(nil), selected...), entity)
		raw, _ := json.Marshal(next)
		if estimateTextTokens(string(raw)) > tokenBudget {
			continue
		}
		selected = next
		versions[entity.ID] = entity.Version
	}
	return selected, versions
}

func documentEntityMention(text, source string) bool {
	if source == "" {
		return false
	}
	first, _ := utf8.DecodeRuneInString(source)
	last, _ := utf8.DecodeLastRuneInString(source)
	wordRune := func(r rune) bool {
		return unicode.Is(unicode.Latin, r) || unicode.Is(unicode.Greek, r) || unicode.Is(unicode.Cyrillic, r) || unicode.IsDigit(r)
	}
	for start := 0; start < len(text); {
		index := strings.Index(text[start:], source)
		if index < 0 {
			return false
		}
		index += start
		end := index + len(source)
		before, _ := utf8.DecodeLastRuneInString(text[:index])
		after, _ := utf8.DecodeRuneInString(text[end:])
		if !(wordRune(first) && wordRune(before)) && !(wordRune(last) && wordRune(after)) {
			return true
		}
		start = end
	}
	return false
}

// Promote a proposal only after a review of the exact paragraph containing its
// evidence. Source-quote existence alone does not establish a correct identity.
func documentEntityReviewUpdates(entities []workflows.DocumentEntity, units []workflows.DocumentUnit) []workflows.DocumentEntity {
	var changed []workflows.DocumentEntity
	for _, entity := range entities {
		if entity.Origin == "user" || len(entity.Evidence) == 0 {
			continue
		}
		// Rechecking a dependent paragraph does not itself disprove this
		// record. Downgrade only on fresh evidence, avoiding version cycles.
		status := entity.Status
		for _, u := range units {
			if !documentReviewCurrent(u) {
				continue
			}
			for _, item := range u.Quality.Reviews {
				matched := false
				for _, id := range item.InputIDs {
					matched = matched || id == entity.Evidence[0].InputID
				}
				if !matched {
					continue
				}
				status = "reviewed"
				for _, f := range item.Findings {
					if f.Uncertain || f.Severity != "minor" {
						status = "unresolved"
					}
				}
				var target strings.Builder
				for _, id := range item.InputIDs {
					target.WriteString(u.Output[id])
				}
				if !strings.Contains(target.String(), entity.Target) {
					status = "stale"
				}
			}
		}
		if status != entity.Status {
			entity.Version++
			entity.Status = status
			changed = append(changed, entity)
		}
	}
	return changed
}

func documentInvalidateEntityDependents(units []workflows.DocumentUnit, changed []workflows.DocumentEntity) []int {
	var indices []int
	for i := range units {
		u := &units[i]
		if u.Quality == nil {
			continue
		}
		for _, entity := range changed {
			if version, used := u.Quality.EntityVersions[entity.ID]; used && version != entity.Version {
				u.Quality.Status = "pending"
				u.Quality.OutputHash = ""
				u.Quality.Reviews = nil
				u.Quality.ReviewRetries = 0
				delete(u.Quality.EntityVersions, entity.ID)
				indices = append(indices, i)
				break
			}
		}
	}
	return indices
}

func documentEntityContext(entities []workflows.DocumentEntity) string {
	if len(entities) == 0 {
		return ""
	}
	raw, _ := json.Marshal(entities)
	return "\nSource-grounded referent proposals (not binding replacements; equal spelling can denote different entities; source evidence overrides tentative labels):\n" + string(raw)
}
