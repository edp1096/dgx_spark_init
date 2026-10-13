package server

import (
	"encoding/json"
	"fmt"

	"sparktalk/internal/llm"
)

const documentReviewProtocolV2 = "source_target_review_v2"

// Evidence comes before the judgement in constrained decoding. Small findings
// keep review cost proportional to actual discrepancies, not rewritten prose.
func documentReviewSchemaV2(pairs []documentReviewPair) []byte {
	finding := map[string]any{"type": "object", "properties": map[string]any{
		"evidence": map[string]any{"type": "object", "properties": map[string]any{
			"source_quote": map[string]any{"type": "string", "minLength": 1, "maxLength": 128},
			"target_quote": map[string]any{"type": "string", "maxLength": 128},
		}, "required": []string{"source_quote", "target_quote"}, "additionalProperties": false},
		"finding": map[string]any{"type": "object", "properties": map[string]any{
			"category":    map[string]any{"type": "string", "enum": documentSemanticFindingCategories()},
			"severity":    map[string]any{"type": "string", "enum": []string{"critical", "major", "minor"}},
			"explanation": map[string]any{"type": "string", "minLength": 1, "maxLength": 160},
			"uncertain":   map[string]any{"type": "boolean"},
		}, "required": []string{"category", "severity", "explanation", "uncertain"}, "additionalProperties": false},
	}, "required": []string{"evidence", "finding"}, "additionalProperties": false}
	properties := map[string]any{}
	ids := []string{}
	for _, p := range pairs {
		ids = append(ids, p.ID)
		properties[p.ID] = map[string]any{"type": "array", "maxItems": 12, "items": finding}
	}
	raw, _ := json.Marshal(map[string]any{"type": "object", "properties": map[string]any{"checks": map[string]any{"type": "object", "properties": properties, "required": ids, "additionalProperties": false}}, "required": []string{"checks"}, "additionalProperties": false})
	return raw
}

// Normalize only representation; no invented evidence or inferred field values.
// An invalid nested finding invalidates its paragraph, preserving valid siblings.
func documentReviewRepresentation(result llm.StreamResult) (llm.StreamResult, string, error) {
	protocol := documentReviewProtocol
	if err := documentUniqueJSON(result.Content); err != nil {
		return result, protocol, err
	}
	var root map[string]json.RawMessage
	if err := json.Unmarshal([]byte(result.Content), &root); err != nil {
		return result, protocol, err
	}
	checks, ok := root["checks"]
	if !ok {
		return result, protocol, nil
	}
	protocol = documentReviewProtocolV2
	if len(root) != 1 {
		return result, protocol, fmt.Errorf("unexpected review root field")
	}
	var paragraphs map[string]json.RawMessage
	if err := json.Unmarshal(checks, &paragraphs); err != nil {
		return result, protocol, err
	}
	normalized := map[string]json.RawMessage{}
	exact := func(object map[string]json.RawMessage, keys ...string) bool {
		if len(object) != len(keys) {
			return false
		}
		for _, k := range keys {
			if _, ok := object[k]; !ok {
				return false
			}
		}
		return true
	}
	for id, raw := range paragraphs {
		var findings []map[string]json.RawMessage
		if json.Unmarshal(raw, &findings) != nil || findings == nil {
			normalized[id] = json.RawMessage("null")
			continue
		}
		flattened := []map[string]json.RawMessage{}
		valid := true
		for _, nested := range findings {
			var evidence, judgement map[string]json.RawMessage
			if !exact(nested, "evidence", "finding") || json.Unmarshal(nested["evidence"], &evidence) != nil || json.Unmarshal(nested["finding"], &judgement) != nil || !exact(evidence, "source_quote", "target_quote") || !exact(judgement, "category", "severity", "explanation", "uncertain") {
				valid = false
				break
			}
			for _, field := range []struct {
				object map[string]json.RawMessage
				key    string
				limit  int
			}{{evidence, "source_quote", 128}, {evidence, "target_quote", 128}, {judgement, "explanation", 160}} {
				var text string
				if json.Unmarshal(field.object[field.key], &text) != nil || len([]rune(text)) > field.limit {
					valid = false
					break
				}
			}
			for k, v := range evidence {
				judgement[k] = v
			}
			flattened = append(flattened, judgement)
		}
		if !valid {
			normalized[id] = json.RawMessage("null")
		} else {
			normalized[id], _ = json.Marshal(flattened)
		}
	}
	raw, _ := json.Marshal(map[string]any{"reviews": normalized})
	result.Content = string(raw)
	return result, protocol, nil
}

// Give the model the same grammar the backend enforces. The schema is task
// metadata, never source prose; quoted speech must survive JSON serialization.
func documentSchemaInstructions(schema []byte) string {
	return `
Output format definition (JSON Schema; do not translate this definition):
` + string(schema) + `
Use the exact field names and allowed enum strings above. Return all requested text, including quoted speech. Within a JSON string, escape an ASCII double quote as \" and a backslash as \\; a quotation in the source is not the end of the translated paragraph. Do not omit the rest of a sentence to make JSON valid.
For example, the text He said "hello". is encoded as "He said \"hello\".".`
}

// Structural layout is validated against the source tree. This reviewer sees
// plain paragraph pairs, so it must not trigger rewrites for presentation alone.
func documentSemanticFindingCategories() []string {
	out := make([]string, 0, len(documentFindingCategories))
	for _, category := range documentFindingCategories {
		if category != "layout" {
			out = append(out, category)
		}
	}
	return out
}
