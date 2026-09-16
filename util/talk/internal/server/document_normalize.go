package server

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// Some models serialize nested tool arguments as JSON strings. Decode one
// extra layer, but never interpret plain text as JSON or silently discard it.
func documentArray(raw json.RawMessage) json.RawMessage {
	if len(bytes.TrimSpace(raw)) > 0 && bytes.TrimSpace(raw)[0] == '"' {
		var value string
		if json.Unmarshal(raw, &value) == nil && len(bytes.TrimSpace([]byte(value))) > 0 && bytes.TrimSpace([]byte(value))[0] == '[' && json.Valid([]byte(value)) {
			return json.RawMessage(value)
		}
	}
	return raw
}

func normalizeDocumentSections(root map[string]json.RawMessage) error {
	raw, ok := root["sections"]
	if !ok {
		return nil
	}
	raw = bytes.TrimSpace(raw)
	if len(raw) > 0 && raw[0] == '"' {
		var encoded string
		if err := json.Unmarshal(raw, &encoded); err != nil {
			return err
		}
		raw = json.RawMessage(encoded)
	}
	var sections []map[string]json.RawMessage
	if err := json.Unmarshal(raw, &sections); err != nil {
		return fmt.Errorf("sections는 섹션 객체 배열이어야 합니다. JSON 문자열로 감싸지 말고 배열을 직접 보내세요. 예: sections:[{\"paragraphs\":[\"본문\"]}]. JSON 오류: %v", err)
	}
	if len(sections) == 0 || len(sections) > 80 {
		return fmt.Errorf("문서에는 1–80개 섹션이 필요합니다")
	}
	var out []map[string]json.RawMessage
	for i, section := range sections {
		if section == nil {
			return fmt.Errorf("sections[%d]는 객체여야 합니다", i)
		}
		if blocks, ok := section["blocks"]; ok {
			var items []map[string]json.RawMessage
			if err := json.Unmarshal(documentArray(blocks), &items); err != nil || items == nil {
				return fmt.Errorf("sections[%d].blocks는 객체 배열이어야 합니다", i)
			}
			section["blocks"], _ = json.Marshal(items)
			out = append(out, section)
			continue
		}
		raw, ok := section["paragraphs"]
		if !ok {
			return fmt.Errorf("sections[%d].paragraphs 배열이 필요합니다", i)
		}
		raw = documentArray(raw)
		var paragraphs []string
		if err := json.Unmarshal(raw, &paragraphs); err != nil {
			var paragraph string
			if json.Unmarshal(raw, &paragraph) != nil {
				return fmt.Errorf("sections[%d].paragraphs는 문자열 배열이어야 합니다", i)
			}
			paragraphs = []string{paragraph}
		}
		if paragraphs == nil {
			return fmt.Errorf("sections[%d].paragraphs는 문자열 배열이어야 합니다", i)
		}
		section["paragraphs"], _ = json.Marshal(paragraphs)
		out = append(out, section)

	}
	root["sections"], _ = json.Marshal(out)
	return nil
}
