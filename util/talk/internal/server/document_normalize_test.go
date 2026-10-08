package server

import (
	"context"
	"encoding/json"
	"fmt"
	"sparktalk/internal/llm"
	"strings"
	"testing"
)

func TestDocumentNormalizePreservesContent(t *testing.T) {
	paragraphs := make([]string, 81)
	for i := range paragraphs {
		paragraphs[i] = fmt.Sprintf("문단 %d", i)
	}
	sections, _ := json.Marshal([]any{map[string]any{"heading": "제목", "paragraphs": paragraphs, "table": [][]string{{"열"}, {"값"}}}})
	for _, encoded := range []bool{false, true} {
		var value any = json.RawMessage(sections)
		if encoded {
			value = string(sections)
		}
		args, _ := json.Marshal(map[string]any{"format": "pdf", "title": "보고서", "sections": value})
		result, err := (&Server{}).documentImagePayload("", string(args))
		if err != nil {
			t.Fatal(err)
		}
		var out struct {
			Sections []struct {
				Heading    string
				Paragraphs []string
				Table      [][]string
			}
		}
		if err = json.Unmarshal(result, &out); err != nil {
			t.Fatal(err)
		}
		if len(out.Sections) != 1 {
			t.Fatalf("%s", result)
		}
		var joined []string
		for i, s := range out.Sections {
			if i == 0 && s.Heading != "제목" {
				t.Fatal("heading lost")
			}
			if i > 0 && s.Heading != "" {
				t.Fatal("heading repeated")
			}
			joined = append(joined, s.Paragraphs...)
		}
		if len(out.Sections[0].Table) != 2 || fmt.Sprint(joined) != fmt.Sprint(paragraphs) {
			t.Fatal("content changed")
		}
	}
}
func TestDocumentNormalizeTypesAndLimit(t *testing.T) {
	for _, raw := range []string{`["본문"]`, `"본문"`, `"[\"본문\"]"`} {
		_, err := (&Server{}).documentImagePayload("", `{"sections":[{"paragraphs":`+raw+`}]}`)
		if err != nil {
			t.Fatal(err)
		}
	}
	for _, raw := range []string{`"not JSON"`, `[null]`, `[{"paragraphs":null}]`, `[{"paragraphs":[1]}]`} {
		if _, err := (&Server{}).documentImagePayload("", `{"sections":`+raw+`}`); err == nil {
			t.Fatalf("accepted %s", raw)
		}
	}
	p := make([]string, 3201)
	args, _ := json.Marshal(map[string]any{"sections": []any{map[string]any{"paragraphs": p}}})
	if _, err := (&Server{}).documentImagePayload("", string(args)); err != nil {
		t.Fatal(err)
	}
}

func TestDocumentMalformedEncodedSections(t *testing.T) {
	args, _ := json.Marshal(map[string]any{"sections": `[{"blocks":[{"type":"paragraph","text":"본문"}]}`})
	_, err := (&Server{}).documentImagePayload("", string(args))
	if err == nil || !strings.Contains(err.Error(), "JSON 오류") || !strings.Contains(err.Error(), "배열을 직접") {
		t.Fatalf("expected actionable malformed JSON error, got %v", err)
	}
}
func TestDocumentRejectsSourceFileDelivery(t *testing.T) {
	for _, filename := range []string{"game.js", "index.html", "style.css", "SCRIPT.PY"} {
		args, _ := json.Marshal(map[string]any{"format": "docx", "filename": filename, "sections": []any{map[string]any{"paragraphs": []string{"code"}}}})
		_, err := (&Server{}).executeDocumentGenerate(context.Background(), llm.ToolCall{Function: llm.FunctionCall{Arguments: string(args)}})
		if err == nil || !strings.Contains(err.Error(), "HTML/ZIP") {
			t.Fatalf("%s: %v", filename, err)
		}
	}
}
