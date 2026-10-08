package llm

import (
	"fmt"
	"reflect"
	"testing"
)

func TestImageBudgetKeepsNewestWithoutMutatingTranscript(t *testing.T) {
	var messages []Message
	for i := 0; i < 6; i++ {
		messages = append(messages, Message{Role: "user", Content: []map[string]any{{"type": "text", "text": "caption"}, {"type": "image_url", "image_url": map[string]string{"url": fmt.Sprint(i)}}}})
	}
	c := New("", "", "", "qwen3.5")
	out := c.InputMessages(messages)
	for i, m := range out {
		parts := m.Content.([]map[string]any)
		want := "image_url"
		if i < 2 {
			want = "text"
		}
		if parts[1]["type"] != want {
			t.Fatalf("image %d: %+v", i, parts)
		}
		if messages[i].Content.([]map[string]any)[1]["type"] != "image_url" {
			t.Fatal("source changed")
		}
	}
	if !reflect.DeepEqual(out, c.InputMessages(out)) {
		t.Fatal("not idempotent")
	}
	// Follow-up tool images consume the same request budget.
	out = append(out, Message{Role: "user", Content: []map[string]any{{"type": "image_url", "image_url": map[string]string{"url": "tool-result"}}}})
	out = c.InputMessages(out)
	n := 0
	for _, m := range out {
		for _, p := range m.Content.([]map[string]any) {
			if p["type"] == "image_url" {
				n++
			}
		}
	}
	if n != 4 {
		t.Fatalf("tool followup images=%d", n)
	}
}
