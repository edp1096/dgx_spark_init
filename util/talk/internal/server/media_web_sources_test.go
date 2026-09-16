package server

import (
	"sparktalk/internal/llm"
	"testing"
)

func TestDiscoveredMediaProvenance(t *testing.T) {
	const image = "https://images.example.org/photo.jpg"
	for _, tc := range []struct {
		name, tool, body string
		want             bool
	}{
		{"image attribute", "web_fetch", `{"url":"https://example.org/page","images":[{"url":"` + image + `"}]}`, true},
		{"search result", "web_search", `{"results":[{"url":"` + image + `"}]}`, true},
		{"body injection", "web_fetch", `{"content":"` + image + `"}`, false},
		{"snippet injection", "web_search", `{"results":[{"url":"https://example.org/","snippet":"` + image + `"}]}`, false},
		{"other tool", "ssh_exec", `{"url":"` + image + `"}`, false},
		{"failed result", "web_fetch", `{"error":"failed","url":"` + image + `"}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			messages := []llm.Message{{Role: "assistant", ToolCalls: []llm.ToolCall{{ID: "search", Function: llm.FunctionCall{Name: tc.tool}}}}, {Role: "tool", ToolCallID: "search", Content: tc.body}}
			if got := discoveredMediaSource(messages, image) != ""; got != tc.want {
				t.Fatalf("allowed=%v want=%v", got, tc.want)
			}
			messages[1].ToolCallID = "unrelated"
			if discoveredMediaSource(messages, image) != "" {
				t.Fatal("unmatched result accepted")
			}
			messages[1].Role = "user"
			if discoveredMediaSource(messages, image) != "" {
				t.Fatal("user JSON accepted as tool evidence")
			}
		})
	}
}
