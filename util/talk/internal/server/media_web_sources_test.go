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

func TestDiscoveredVideoPageIsNotAnImage(t *testing.T) {
	const page = "https://news.example.org/story"
	const video = "https://video.example.org/watch?id=42&cid=news"
	const poster = "https://images.example.org/poster"
	conversation := []llm.Message{
		{Role: "assistant", ToolCalls: []llm.ToolCall{{ID: "collect", Function: llm.FunctionCall{Name: "web_collect", Arguments: `{"url":"` + page + `"}`}}}},
		{Role: "tool", ToolCallID: "collect", Content: `{"url":"` + page + `","images":[{"url":"` + poster + `"}],"links":[{"url":"` + video + `","kind":"link"}]}`},
	}
	if got := discoveredMediaReference(conversation, video); got.SourcePage != page || got.Image {
		t.Fatalf("video page misclassified: %+v", got)
	}
	if got := discoveredMediaReference(conversation, poster); got.SourcePage != page || !got.Image {
		t.Fatalf("extensionless poster misclassified: %+v", got)
	}
	if got := discoveredMediaReference(conversation, "https://video.example.org/watch?id=42"); got.SourcePage != "" {
		t.Fatal("silently accepted an altered query")
	}
	const original = "https://video.example.org/watch"
	const redirect = "https://edition.example.org/watch"
	conversation = append(conversation,
		llm.Message{Role: "assistant", ToolCalls: []llm.ToolCall{{ID: "fetch", Function: llm.FunctionCall{Name: "web_fetch", Arguments: `{"url":"` + original + `"}`}}}},
		llm.Message{Role: "tool", ToolCallID: "fetch", Content: `{"url":"` + redirect + `","content":"video caption"}`})
	for _, target := range []string{original, redirect} {
		if got := discoveredMediaReference(conversation, target); got.SourcePage != redirect || got.Image {
			t.Fatalf("observed redirect not accepted: %+v", got)
		}
	}
}
