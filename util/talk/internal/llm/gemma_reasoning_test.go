package llm

import "testing"

func TestGemmaToolReasoningCurrentTurnOnly(t *testing.T) {
	for _, modelType := range []string{"gemma4", "gemma4-vllm", "generic"} {
		t.Run(modelType, func(t *testing.T) {
			messages := []Message{
				{Role: "user", Content: "old"},
				{Role: "assistant", ReasoningContent: "old thought", ToolCalls: []ToolCall{{}}},
				{Role: "user", Content: "current"},
				{Role: "assistant", ReasoningContent: "search plan", ToolCalls: []ToolCall{{}}},
				{Role: "tool", Content: "result"},
				{Role: "assistant", ReasoningContent: "read plan", ToolCalls: []ToolCall{{}}},
				{Role: "assistant", ReasoningContent: "final thought"},
			}
			got := New("", "", "", modelType).InputMessages(messages)
			for i := range got {
				want := ""
				if modelType != "generic" && (i == 3 || i == 5) {
					want = messages[i].ReasoningContent
				}
				if got[i].ReasoningContent != want {
					t.Fatalf("message %d: got %q, want %q", i, got[i].ReasoningContent, want)
				}
			}
			if messages[1].ReasoningContent != "old thought" || messages[3].ReasoningContent != "search plan" {
				t.Fatal("mutated original messages")
			}
		})
	}
}
