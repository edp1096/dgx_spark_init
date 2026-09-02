package server

import (
	"strings"
	"testing"

	"sparktalk/internal/llm"
)

func TestPromptBuilderKeepsOneOrderedLeadingSystemMessage(t *testing.T) {
	messages := []llm.Message{
		{Role: "system", Content: "retrieved memory"},
		{Role: "system", Content: "conversation checkpoint"},
		{Role: "user", Content: "current question"},
	}
	result := assembleModelConversation("stable user instruction", messages, []string{"active tool guidance"}, 0)
	if len(result) != 2 || result[0].Role != "system" || result[1].Role != "user" {
		t.Fatalf("unexpected roles: %+v", result)
	}
	content := result[0].Content.(string)
	positions := []int{
		strings.Index(content, "stable user instruction"),
		strings.Index(content, "retrieved memory"),
		strings.Index(content, "conversation checkpoint"),
		strings.Index(content, "active tool guidance"),
	}
	for index, position := range positions {
		if position < 0 || (index > 0 && position <= positions[index-1]) {
			t.Fatalf("system layers are out of order: %q", content)
		}
	}
}
