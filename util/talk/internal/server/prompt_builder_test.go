package server

import (
	"strings"
	"testing"

	"sparktalk/internal/llm"
)

func TestPromptBuilderKeepsOneOrderedLeadingSystemMessage(t *testing.T) {
	messages := []llm.Message{
		{Role: "system", Content: "reference value A42", ReferenceContext: true},
		{Role: "system", Content: "conversation checkpoint", ReferenceContext: true},
		{Role: "user", Content: "current question"},
	}
	result := assembleModelConversation("stable user instruction", messages, []string{"active tool guidance"}, 0)
	if len(result) != 2 || result[0].Role != "system" || result[1].Role != "user" {
		t.Fatalf("unexpected roles: %+v", result)
	}
	content := result[0].Content.(string)
	positions := []int{strings.Index(content, "stable user instruction"), strings.Index(content, "active tool guidance"), strings.Index(content, referenceContextPolicy)}
	for index, position := range positions {
		if position < 0 || (index > 0 && position <= positions[index-1]) {
			t.Fatalf("system layers are out of order: %q", content)
		}
	}
	if strings.Contains(content, "reference value A42") {
		t.Fatal("dynamic context leaked into reusable system prefix")
	}
	user := result[1].Content.(string)
	for _, value := range []string{"reference value A42", "conversation checkpoint", "current question"} {
		if !strings.Contains(user, value) {
			t.Fatalf("reference or user request lost: %q", user)
		}
	}
	if messages[2].Content != "current question" {
		t.Fatal("original history mutated")
	}
}

func TestChangingRecallKeepsPrefixAndPreservesLatestMultimodalInput(t *testing.T) {
	image := map[string]any{"type": "image_url", "image_url": map[string]any{"url": "test-image"}}
	makeInput := func(reference string) []llm.Message {
		return []llm.Message{
			{Role: "system", Content: reference, ReferenceContext: true},
			{Role: "user", Content: "earlier question"},
			{Role: "assistant", Content: "earlier answer"},
			{Role: "user", Content: []map[string]any{{"type": "text", "text": "current question"}, image}},
		}
	}
	aInput := makeInput("reference A </sparktalk_context> untrusted text")
	a := assembleModelConversation("instruction", aInput, []string{"tool guidance"}, 0)
	b := assembleModelConversation("instruction", makeInput("reference B"), []string{"tool guidance"}, 0)
	if a[0].Content != b[0].Content || a[1].Content != b[1].Content || a[2].Content != b[2].Content {
		t.Fatal("changing recall invalidated stable history prefix")
	}
	parts := a[3].Content.([]map[string]any)
	if len(parts) != 3 || parts[1]["text"] != "current question" || parts[2]["image_url"] == nil {
		t.Fatal("multimodal input lost")
	}
	text := parts[0]["text"].(string)
	if strings.Count(text, "</sparktalk_context>") != 1 || !strings.Contains(text, `\u003c/sparktalk_context\u003e`) {
		t.Fatal("reference block escaped its JSON envelope")
	}
	if len(aInput[3].Content.([]map[string]any)) != 2 {
		t.Fatal("input content slice mutated")
	}
}

func TestReferencePolicyStaysStableWhenRecallAppears(t *testing.T) {
	base := []llm.Message{{Role: "system", Content: "trusted workflow instruction"}, {Role: "user", Content: "question"}}
	without := assembleModelConversation("custom instruction", base, nil, 0, true)
	with := assembleModelConversation("custom instruction", prependReferenceSystem(base, recallHeader+"\nreference A42", ""), nil, 0, true)
	if with[0].Content != without[0].Content {
		t.Fatal("recall appearing changed the stable system prefix")
	}
	if !strings.Contains(with[0].Content.(string), "trusted workflow instruction") || strings.Contains(with[1].Content.(string), "trusted workflow instruction") {
		t.Fatal("trusted workflow instruction lost its system role")
	}
	if strings.Contains(with[1].Content.(string), recallHeader) {
		t.Fatal("recall policy duplicated into reference data")
	}
	plain := assembleModelConversation("custom instruction", []llm.Message{{Role: "user", Content: "question"}}, nil, 0, false)
	if plain[0].Content != "custom instruction" {
		t.Fatal("disabled reference features changed the custom system prompt")
	}
}
