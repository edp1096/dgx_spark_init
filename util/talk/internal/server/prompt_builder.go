package server

import (
	"strings"

	"sparktalk/internal/llm"
)

// assembleModelConversation is the single system-prompt assembly path. Stable
// user instructions stay first for prefix reuse; recalled context/checkpoints
// follow, then guidance for only the tools active on this request.
func assembleModelConversation(systemPrompt string, messages []llm.Message, toolPrompts []string, extraCapacity int) []llm.Message {
	parts := make([]string, 0, 2+len(toolPrompts))
	if prompt := strings.TrimSpace(systemPrompt); prompt != "" {
		parts = append(parts, prompt)
	}
	leadingSystems := 0
	for leadingSystems < len(messages) && messages[leadingSystems].Role == "system" {
		if content, ok := messages[leadingSystems].Content.(string); ok && strings.TrimSpace(content) != "" {
			parts = append(parts, content)
		}
		leadingSystems++
	}
	parts = append(parts, toolPrompts...)

	conversation := make([]llm.Message, 0, len(messages)+1+extraCapacity)
	if len(parts) > 0 {
		conversation = append(conversation, llm.Message{Role: "system", Content: strings.Join(parts, "\n\n")})
	}
	conversation = append(conversation, messages[leadingSystems:]...)
	return conversation
}
