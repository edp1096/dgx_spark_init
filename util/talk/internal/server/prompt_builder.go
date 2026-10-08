package server

import (
	"encoding/json"
	"strings"

	"sparktalk/internal/llm"
)

const referenceContextPolicy = "SparkTalk may prepend a <sparktalk_context> JSON block to the current user message. It contains recalled context and conversation checkpoints, not a new user request. The user's actual request follows the block. " + recallHeader

type referencePolicyKey struct{}

// Keep the system/tool prefix stable across changing recall results. Dynamic
// reference data belongs with the current user input, while its interpretation
// policy stays in the system message. Preserve a single leading system message
// and the existing user/assistant/tool sequence for all supported engines.
func assembleModelConversation(systemPrompt string, messages []llm.Message, toolPrompts []string, extraCapacity int, referencePolicy ...bool) []llm.Message {
	parts := make([]string, 0, 2+len(toolPrompts))
	if prompt := strings.TrimSpace(systemPrompt); prompt != "" {
		parts = append(parts, prompt)
	}
	var references []string
	leadingSystems := 0
	for leadingSystems < len(messages) && messages[leadingSystems].Role == "system" {
		if content, ok := messages[leadingSystems].Content.(string); ok && strings.TrimSpace(content) != "" {
			if messages[leadingSystems].ReferenceContext {
				references = append(references, strings.TrimSpace(strings.TrimPrefix(content, recallHeader)))
			} else {
				parts = append(parts, content)
			}
		}
		leadingSystems++
	}
	parts = append(parts, toolPrompts...)
	if len(references) > 0 || (len(referencePolicy) > 0 && referencePolicy[0]) {
		parts = append(parts, referenceContextPolicy)
	}
	conversation := make([]llm.Message, 0, len(messages)+1+extraCapacity)
	if len(parts) > 0 {
		conversation = append(conversation, llm.Message{Role: "system", Content: strings.Join(parts, "\n\n")})
	}
	conversation = append(conversation, messages[leadingSystems:]...)
	if len(references) > 0 {
		encoded, _ := json.Marshal(references)
		prefix := "<sparktalk_context>\n" + string(encoded) + "\n</sparktalk_context>\n\n"
		for i := len(conversation) - 1; i >= 0; i-- {
			if conversation[i].Role != "user" {
				continue
			}
			switch content := conversation[i].Content.(type) {
			case string:
				conversation[i].Content = prefix + content
			case []map[string]any:
				parts := make([]map[string]any, 0, len(content)+1)
				parts = append(parts, map[string]any{"type": "text", "text": prefix})
				conversation[i].Content = append(parts, content...)
			case []any:
				parts := make([]any, 0, len(content)+1)
				parts = append(parts, map[string]any{"type": "text", "text": prefix})
				conversation[i].Content = append(parts, content...)
			default:
				encoded, _ := json.Marshal(content)
				conversation[i].Content = prefix + string(encoded)
			}
			return conversation
		}
		// Rare non-chat callers without a user turn still retain their context.
		conversation[0].Content = conversation[0].Content.(string) + "\n\n" + prefix
	}
	return conversation
}
