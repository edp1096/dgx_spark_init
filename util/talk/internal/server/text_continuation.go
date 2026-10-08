package server

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"sparktalk/internal/llm"
)

type textRequest func([]llm.Message, string, []llm.Tool, func(string, string) error) (llm.StreamResult, error)

// Only a confirmed output limit can trigger another request. Never reconstruct
// partial tool arguments or replay tools. Each request goes through context budgeting.
func continueLimitedText(ctx context.Context, messages []llm.Message, effort string, tools []llm.Tool, receive func(string, string) error, request textRequest, enabled bool) (llm.StreamResult, error) {
	result, err := request(messages, effort, tools, receive)
	if !enabled {
		return result, err
	}
	base := append([]llm.Message(nil), messages...)
	for attempt := 0; attempt < 2 && errors.Is(err, llm.ErrOutputLimit) && len(result.ToolCalls) == 0 && strings.TrimSpace(result.Content) != ""; attempt++ {
		if ctx.Err() != nil {
			return result, ctx.Err()
		}
		if _, leaked := cleanToolProtocol(result.Content); leaked {
			return result, err
		}
		nextMessages := append(append([]llm.Message(nil), base...),
			llm.Message{Role: "assistant", Content: result.Content},
			llm.Message{Role: "user", Content: "The previous answer stopped at the output token limit. Continue its exact text from the next character. Do not repeat any existing text, add an introduction, or reopen a code fence already open. Preserve indentation and finish the requested answer. Tools are unavailable; do not request or simulate tools."})
		// Buffer the continuation so an unexpected tool/protocol response cannot be
		// appended to a previously valid answer. Preserve the bytes, including whitespace.
		next, nextErr := request(nextMessages, "off", nil, func(string, string) error { return nil })
		if len(next.ToolCalls) > 0 {
			return result, fmt.Errorf("이어쓰기에서 예상하지 않은 도구 호출을 반환했습니다. 도구를 실행하지 않았습니다.")
		}
		if _, leaked := cleanToolProtocol(next.Content); leaked {
			return result, fmt.Errorf("이어쓰기에 도구 프로토콜이 포함되어 중단했습니다.")
		}
		if next.Reasoning != "" {
			if emitErr := receive("reasoning", next.Reasoning); emitErr != nil {
				return result, emitErr
			}
		}
		if next.Content != "" {
			if emitErr := receive("delta", next.Content); emitErr != nil {
				return result, emitErr
			}
		}
		result.Content += next.Content
		result.Reasoning = mergeReasoning(result.Reasoning, next.Reasoning)
		result.FinishReason = next.FinishReason
		result.Usage = next.Usage
		err = nextErr
		if strings.TrimSpace(next.Content) == "" {
			if err == nil {
				err = fmt.Errorf("이어쓰기에서 본문을 반환하지 않아 응답이 불완전합니다.")
			}
			return result, err
		}
	}
	return result, err
}
