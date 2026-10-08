package server

import (
	"context"
	"fmt"
	"sparktalk/internal/llm"
)

type steeringContinuationKey struct{}

func steeredRequest(ctx context.Context, conversation *[]llm.Message, inStage bool, emit eventEmitter, request func(context.Context, []llm.Message) (llm.StreamResult, error)) (llm.StreamResult, error) {
	turn := turnFrom(ctx)
	for {
		requestCtx, cancel, inputs := turn.inferContext(ctx)
		if len(inputs) > 0 {
			for _, input := range inputs {
				*conversation = append(*conversation, llm.Message{Role: "user", Content: input.Content})
			}
			if err := emit("steering_applied", inputs); err != nil {
				cancel()
				turn.inferenceEnded(false)
				return llm.StreamResult{}, err
			}
		}
		result, err := request(requestCtx, *conversation)
		cancel()
		pending := turn.inferenceEnded(err == nil && len(result.ToolCalls) == 0 && !inStage)
		if ctx.Err() != nil {
			return result, ctx.Err()
		}
		if !pending {
			return result, err
		}
		// A partially generated tool call has never executed and must not be replayed.
		// Completed tools are already in conversation and remain there.
		if result.Content != "" {
			*conversation = append(*conversation, llm.Message{Role: "assistant", Content: result.Content})
		}
	}
}

func executeTurnTool(ctx context.Context, call llm.ToolCall, conversation []llm.Message, emit eventEmitter, execute registeredToolHandler) (registeredToolResult, error) {
	if turnFrom(ctx).hasPending() {
		return registeredToolResult{}, fmt.Errorf("추가 지시가 도착하여 아직 시작하지 않은 도구 호출을 건너뛰었습니다")
	}
	return execute(ctx, call, conversation, emit)
}
