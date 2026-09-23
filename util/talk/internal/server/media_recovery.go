package server

import (
	"context"
	"encoding/json"
	"fmt"
	"sparktalk/internal/llm"
)

const mediaRecoveryFinalInstruction = "Media retrieval failed and bounded recovery attempts are exhausted. Stop retrieving URLs. Explain the actual failure concisely using existing results. Distinguish page metadata, sampled frames, and actual transcripts. Do not claim to have watched or summarized the entire video without evidence. Do not suggest increasing tool rounds or invent other proxy/transcript endpoints."

// Scoped to one turn. Explicit new user input permits another attempt.
type mediaRecovery struct {
	failed   map[string]error
	active   bool
	attempts int
	inputs   int
}

func (g *mediaRecovery) exhausted() bool { return g.active && g.attempts >= 12 }
func (g *mediaRecovery) execute(next registeredToolHandler) registeredToolHandler {
	return func(ctx context.Context, call llm.ToolCall, messages []llm.Message, emit eventEmitter) (registeredToolResult, error) {
		if n := len(turnFrom(ctx).appliedInputs()); n != g.inputs {
			*g = mediaRecovery{inputs: n}
		}
		name := call.Function.Name
		retrieval := name == "media_import" || name == "web_fetch" || name == "web_collect" || name == "web_search" || name == "browser"
		if retrieval && g.exhausted() {
			return registeredToolResult{}, fmt.Errorf("media retrieval recovery exhausted; report available evidence and failure")
		}
		if retrieval && g.active {
			g.attempts++
		}
		key := call.Function.Arguments
		var args any
		if json.Unmarshal([]byte(key), &args) == nil {
			b, _ := json.Marshal(args)
			key = string(b)
		}
		if name == "media_import" {
			if err := g.failed[key]; err != nil {
				return registeredToolResult{}, fmt.Errorf("same media import already failed in this turn; not retried: %w", err)
			}
		}
		result, err := next(ctx, call, messages, emit)
		if name == "media_import" && err != nil && ctx.Err() == nil {
			if g.failed == nil {
				g.failed = map[string]error{}
			}
			g.failed[key] = err
			g.active = true
		}
		if err == nil && result.Attachment != nil {
			g.active = false
			g.attempts = 0
		}
		return result, err
	}
}
