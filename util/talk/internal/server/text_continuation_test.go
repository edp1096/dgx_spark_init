package server

import (
	"context"
	"errors"
	"sparktalk/internal/llm"
	"strings"
	"testing"
)

func TestLimitedTextContinuation(t *testing.T) {
	for _, mode := range []string{"complete", "limit", "network", "tool", "empty", "cancel", "disabled"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			calls := 0
			emitted := ""
			request := func(messages []llm.Message, effort string, tools []llm.Tool, receive func(string, string) error) (llm.StreamResult, error) {
				calls++
				if calls == 1 {
					if mode == "cancel" {
						cancel()
					}
					result := llm.StreamResult{Content: "```go\nfunc main() {\n", FinishReason: "length"}
					if mode == "tool" {
						result.ToolCalls = []llm.ToolCall{{ID: "partial"}}
					}
					if mode == "network" {
						return result, errors.New("EOF")
					}
					return result, llm.ErrOutputLimit
				}
				if tools != nil || effort != "off" || !strings.Contains(messages[len(messages)-2].Content.(string), "func main") {
					t.Fatal("invalid continuation request")
				}
				if mode == "empty" {
					return llm.StreamResult{FinishReason: "stop"}, nil
				}
				next := llm.StreamResult{Content: "    run()\n}\n```", FinishReason: "stop"}
				if mode == "limit" {
					next.FinishReason = "length"
					return next, llm.ErrOutputLimit
				}
				return next, nil
			}
			result, err := continueLimitedText(ctx, []llm.Message{{Role: "user", Content: "write code"}}, "high", nil, func(kind, part string) error {
				if kind == "delta" {
					emitted += part
				}
				return nil
			}, request, mode != "disabled")
			if mode == "complete" {
				if err != nil || calls != 2 || result.Content != "```go\nfunc main() {\n    run()\n}\n```" || emitted != "    run()\n}\n```" {
					t.Fatalf("calls=%d result=%+v err=%v", calls, result, err)
				}
			} else {
				if err == nil {
					t.Fatal("incomplete answer reported as complete")
				}
				want := 1
				if mode == "limit" {
					want = 3
				}
				if mode == "empty" {
					want = 2
				}
				if calls != want {
					t.Fatalf("calls=%d want=%d", calls, want)
				}
			}
		})
	}
}
