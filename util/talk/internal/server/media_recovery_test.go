package server

import (
	"context"
	"fmt"
	"sparktalk/internal/llm"
	"testing"
)

func TestFailedMediaRecoveryBounded(t *testing.T) {
	g := &mediaRecovery{}
	calls := 0
	execute := g.execute(func(_ context.Context, c llm.ToolCall, _ []llm.Message, _ eventEmitter) (registeredToolResult, error) {
		calls++
		if c.Function.Name == "media_import" {
			return registeredToolResult{}, fmt.Errorf("format unavailable")
		}
		return registeredToolResult{Result: "page metadata only"}, nil
	})
	var c llm.ToolCall
	c.Function.Name = "media_import"
	c.Function.Arguments = `{"url":"https://example.com/video"}`
	execute(context.Background(), c, nil, nil)
	c.Function.Arguments = `{ "url" : "https://example.com/video" }`
	execute(context.Background(), c, nil, nil)
	if calls != 1 {
		t.Fatal("duplicate failure was executed")
	}
	c.Function.Name = "web_fetch"
	for i := 0; i < 30; i++ {
		c.Function.Arguments = fmt.Sprintf(`{"url":"https://example.com/%d"}`, i)
		execute(context.Background(), c, nil, nil)
	}
	if calls != 12 || !g.exhausted() {
		t.Fatalf("calls=%d exhausted=%v", calls, g.exhausted())
	}
}

func TestMediaProvenanceCanBeEstablishedAfterRejection(t *testing.T) {
	g := &mediaRecovery{}
	observed, calls := false, 0
	execute := g.execute(func(_ context.Context, c llm.ToolCall, _ []llm.Message, _ eventEmitter) (registeredToolResult, error) {
		calls++
		if !observed {
			return registeredToolResult{}, errMediaURLNotObserved
		}
		return registeredToolResult{Result: "downloaded"}, nil
	})
	call := llm.ToolCall{Function: llm.FunctionCall{Name: "media_import", Arguments: `{"url":"https://example.org/video"}`}}
	_, _ = execute(context.Background(), call, nil, nil)
	observed = true
	if _, err := execute(context.Background(), call, nil, nil); err != nil || calls != 2 {
		t.Fatalf("new provenance was masked by old failure: calls=%d err=%v", calls, err)
	}
}
