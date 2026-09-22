package execution

import (
	"context"
	"encoding/json"
	"errors"
	"sparktalk/internal/llm"
	"testing"
)

type fakeStream struct{ t *testing.T }

func (f fakeStream) Stream(ctx context.Context, m []llm.Message, model, effort string, tools []llm.Tool, emit func(string, string) error) (llm.StreamResult, error) {
	if len(m) != 1 || m[0].Content != "hello" || model != "current" || len(tools) != 0 {
		f.t.Fatal("wrong model request")
	}
	return llm.StreamResult{Content: "answer"}, nil
}
func TestCompleteReleasesLeaseAndUsesCurrentModel(t *testing.T) {
	done := 0
	s := Service{Track: func(ctx context.Context) (context.Context, func(), error) { return ctx, func() { done++ }, nil }, Snapshot: func() (string, Streamer) { return "current", fakeStream{t} }}
	out, err := s.Complete(context.Background(), Request{Input: json.RawMessage(`{"prompt":"hello"}`)})
	if err != nil || len(out) == 0 || done != 1 {
		t.Fatal(string(out), err, done)
	}
	_, err = s.Complete(context.Background(), Request{Input: json.RawMessage(`{}`)})
	if err == nil || done != 2 {
		t.Fatal("invalid input leaked lease")
	}
}
func TestToolServiceKeepsSessionApprovalAndRecursionBoundaries(t *testing.T) {
	calls := 0
	s := Service{SessionExists: func(id string) error {
		if id != "allowed" {
			return errors.New("missing session")
		}
		return nil
	}, ExecuteTool: func(ctx context.Context, id string, call llm.ToolCall, emit func(string, any) error) (json.RawMessage, error) {
		calls++
		return nil, emit("tool_approval", nil)
	}}
	for _, r := range []Request{{Input: json.RawMessage(`{}`)}, {SessionID: "missing", Input: json.RawMessage(`{}`)}, {SessionID: "allowed", Input: json.RawMessage(`{"name":"plugin__x__y"}`)}} {
		if _, err := s.CallTool(context.Background(), r); err == nil {
			t.Fatal("boundary bypassed")
		}
	}
	if calls != 0 {
		t.Fatal("invalid request reached execution")
	}
	if _, err := s.CallTool(context.Background(), Request{SessionID: "allowed", Input: json.RawMessage(`{"name":"ssh_exec","arguments":{}}`)}); err == nil {
		t.Fatal("approval auto granted")
	}
}
