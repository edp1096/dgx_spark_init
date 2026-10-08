// Package execution exposes transport-independent model and tool services.
package execution

import (
	"context"
	"encoding/json"
	"fmt"
	"sparktalk/internal/llm"
	"strings"
)

type Streamer interface {
	Stream(context.Context, []llm.Message, string, string, []llm.Tool, func(string, string) error) (llm.StreamResult, error)
}
type Request struct {
	SessionID string
	Input     json.RawMessage
}
type Service struct {
	Snapshot      func() (string, Streamer)
	Track         func(context.Context) (context.Context, func(), error)
	SessionExists func(string) error
	ExecuteTool   func(context.Context, string, llm.ToolCall, func(string, any) error) (json.RawMessage, error)
}

func (s Service) Complete(ctx context.Context, r Request) (json.RawMessage, error) {
	ctx, finish, err := s.Track(ctx)
	if err != nil {
		return nil, err
	}
	defer finish()
	var in struct {
		Prompt    string `json:"prompt"`
		Model     string `json:"model"`
		Reasoning string `json:"reasoning_effort"`
	}
	if err = json.Unmarshal(r.Input, &in); err != nil {
		return nil, err
	}
	if strings.TrimSpace(in.Prompt) == "" {
		return nil, fmt.Errorf("prompt is required")
	}
	model, client := s.Snapshot()
	if in.Model == "" {
		in.Model = model
	}
	out, err := client.Stream(ctx, []llm.Message{{Role: "user", Content: in.Prompt}}, in.Model, in.Reasoning, nil, func(string, string) error { return ctx.Err() })
	if err != nil {
		return nil, err
	}
	return json.Marshal(map[string]any{"content": out.Content, "reasoning": out.Reasoning, "finish_reason": out.FinishReason})
}

// CallTool requires an existing session and refuses implicit interactive approval.
func (s Service) CallTool(ctx context.Context, r Request) (json.RawMessage, error) {
	if r.SessionID == "" {
		return nil, fmt.Errorf("session_id is required")
	}
	if err := s.SessionExists(r.SessionID); err != nil {
		return nil, err
	}
	var in struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal(r.Input, &in); err != nil {
		return nil, err
	}
	if strings.HasPrefix(in.Name, "plugin__") {
		return nil, fmt.Errorf("recursive plugin tool calls are not supported")
	}
	return s.ExecuteTool(ctx, r.SessionID, llm.ToolCall{Type: "function", Function: llm.FunctionCall{Name: in.Name, Arguments: string(in.Arguments)}}, func(event string, _ any) error {
		if event == "tool_approval" {
			return fmt.Errorf("interactive approval required; background execution stopped")
		}
		return ctx.Err()
	})
}
