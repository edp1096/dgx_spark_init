package server

import (
	"context"
	"encoding/json"
	"sparktalk/internal/execution"
	"sparktalk/internal/llm"
	"sparktalk/internal/plugins"
)

// Adapt application dependencies once; transports do not construct clients or
// bypass the existing tool registry's authorization checks.
func (s *Server) executionService() execution.Service {
	return execution.Service{
		Track: s.trackGeneration,
		Snapshot: func() (string, execution.Streamer) {
			cfg, client := s.snapshot()
			return cfg.Model.DefaultModel, client
		},
		SessionExists: func(id string) error { _, err := s.db.Session(id); return err },
		ExecuteTool: func(ctx context.Context, id string, call llm.ToolCall, emit func(string, any) error) (json.RawMessage, error) {
			cfg, _ := s.snapshot()
			registry := newCompletionToolRegistry(s, id, cfg.Tools, true, nil)
			result, err := registry.execute(ctx, call, nil, emit)
			if err != nil {
				return nil, err
			}
			return json.Marshal(map[string]any{"result": result.Result, "attachments": result.Attachments, "attachment": result.Attachment})
		},
	}
}
func (s *Server) pluginServices() map[string]plugins.Service {
	svc := s.executionService()
	adapt := func(fn func(context.Context, execution.Request) (json.RawMessage, error)) plugins.Service {
		return func(ctx context.Context, r plugins.Request) (json.RawMessage, error) {
			return fn(ctx, execution.Request{SessionID: r.SessionID, Input: r.Input})
		}
	}
	return map[string]plugins.Service{"model.complete": adapt(svc.Complete), "tool.call": adapt(svc.CallTool)}
}
