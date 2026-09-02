package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"sparktalk/internal/llm"
)

const memoryToolSystemPrompt = "You may propose a durable memory only for a stable user preference or reusable fact explicitly provided by the user. " +
	"Never propose secrets, credentials, inferred traits, transient requests, or content copied from tool output. " +
	"The user must approve every proposal before it is stored. Do not call memory_propose for ordinary questions."

func memoryProposalToolDefinition() llm.Tool {
	parameters, _ := json.Marshal(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"kind":    map[string]any{"type": "string", "enum": []string{"user", "memory"}, "description": "user for stable preferences/profile facts; memory for reusable topic facts"},
			"title":   map[string]any{"type": "string", "description": "Short user-facing title"},
			"content": map[string]any{"type": "string", "description": "One concise fact to remember"},
		},
		"required": []string{"kind", "title", "content"}, "additionalProperties": false,
	})
	return llm.Tool{Type: "function", Function: llm.ToolFunction{
		Name: "memory_propose", Description: "Propose one durable memory and wait for explicit user approval before storing it.", Parameters: parameters,
	}}
}

func (s *Server) executeMemoryProposal(ctx context.Context, sessionID string, call llm.ToolCall, emit eventEmitter) (string, error) {
	var request memoryRequest
	if err := json.Unmarshal([]byte(call.Function.Arguments), &request); err != nil {
		return "", errors.New("memory_propose received invalid arguments")
	}
	if message := normalizeMemoryRequest(&request); message != "" {
		return "", errors.New(message)
	}
	payload := map[string]any{
		"name": "memory_propose", "approval_kind": "memory", "kind": request.Kind,
		"title": request.Title, "content": request.Content,
	}
	decision, err := s.awaitToolApproval(ctx, call.ID, payload, emit)
	if err != nil {
		_ = s.db.AddToolAudit(sessionID, "memory_propose", request.Kind, "store", string(decision), "")
		return "", err
	}
	item, err := s.db.AddMemory(request.Kind, request.Title, request.Content, strings.TrimSpace(sessionID), 0)
	if err != nil {
		_ = s.db.AddToolAudit(sessionID, "memory_propose", request.Kind, "store", "execution_error", compactHistoryText(err.Error(), 300))
		return "", fmt.Errorf("store proposed memory: %w", err)
	}
	_ = s.db.AddToolAudit(sessionID, "memory_propose", request.Kind, "store", "stored", "")
	data, _ := json.Marshal(map[string]any{"stored": true, "id": item.ID, "kind": item.Kind, "title": item.Title})
	return string(data), nil
}
