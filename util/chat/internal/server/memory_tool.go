package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"sparktalk/internal/db"
	"sparktalk/internal/llm"
)

func memoryManageSystemPrompt(allowProposals bool) string {
	prompt := "Use memory_manage when the user asks what is remembered or explicitly asks to remember, forget, edit, enable, disable, or change how a memory is applied. " +
		"For update or delete, search first and use the exact memory_id returned by that search. Never guess an ID or mutate multiple candidates. " +
		"Every create, update, and delete action requires user approval in the UI; do not claim it succeeded until the tool result confirms it. " +
		"Use kind=user only for facts that should be recalled on every turn, and kind=memory for facts retrieved only when relevant. " +
		"Use priority=preferred for user-confirmed facts that should override general or web knowledge, and priority=reference for non-authoritative background. " +
		"Behavior rules belong in the system prompt, not memory. Never store secrets, credentials, inferred traits, transient requests, or tool output."
	if allowProposals {
		return prompt + " You may propose a stable, clearly reusable fact even without an explicit memory command, but it still requires approval."
	}
	return prompt + " Do not propose new memory unless the user explicitly asks to remember something."
}

func memoryManageToolDefinition() llm.Tool {
	parameters, _ := json.Marshal(map[string]any{
		"type": "object",
		"properties": map[string]any{
			"action": map[string]any{
				"type": "string", "enum": []string{"search", "create", "update", "delete"},
				"description": "search lists matching memories; create/update/delete require UI approval",
			},
			"query": map[string]any{
				"type": "string", "description": "Concise identifying keywords for search. Use an empty string to list recent memories",
			},
			"memory_id": map[string]any{
				"type": "integer", "description": "Exact ID returned by a previous search; required for update and delete",
			},
			"kind": map[string]any{
				"type": "string", "enum": []string{"user", "memory"},
				"description": "user means always reference; memory means retrieve only when relevant",
			},
			"priority": map[string]any{
				"type": "string", "enum": []string{"reference", "preferred"},
				"description": "preferred means user-authoritative when recalled; reference means background only",
			},
			"title":   map[string]any{"type": "string", "description": "Short user-facing title"},
			"content": map[string]any{"type": "string", "description": "Memory content"},
			"enabled": map[string]any{"type": "boolean", "description": "Whether this memory participates in recall"},
		},
		"required":             []string{"action"},
		"additionalProperties": false,
	})
	return llm.Tool{Type: "function", Function: llm.ToolFunction{
		Name:        "memory_manage",
		Description: "Search, create, edit, enable, disable, change recall scope or priority, or delete SparkTalk memories.",
		Parameters:  parameters,
	}}
}

type memoryManageRequest struct {
	Action   string  `json:"action"`
	Query    string  `json:"query,omitempty"`
	MemoryID int64   `json:"memory_id,omitempty"`
	Kind     *string `json:"kind,omitempty"`
	Priority *string `json:"priority,omitempty"`
	Title    *string `json:"title,omitempty"`
	Content  *string `json:"content,omitempty"`
	Enabled  *bool   `json:"enabled,omitempty"`
}

type memoryToolView struct {
	ID       int64  `json:"id"`
	Kind     string `json:"kind"`
	Priority string `json:"priority"`
	Title    string `json:"title"`
	Content  string `json:"content"`
	Enabled  bool   `json:"enabled"`
	Source   string `json:"source"`
}

func (s *Server) executeMemoryManage(ctx context.Context, sessionID string, call llm.ToolCall, emit eventEmitter) (string, error) {
	var request memoryManageRequest
	if err := json.Unmarshal([]byte(call.Function.Arguments), &request); err != nil {
		return "", errors.New("memory_manage received invalid arguments")
	}
	request.Action = strings.ToLower(strings.TrimSpace(request.Action))
	request.Query = strings.TrimSpace(request.Query)
	switch request.Action {
	case "search":
		return s.searchMemoriesForTool(sessionID, request.Query)
	case "create":
		return s.createMemoryFromTool(ctx, sessionID, call.ID, request, emit)
	case "update":
		return s.updateMemoryFromTool(ctx, sessionID, call.ID, request, emit)
	case "delete":
		return s.deleteMemoryFromTool(ctx, sessionID, call.ID, request, emit)
	default:
		return "", errors.New("memory_manage action must be search, create, update, or delete")
	}
}

func (s *Server) searchMemoriesForTool(sessionID, query string) (string, error) {
	items, err := s.db.FindMemories(query, 8)
	if err != nil {
		_ = s.db.AddToolAudit(sessionID, "memory_manage", query, "search", "execution_error", compactHistoryText(err.Error(), 300))
		return "", fmt.Errorf("search memories: %w", err)
	}
	views := make([]memoryToolView, 0, len(items))
	for _, item := range items {
		views = append(views, memoryView(item, 1200))
	}
	_ = s.db.AddToolAudit(sessionID, "memory_manage", query, "search", "executed", fmt.Sprintf("%d result(s)", len(views)))
	data, _ := json.Marshal(map[string]any{"action": "search", "query": query, "memories": views})
	return string(data), nil
}

func (s *Server) createMemoryFromTool(ctx context.Context, sessionID, callID string, request memoryManageRequest, emit eventEmitter) (string, error) {
	kind := "memory"
	priority := "preferred"
	if request.Kind != nil {
		kind = *request.Kind
	}
	if request.Priority != nil {
		priority = *request.Priority
	}
	title, content := "", ""
	if request.Title != nil {
		title = *request.Title
	}
	if request.Content != nil {
		content = *request.Content
	}
	normalized := memoryRequest{Kind: kind, Priority: priority, Title: title, Content: content, SourceSessionID: sessionID}
	if message := normalizeMemoryRequest(&normalized); message != "" {
		return "", errors.New(message)
	}
	payload := map[string]any{
		"name": "memory_manage", "approval_kind": "memory_manage", "action": "create",
		"kind": normalized.Kind, "priority": normalized.Priority, "title": normalized.Title, "content": normalized.Content, "enabled": true,
	}
	decision, err := s.awaitToolApproval(ctx, callID, payload, emit)
	if err != nil {
		_ = s.db.AddToolAudit(sessionID, "memory_manage", "", "create", string(decision), compactHistoryText(err.Error(), 300))
		return "", err
	}
	item, err := s.db.AddMemory(normalized.Kind, normalized.Priority, normalized.Title, normalized.Content, sessionID, 0)
	if err != nil {
		_ = s.db.AddToolAudit(sessionID, "memory_manage", "", "create", "execution_error", compactHistoryText(err.Error(), 300))
		return "", fmt.Errorf("create memory: %w", err)
	}
	_ = s.db.AddToolAudit(sessionID, "memory_manage", strconv.FormatInt(item.ID, 10), "create", string(decision), "")
	return marshalMemoryMutation("create", item)
}

func (s *Server) updateMemoryFromTool(ctx context.Context, sessionID, callID string, request memoryManageRequest, emit eventEmitter) (string, error) {
	if request.MemoryID < 1 {
		return "", errors.New("memory_id is required for update; search memories first")
	}
	if request.Kind == nil && request.Priority == nil && request.Title == nil && request.Content == nil && request.Enabled == nil {
		return "", errors.New("update requires at least one of kind, priority, title, content, or enabled")
	}
	before, err := s.db.Memory(request.MemoryID)
	if err != nil {
		return "", memoryManageLookupError(err)
	}
	kind, priority, title, content, enabled := before.Kind, before.Priority, before.Title, before.Content, before.Enabled
	if request.Kind != nil {
		kind = *request.Kind
	}
	if request.Priority != nil {
		priority = *request.Priority
	}
	if request.Title != nil {
		title = *request.Title
	}
	if request.Content != nil {
		content = *request.Content
	}
	if request.Enabled != nil {
		enabled = *request.Enabled
	}
	normalized := memoryRequest{Kind: kind, Priority: priority, Title: title, Content: content}
	if message := normalizeMemoryRequest(&normalized); message != "" {
		return "", errors.New(message)
	}
	payload := memoryMutationApproval("update", before)
	payload["kind"], payload["priority"], payload["title"], payload["content"], payload["enabled"] = normalized.Kind, normalized.Priority, normalized.Title, normalized.Content, enabled
	decision, err := s.awaitToolApproval(ctx, callID, payload, emit)
	resource := strconv.FormatInt(before.ID, 10)
	if err != nil {
		_ = s.db.AddToolAudit(sessionID, "memory_manage", resource, "update", string(decision), compactHistoryText(err.Error(), 300))
		return "", err
	}
	item, err := s.db.UpdateMemory(before.ID, normalized.Kind, normalized.Priority, normalized.Title, normalized.Content, enabled)
	if err != nil {
		_ = s.db.AddToolAudit(sessionID, "memory_manage", resource, "update", "execution_error", compactHistoryText(err.Error(), 300))
		return "", fmt.Errorf("update memory: %w", err)
	}
	_ = s.db.AddToolAudit(sessionID, "memory_manage", resource, "update", string(decision), "")
	return marshalMemoryMutation("update", item)
}

func (s *Server) deleteMemoryFromTool(ctx context.Context, sessionID, callID string, request memoryManageRequest, emit eventEmitter) (string, error) {
	if request.MemoryID < 1 {
		return "", errors.New("memory_id is required for delete; search memories first")
	}
	item, err := s.db.Memory(request.MemoryID)
	if err != nil {
		return "", memoryManageLookupError(err)
	}
	payload := memoryMutationApproval("delete", item)
	decision, err := s.awaitToolApproval(ctx, callID, payload, emit)
	resource := strconv.FormatInt(item.ID, 10)
	if err != nil {
		_ = s.db.AddToolAudit(sessionID, "memory_manage", resource, "delete", string(decision), compactHistoryText(err.Error(), 300))
		return "", err
	}
	if err := s.db.DeleteMemory(item.ID); err != nil {
		_ = s.db.AddToolAudit(sessionID, "memory_manage", resource, "delete", "execution_error", compactHistoryText(err.Error(), 300))
		return "", fmt.Errorf("delete memory: %w", err)
	}
	_ = s.db.AddToolAudit(sessionID, "memory_manage", resource, "delete", string(decision), "")
	data, _ := json.Marshal(map[string]any{"action": "delete", "deleted": memoryView(item, 1200)})
	return string(data), nil
}

func memoryMutationApproval(action string, item db.Memory) map[string]any {
	return map[string]any{
		"name": "memory_manage", "approval_kind": "memory_manage", "action": action, "memory_id": item.ID,
		"before_kind": item.Kind, "before_priority": item.Priority, "before_title": item.Title, "before_content": item.Content, "before_enabled": item.Enabled,
		"kind": item.Kind, "priority": item.Priority, "title": item.Title, "content": item.Content, "enabled": item.Enabled,
	}
}

func memoryManageLookupError(err error) error {
	if db.IsMemoryNotFound(err) {
		return errors.New("memory not found; search memories again")
	}
	return fmt.Errorf("load memory: %w", err)
}

func memoryView(item db.Memory, contentLimit int) memoryToolView {
	return memoryToolView{
		ID: item.ID, Kind: item.Kind, Priority: item.Priority, Title: item.Title,
		Content: truncateRunes(item.Content, contentLimit), Enabled: item.Enabled,
		Source: memorySourceName(item),
	}
}

func memorySourceName(item db.Memory) string {
	if item.SourceMessageID > 0 {
		return "conversation"
	}
	if item.SourceSessionID != "" {
		return "model_proposal"
	}
	return "manual"
}

func truncateRunes(value string, limit int) string {
	runes := []rune(value)
	if limit < 1 || len(runes) <= limit {
		return value
	}
	return strings.TrimSpace(string(runes[:limit])) + "…"
}

func marshalMemoryMutation(action string, item db.Memory) (string, error) {
	data, err := json.Marshal(map[string]any{"action": action, "memory": memoryView(item, 8000)})
	return string(data), err
}
