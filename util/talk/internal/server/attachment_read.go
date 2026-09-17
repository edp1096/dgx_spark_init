package server

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"sparktalk/internal/db"
	"sparktalk/internal/llm"
)

func (s *Server) registerAttachmentReader(registry *completionToolRegistry, sessionID string) {
	registry.register(llm.Tool{Type: "function", Function: llm.ToolFunction{
		Name: "attachment_read", Description: "Read uploaded ZIP/document/text attachments directly from this conversation. Omit attachment_id to list available files. Use the exact returned ID; offset pages through extracted text. Uploaded attachments exist in SparkTalk storage, not necessarily on an SSH host. Never ask for another upload merely because an SSH /tmp path is missing.",
		Parameters: json.RawMessage(`{"type":"object","properties":{"attachment_id":{"type":"string"},"offset":{"type":"integer","minimum":0}},"additionalProperties":false}`),
	}}, func(ctx context.Context, call llm.ToolCall, _ []llm.Message, _ eventEmitter) (registeredToolResult, error) {
		var args struct {
			ID     string `json:"attachment_id"`
			Offset int    `json:"offset"`
		}
		if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
			return registeredToolResult{}, err
		}
		if args.Offset < 0 {
			return registeredToolResult{}, fmt.Errorf("offset must be nonnegative")
		}
		messages, err := s.db.Messages(sessionID)
		if err != nil {
			return registeredToolResult{}, err
		}
		files := []db.Attachment{}
		seen := make(map[string]bool)
		for _, m := range messages {
			for _, a := range m.Attachments {
				if isDocumentAttachment(a) && !seen[a.ID] {
					files = append(files, a)
					seen[a.ID] = true
				}
			}
		}
		response := map[string]any{"attachments": files}
		if args.ID != "" {
			var found *db.Attachment
			for i := range files {
				if files[i].ID == args.ID {
					found = &files[i]
					break
				}
			}
			if found == nil {
				return registeredToolResult{}, fmt.Errorf("attachment ID is not available in this conversation; call attachment_read without an ID to list files")
			}
			cached, err := s.extractDocumentAttachment(ctx, *found)
			if err != nil {
				return registeredToolResult{}, err
			}
			text := []rune(cached.Text)
			start := min(args.Offset, len(text))
			end := min(start+12000, len(text))
			response = map[string]any{"attachment_id": args.ID, "filename": found.Name, "offset": start, "total_chars": len(text), "content": string(text[start:end]), "has_more": end < len(text)}
			if end < len(text) {
				response["next_offset"] = end
			}
		}
		b, _ := json.Marshal(response)
		return registeredToolResult{Result: string(b)}, nil
	})
	registry.prompts = append(registry.prompts, strings.Join([]string{
		"Uploaded document/ZIP attachments are accessible through document_attachment blocks and attachment_read.",
		"Those blocks contain actual extracted uploaded data, not hypothetical file descriptions. Prior URL/SSH download failures do not invalidate a new upload.",
		"If a block is truncated, use attachment_read with its attachment_id and next_offset. Never invent an SSH path or claim attachments are inaccessible without checking this tool.",
	}, " "))
}
