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
		Name: "attachment_read", Description: "List all attachments in this conversation, including video/audio/images. Read documents as text; read media by attachment_id to load visuals/transcript using the current model transport. Omit attachment_id to list available files. Use the exact returned ID; offset pages through extracted text. Uploaded attachments exist in SparkTalk storage, not necessarily on an SSH host. Never ask for another upload merely because an SSH /tmp path is missing.",
		Parameters: json.RawMessage(`{"type":"object","properties":{"attachment_id":{"type":"string"},"offset":{"type":"integer","minimum":0},"source_id":{"type":"string"}},"additionalProperties":false}`),
	}}, func(ctx context.Context, call llm.ToolCall, _ []llm.Message, _ eventEmitter) (registeredToolResult, error) {
		var args struct {
			ID     string `json:"attachment_id"`
			Offset int    `json:"offset"`
			Source string `json:"source_id"`
		}
		if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
			return registeredToolResult{}, err
		}
		if args.Offset < 0 {
			return registeredToolResult{}, fmt.Errorf("offset must be nonnegative")
		}
		if args.Source != "" {
			book, err := s.db.Source(sessionID, args.Source, false)
			if err != nil {
				return registeredToolResult{}, err
			}
			result, err := sourcePage(args.Source, book, args.Offset, 12000)
			return registeredToolResult{Result: result}, err
		}
		attachments, err := s.conversationAttachments(ctx, sessionID)
		if err != nil {
			return registeredToolResult{}, err
		}
		files := []db.Attachment{}
		generated := false
		for _, item := range attachments {
			files = append(files, item.Attachment)
			if item.Attachment.ID == args.ID && (item.Origin == "assistant" || item.Origin == "generated_this_turn") {
				generated = true
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

			if !isDocumentAttachment(*found) {
				if generated {
					ctx = context.WithValue(ctx, imageAttachmentOriginKey{}, "assistant")
				}
				cfg, _ := s.snapshot()
				followups, err := s.llmMessages(ctx, []db.Message{{Role: "user", Content: "Existing conversation attachment loaded for analysis. Use the supplied visuals/transcript; no SSH download is needed.", Attachments: []db.Attachment{*found}}}, cfg)
				if err != nil {
					return registeredToolResult{}, err
				}
				b, _ := json.Marshal(map[string]any{"attachment": found, "status": "loaded from SparkTalk storage into model input"})
				return registeredToolResult{Result: string(b), Followups: followups}, nil
			}
			id, book, err := s.attachmentSource(ctx, sessionID, *found)
			if err != nil {
				return registeredToolResult{}, err
			}
			result, err := sourcePage(id, book, args.Offset, 12000)
			return registeredToolResult{Result: result}, err
		}
		b, _ := json.Marshal(response)
		return registeredToolResult{Result: string(b)}, nil
	})
	registry.prompts = append(registry.prompts, strings.Join([]string{
		"attachment_read lists all stored conversation attachments, including media. Media files use opaque IDs, not original filenames or .mp4 extensions on disk; do not search SSH hosts for them. The configured Extra Media service runs its own yt-dlp/ffmpeg; a host shell missing these commands does not mean that service is unavailable.",
		"Uploaded document/ZIP attachments are accessible through document_attachment blocks and attachment_read.",
		"Those blocks contain actual extracted uploaded data, not hypothetical file descriptions. Prior URL/SSH download failures do not invalidate a new upload.",
		"If a block is truncated, use attachment_read with its attachment_id and next_offset. Never invent an SSH path or claim attachments are inaccessible without checking this tool.",
	}, " "))
}
