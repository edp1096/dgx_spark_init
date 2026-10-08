package server

import (
	"context"
	"encoding/json"
	"fmt"
	"path"
	"strings"

	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	supportssh "sparktalk/internal/support/ssh"
)

func sshDownloadDefinition(hosts []db.SSHHost) llm.Tool {
	aliases := make([]string, 0, len(hosts))
	for _, h := range hosts {
		aliases = append(aliases, h.Alias)
	}
	parameters, _ := json.Marshal(map[string]any{"type": "object", "properties": map[string]any{
		"host":   map[string]any{"type": "string", "enum": aliases},
		"path":   map[string]any{"type": "string", "description": "Exact absolute path of an existing regular file on the registered SSH host; not a URL or directory"},
		"reason": map[string]any{"type": "string", "description": "Why this file should be delivered to the user"},
	}, "required": []string{"host", "path", "reason"}, "additionalProperties": false})
	return llm.Tool{Type: "function", Function: llm.ToolFunction{Name: "ssh_download", Description: "Copy a file (ZIP, document, text, source code or supported media, within the configured attachment limits) from a registered SSH host into SparkTalk and attach it to this conversation for browser download. Uses existing SSH approval. For a folder, first create a ZIP with ssh_exec. Returns a SparkTalk-relative download URL; use it exactly. No remote HTTP server, media_import, or user re-upload is needed.", Parameters: parameters}}
}

func (s *Server) executeSSHDownload(ctx context.Context, sessionID string, call llm.ToolCall, emit eventEmitter) (registeredToolResult, error) {
	if sessionID == "" || s.media == nil {
		return registeredToolResult{}, fmt.Errorf("ssh_download requires a conversation with attachment storage")
	}
	var args struct {
		Host   string `json:"host"`
		Path   string `json:"path"`
		Reason string `json:"reason"`
	}
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
		return registeredToolResult{}, fmt.Errorf("invalid ssh_download arguments")
	}
	args.Host = strings.TrimSpace(args.Host)
	if args.Host == "" || !path.IsAbs(args.Path) || len(args.Path) > 4096 || strings.ContainsFunc(args.Path, func(r rune) bool { return r < 32 || r == 127 }) {
		return registeredToolResult{}, fmt.Errorf("ssh_download requires a registered host and an absolute file path without control characters")
	}
	host, err := s.authorizeSSHTool(ctx, sessionID, call, args.Host, args.Path, args.Reason, "download", emit)
	if err != nil {
		return registeredToolResult{}, err
	}
	result, err := s.sshSnapshot().Download(ctx, supportssh.DownloadRequest{Target: supportssh.Target{Host: host.Hostname, Port: host.Port, User: host.Username, KeyID: host.KeyID}, Path: args.Path, MaxBytes: s.media.Limits().MaxBytes(), TimeoutSeconds: host.TimeoutSeconds})
	if err != nil {
		_ = s.db.AddToolAudit(sessionID, "ssh_download", host.ID, "download", "execution_error", compactHistoryText(err.Error(), 500))
		return registeredToolResult{}, fmt.Errorf("Extra SSH download: %w", err)
	}
	defer result.Body.Close()
	item, err := s.media.SaveReader(result.Body, result.Name, "", s.media.Limits().MaxBytes())
	if err != nil {
		_ = s.db.AddToolAudit(sessionID, "ssh_download", host.ID, "download", "execution_error", compactHistoryText(err.Error(), 500))
		return registeredToolResult{}, err
	}
	_ = s.db.AddToolAudit(sessionID, "ssh_download", host.ID, "download", "executed", fmt.Sprintf("bytes=%d sha256=%s", item.Size, result.SHA256))
	payload, _ := json.Marshal(map[string]any{"host": host.Alias, "path": args.Path, "attachment": item, "sha256": result.SHA256, "status": "copied to SparkTalk and attached; use attachment.url for browser download"})
	return registeredToolResult{Result: string(payload), Attachments: []db.Attachment{item}}, nil
}
