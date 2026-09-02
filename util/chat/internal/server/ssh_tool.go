package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"sparktalk/internal/extra"
	"sparktalk/internal/llm"
)

func (s *Server) executeSSHTool(ctx context.Context, sessionID string, call llm.ToolCall, emit eventEmitter) (string, error) {
	var arguments struct {
		Host    string `json:"host"`
		Command string `json:"command"`
		Reason  string `json:"reason"`
	}
	if err := json.Unmarshal([]byte(call.Function.Arguments), &arguments); err != nil {
		return "", errors.New("ssh_exec received invalid arguments")
	}
	arguments.Host = strings.TrimSpace(arguments.Host)
	arguments.Command = strings.TrimSpace(arguments.Command)
	arguments.Reason = strings.TrimSpace(arguments.Reason)
	if arguments.Host == "" || arguments.Command == "" {
		return "", errors.New("ssh_exec requires host and command")
	}
	host, err := s.db.SSHHostByAlias(arguments.Host)
	if err != nil {
		return "", err
	}
	target := extra.Target{Host: host.Hostname, Port: host.Port, User: host.Username, KeyID: host.KeyID}
	var untrustedHostKey *extra.HostKey
	if err := s.extraSnapshot().Check(ctx, target); err != nil {
		var apiErr *extra.HTTPError
		if errors.As(err, &apiErr) && apiErr.Status == 409 && apiErr.HostKey != nil {
			untrustedHostKey = apiErr.HostKey
		} else {
			return "", fmt.Errorf("SparkTalk Extra SSH connection check: %w", err)
		}
	}
	approval := map[string]any{
		"name": "ssh_exec", "host": host.Alias, "host_name": host.Name,
		"host_id": host.ID, "command": arguments.Command, "reason": arguments.Reason,
	}
	if untrustedHostKey != nil {
		approval["host_key"] = untrustedHostKey
	}
	approval["conversation_scope_available"] = sessionID != ""
	conversationGranted := false
	if sessionID != "" {
		conversationGranted, err = s.db.HasSSHConversationGrant(sessionID, host.ID)
		if err != nil {
			return "", fmt.Errorf("load SSH conversation permission: %w", err)
		}
	}
	if !conversationGranted || untrustedHostKey != nil {
		decision, err := s.awaitToolApproval(ctx, call.ID, approval, emit)
		_ = s.db.AddToolAudit(sessionID, "ssh_exec", host.ID, "execute", string(decision), "")
		if err != nil {
			return "", err
		}
		if decision == approvalConversation {
			if err := s.db.GrantSSHConversation(sessionID, host.ID); err != nil {
				return "", fmt.Errorf("save SSH conversation permission: %w", err)
			}
			if err := emit("ssh_grant_changed", map[string]any{"host_id": host.ID, "host": host.Alias, "host_name": host.Name}); err != nil {
				return "", err
			}
		}
	} else {
		_ = s.db.AddToolAudit(sessionID, "ssh_exec", host.ID, "execute", "automatic", "")
		if err := emit("tool_approval_resolved", map[string]any{"id": call.ID, "approved": true, "decision": approvalConversation, "automatic": true}); err != nil {
			return "", err
		}
	}
	if untrustedHostKey != nil {
		if _, err := s.extraSnapshot().Trust(ctx, host.Hostname, host.Port, untrustedHostKey.PublicKey); err != nil {
			return "", fmt.Errorf("SparkTalk Extra SSH host key trust: %w", err)
		}
		if err := emit("tool_execution", map[string]any{"id": call.ID, "status": "host_trusted", "fingerprint": untrustedHostKey.Fingerprint}); err != nil {
			return "", err
		}
	}
	request := extra.ExecRequest{
		Target:  target,
		Command: arguments.Command, TimeoutSeconds: host.TimeoutSeconds,
	}
	result, err := s.extraSnapshot().Execute(ctx, request, func(event extra.Event) error {
		switch event.Type {
		case "stdout", "stderr":
			return emit("tool_output", map[string]any{"id": call.ID, "stream": event.Type, "delta": event.Data})
		case "start":
			return emit("tool_execution", map[string]any{"id": call.ID, "status": "running"})
		}
		return nil
	})
	if err != nil {
		_ = s.db.AddToolAudit(sessionID, "ssh_exec", host.ID, "execute", "execution_error", compactHistoryText(err.Error(), 500))
		return "", fmt.Errorf("SparkTalk Extra SSH: %w", err)
	}
	_ = s.db.AddToolAudit(sessionID, "ssh_exec", host.ID, "execute", "executed", fmt.Sprintf("exit=%d duration_ms=%d", result.ExitCode, result.DurationMS))
	payload := map[string]any{
		"host": host.Alias, "host_name": host.Name, "command": arguments.Command,
		"stdout": result.Stdout, "stderr": result.Stderr, "exit_code": result.ExitCode,
		"duration_ms": result.DurationMS, "truncated": result.Truncated,
	}
	if result.Error != "" {
		payload["execution_error"] = result.Error
	}
	data, _ := json.Marshal(payload)
	return string(data), nil
}
