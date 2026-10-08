package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	supportssh "sparktalk/internal/support/ssh"
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
	host, err := s.authorizeSSHTool(ctx, sessionID, call, arguments.Host, arguments.Command, arguments.Reason, "execute", emit)
	if err != nil {
		return "", err
	}
	target := supportssh.Target{Host: host.Hostname, Port: host.Port, User: host.Username, KeyID: host.KeyID}
	request := supportssh.ExecRequest{
		Target:  target,
		Command: arguments.Command, TimeoutSeconds: host.TimeoutSeconds,
	}
	result, err := s.sshSnapshot().Execute(ctx, request, func(event supportssh.Event) error {
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

// Both commands and downloads use the same persisted per-host conversation grant
// and host-key trust flow. A grant never bypasses host-key verification.
func (s *Server) authorizeSSHTool(ctx context.Context, sessionID string, call llm.ToolCall, alias, description, reason, action string, emit eventEmitter) (db.SSHHost, error) {
	host, err := s.db.SSHHostByAlias(alias)
	if err != nil {
		return db.SSHHost{}, err
	}
	target := supportssh.Target{Host: host.Hostname, Port: host.Port, User: host.Username, KeyID: host.KeyID}
	var untrustedHostKey *supportssh.HostKey
	if err := s.sshSnapshot().Check(ctx, target); err != nil {
		var apiErr *supportssh.HTTPError
		if errors.As(err, &apiErr) && apiErr.Status == 409 && apiErr.HostKey != nil {
			untrustedHostKey = apiErr.HostKey
		} else {
			return db.SSHHost{}, fmt.Errorf("SparkTalk Extra SSH connection check: %w", err)
		}
	}
	approval := map[string]any{
		"name": call.Function.Name, "host": host.Alias, "host_name": host.Name,
		"host_id": host.ID, "command": description, "reason": reason,
	}
	if untrustedHostKey != nil {
		approval["host_key"] = untrustedHostKey
	}
	approval["conversation_scope_available"] = sessionID != ""
	conversationGranted := false
	if sessionID != "" {
		conversationGranted, err = s.db.HasSSHConversationGrant(sessionID, host.ID)
		if err != nil {
			return db.SSHHost{}, fmt.Errorf("load SSH conversation permission: %w", err)
		}
	}
	if !conversationGranted || untrustedHostKey != nil {
		decision, err := s.awaitToolApproval(ctx, call.ID, approval, emit)
		_ = s.db.AddToolAudit(sessionID, call.Function.Name, host.ID, action, string(decision), "")
		if err != nil {
			return db.SSHHost{}, err
		}
		if decision == approvalConversation {
			if err := s.db.GrantSSHConversation(sessionID, host.ID); err != nil {
				return db.SSHHost{}, fmt.Errorf("save SSH conversation permission: %w", err)
			}
			if err := emit("ssh_grant_changed", map[string]any{"host_id": host.ID, "host": host.Alias, "host_name": host.Name}); err != nil {
				return db.SSHHost{}, err
			}
		}
	} else {
		_ = s.db.AddToolAudit(sessionID, call.Function.Name, host.ID, action, "automatic", "")
		if err := emit("tool_approval_resolved", map[string]any{"id": call.ID, "approved": true, "decision": approvalConversation, "automatic": true}); err != nil {
			return db.SSHHost{}, err
		}
	}
	if untrustedHostKey != nil {
		if _, err := s.sshSnapshot().Trust(ctx, host.Hostname, host.Port, untrustedHostKey.PublicKey); err != nil {
			return db.SSHHost{}, fmt.Errorf("SparkTalk Extra SSH host key trust: %w", err)
		}
		if err := emit("tool_execution", map[string]any{"id": call.ID, "status": "host_trusted", "fingerprint": untrustedHostKey.Fingerprint}); err != nil {
			return db.SSHHost{}, err
		}
	}

	return host, nil
}
