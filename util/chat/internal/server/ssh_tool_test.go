package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"sparktalk/internal/db"
	"sparktalk/internal/extra"
	"sparktalk/internal/llm"
)

func TestSSHToolTrustsUnknownHostOnlyAfterApproval(t *testing.T) {
	trusted := false
	approvalCount := 0
	executionCount := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/ssh/check":
			if trusted {
				writeJSON(w, http.StatusOK, map[string]any{"status": "ok"})
			} else {
				writeJSON(w, http.StatusConflict, map[string]any{"error": "unknown host", "host_key": map[string]string{"fingerprint": "SHA256:test", "public_key": "ssh-ed25519 AAAAtest"}})
			}
		case "/v1/ssh/trust":
			trusted = true
			writeJSON(w, http.StatusOK, map[string]any{"status": "trusted"})
		case "/v1/ssh/exec":
			if !trusted {
				http.Error(w, "execution started before host key trust", http.StatusConflict)
				return
			}
			executionCount++
			w.Header().Set("Content-Type", "application/x-ndjson")
			fmt.Fprintln(w, `{"type":"start"}`)
			fmt.Fprintln(w, `{"type":"stdout","data":"hello\n"}`)
			fmt.Fprintln(w, `{"type":"exit","exit_code":0,"duration_ms":4}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer upstream.Close()

	store, err := db.Open(filepath.Join(t.TempDir(), "ssh-tool.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.CreateSession("session-1", "SSH test", "model", "low"); err != nil {
		t.Fatal(err)
	}
	if _, err := store.CreateSSHHost(db.SSHHost{ID: "ssh-1", Alias: "dgx-main", Name: "DGX Spark", Hostname: "192.168.100.61", Port: 22, Username: "edp1096", KeyID: "dgx-main", TimeoutSeconds: 60}); err != nil {
		t.Fatal(err)
	}
	s := &Server{db: store, extra: extra.New(upstream.URL), approvals: make(map[string]*toolApproval)}
	var sawHostKey, sawResolution bool
	emit := func(event string, payload any) error {
		data, _ := payload.(map[string]any)
		switch event {
		case "tool_approval":
			approvalCount++
			hostKey, ok := data["host_key"].(*extra.HostKey)
			if !ok || hostKey.Fingerprint != "SHA256:test" {
				t.Fatalf("missing host key in approval: %#v", data["host_key"])
			}
			sawHostKey = true
			s.approvals[data["approval_id"].(string)].decision <- approvalConversation
		case "tool_approval_resolved":
			sawResolution = true
		}
		return nil
	}
	call := llm.ToolCall{ID: "call-1", Function: llm.FunctionCall{Name: "ssh_exec", Arguments: `{"host":"dgx-main","command":"printf hello","reason":"test"}`}}
	result, err := s.executeSSHTool(context.Background(), "session-1", call, emit)
	if err != nil {
		t.Fatal(err)
	}
	// A newly constructed server represents an application restart. The grant
	// must come from SQLite rather than process memory.
	s = &Server{db: store, extra: extra.New(upstream.URL), approvals: make(map[string]*toolApproval)}
	call.ID = "call-2"
	if _, err := s.executeSSHTool(context.Background(), "session-1", call, emit); err != nil {
		t.Fatal(err)
	}
	if !trusted || !sawHostKey || !sawResolution || approvalCount != 1 || executionCount != 2 || !strings.Contains(result, `"stdout":"hello\n"`) {
		t.Fatalf("trusted=%v hostKey=%v resolved=%v approvals=%d executions=%d result=%s", trusted, sawHostKey, sawResolution, approvalCount, executionCount, result)
	}
	var decoded map[string]any
	if err := json.Unmarshal([]byte(result), &decoded); err != nil || decoded["exit_code"] != float64(0) {
		t.Fatalf("invalid result: %s", result)
	}
}

func TestSSHConversationGrantAPIListsAndRevokes(t *testing.T) {
	store, err := db.Open(filepath.Join(t.TempDir(), "ssh-grant-api.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.CreateSession("session-1", "SSH", "model", "low"); err != nil {
		t.Fatal(err)
	}
	if _, err := store.CreateSSHHost(db.SSHHost{ID: "host-1", Alias: "main", Name: "Main", Hostname: "192.0.2.10", Port: 22, Username: "user", KeyID: "main", TimeoutSeconds: 60}); err != nil {
		t.Fatal(err)
	}
	if err := store.GrantSSHConversation("session-1", "host-1"); err != nil {
		t.Fatal(err)
	}
	s := &Server{db: store}

	response := httptest.NewRecorder()
	s.session(response, httptest.NewRequest(http.MethodGet, "/api/sessions/session-1/ssh-grants", nil))
	if response.Code != http.StatusOK || !strings.Contains(response.Body.String(), `"host_alias":"main"`) {
		t.Fatalf("list status=%d body=%s", response.Code, response.Body.String())
	}
	response = httptest.NewRecorder()
	s.session(response, httptest.NewRequest(http.MethodDelete, "/api/sessions/session-1/ssh-grants/host-1", nil))
	if response.Code != http.StatusNoContent {
		t.Fatalf("delete status=%d body=%s", response.Code, response.Body.String())
	}
	has, err := store.HasSSHConversationGrant("session-1", "host-1")
	if err != nil || has {
		t.Fatalf("grant after revoke=%v err=%v", has, err)
	}
}
