package server

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/media"
	supportssh "sparktalk/internal/support/ssh"
)

func TestSSHDownloadApprovalAttachmentAndFailure(t *testing.T) {
	var archive bytes.Buffer
	zw := zip.NewWriter(&archive)
	f, _ := zw.Create("manifest.json")
	_, _ = f.Write([]byte(`{"name":"test"}`))
	_ = zw.Close()
	data := archive.Bytes()
	corrupt := false
	downloads := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/ssh/check":
			writeJSON(w, 200, map[string]string{"status": "ok"})
		case "/v1/ssh/download":
			downloads++
			var req supportssh.DownloadRequest
			if json.NewDecoder(r.Body).Decode(&req) != nil || req.Host != "192.0.2.25" || req.Path != "/work/result.zip" || req.KeyID != "test" {
				t.Error("wrong remote target")
			}
			w.Header().Set("Content-Length", strconv.Itoa(len(data)))
			w.Header().Set("Content-Disposition", `attachment; filename="result.zip"`)
			hash := fmt.Sprintf("%x", sha256.Sum256(data))
			if corrupt {
				hash = "bad"
			}
			w.Header().Set("X-Content-SHA256", hash)
			_, _ = w.Write(data)
		default:
			t.Error("unexpected route: " + r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	defer upstream.Close()
	database := filepath.Join(t.TempDir(), "talk.db")
	store, err := db.Open(database)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	_, _ = store.CreateSession("chat", "Download", "model", "low")
	_, err = store.CreateSSHHost(db.SSHHost{ID: "host", Alias: "work", Name: "Work", Hostname: "192.0.2.25", Port: 22, Username: "user", KeyID: "test", TimeoutSeconds: 60})
	if err != nil {
		t.Fatal(err)
	}
	ms, err := media.New(database)
	if err != nil {
		t.Fatal(err)
	}
	s := &Server{db: store, media: ms, sshClient: supportssh.New(upstream.URL), approvals: make(map[string]*toolApproval)}
	s.cfg.Extra.SSHEnabled = true
	reg := newCompletionToolRegistry(s, "chat", s.cfg.Tools, false, nil)
	if reg.handlers["ssh_download"] == nil {
		t.Fatal("download tool not registered")
	}
	decision := approvalReject
	approvals := 0
	emit := func(event string, payload any) error {
		if event == "tool_approval" {
			approvals++
			p := payload.(map[string]any)
			if p["name"] != "ssh_download" || p["command"] != "/work/result.zip" {
				t.Error("incorrect approval")
			}
			s.approvals[p["approval_id"].(string)].decision <- decision
		}
		return nil
	}
	call := llm.ToolCall{ID: "download-1", Function: llm.FunctionCall{Name: "ssh_download", Arguments: `{"host":"work","path":"/work/result.zip","reason":"deliver generated ZIP"}`}}
	if out, err := reg.execute(context.Background(), call, nil, emit); err == nil || len(out.Attachments) != 0 || downloads != 0 {
		t.Fatalf("denied download ran: %+v %v", out, err)
	}
	decision = approvalConversation
	call.ID = "download-2"
	out, err := reg.execute(context.Background(), call, nil, emit)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Attachments) != 1 {
		t.Fatalf("missing assistant attachment: %+v", out)
	}
	item := out.Attachments[0]
	if !strings.HasPrefix(item.URL, "/api/files/") || item.Name != "result.zip" {
		t.Fatalf("wrong download URL: %+v", item)
	}
	w := httptest.NewRecorder()
	s.file(w, httptest.NewRequest("GET", item.URL, nil))
	if w.Code != 200 || !bytes.Equal(w.Body.Bytes(), data) || !strings.HasPrefix(w.Header().Get("Content-Disposition"), "attachment;") {
		t.Fatal("same-origin download bytes differ")
	}
	// Existing conversation approval is reused; another session has no such grant.
	call.ID = "download-3"
	if _, err := reg.execute(context.Background(), call, nil, emit); err != nil || approvals != 2 {
		t.Fatalf("grant not reused: %v approvals=%d", err, approvals)
	}
	_, _ = store.CreateSession("other", "Other", "model", "low")
	decision = approvalReject
	if _, err := s.executeSSHDownload(context.Background(), "other", call, emit); err == nil {
		t.Fatal("cross-session permission reused")
	}
	before, _ := os.ReadDir(database + ".media")
	corrupt = true
	if out, err := reg.execute(context.Background(), call, nil, emit); err == nil || len(out.Attachments) != 0 {
		t.Fatal("corrupt transfer attached")
	}
	after, _ := os.ReadDir(database + ".media")
	if len(before) != len(after) {
		t.Fatal("corrupt transfer left a stored file")
	}
	s.cfg.Extra.SSHEnabled = false
	if newCompletionToolRegistry(s, "chat", s.cfg.Tools, false, nil).handlers["ssh_download"] != nil {
		t.Fatal("disabled SSH exposed download")
	}
}
