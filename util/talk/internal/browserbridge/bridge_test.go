package browserbridge

import (
	"context"
	"encoding/json"
	"golang.org/x/net/websocket"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func setup(t *testing.T) (*Bridge, *httptest.Server) {
	t.Helper()
	b, e := New(filepath.Join(t.TempDir(), "key"))
	if e != nil {
		t.Fatal(e)
	}
	b.token = strings.Repeat("a", 64)
	mux := http.NewServeMux()
	b.Register(mux)
	s := httptest.NewServer(mux)
	t.Cleanup(s.Close)
	return b, s
}
func dial(t *testing.T, s *httptest.Server, token string) *websocket.Conn {
	t.Helper()
	ws, e := websocket.Dial("ws"+strings.TrimPrefix(s.URL, "http")+"/api/browser/connect", "", "chrome-extension://"+strings.Repeat("a", 32))
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() { ws.Close() })
	websocket.JSON.Send(ws, map[string]string{"token": token})
	return ws
}
func TestBridgeRejectsBadToken(t *testing.T) {
	b, s := setup(t)
	ws := dial(t, s, "wrong")
	ws.SetReadDeadline(time.Now().Add(time.Second))
	var out any
	if websocket.JSON.Receive(ws, &out) == nil {
		t.Fatal("accepted bad token")
	}
	if b.Connected() {
		t.Fatal("bad connection installed")
	}
}
func TestBridgeRPCAndCancel(t *testing.T) {
	b, s := setup(t)
	ws := dial(t, s, b.token)
	var ready map[string]any
	if e := websocket.JSON.Receive(ws, &ready); e != nil {
		t.Fatal(e)
	}
	done := make(chan error, 1)
	go func() {
		raw, e := b.Call(context.Background(), "tabs", nil)
		if e == nil && string(raw) != `{"ok":true}` {
			e = context.Canceled
		}
		done <- e
	}()
	var request struct {
		ID     string `json:"id"`
		Action string `json:"action"`
	}
	if e := websocket.JSON.Receive(ws, &request); e != nil {
		t.Fatal(e)
	}
	if request.Action != "tabs" {
		t.Fatal(request)
	}
	websocket.JSON.Send(ws, map[string]any{"id": request.ID, "result": json.RawMessage(`{"ok":true}`)})
	if e := <-done; e != nil {
		t.Fatal(e)
	}
	ctx, cancel := context.WithCancel(context.Background())
	go func() { _, e := b.Call(ctx, "submit", nil); done <- e }()
	websocket.JSON.Receive(ws, &request)
	cancel()
	var cancelled map[string]string
	websocket.JSON.Receive(ws, &cancelled)
	if cancelled["cancel"] != request.ID {
		t.Fatal(cancelled)
	}
	if <-done == nil {
		t.Fatal("cancel not reported")
	}
}
func TestPairingRejectsCrossOrigin(t *testing.T) {
	b, _ := setup(t)
	r := httptest.NewRequest("POST", "http://talk/api/browser", nil)
	r.Header.Set("Origin", "https://evil.example")
	w := httptest.NewRecorder()
	b.settings(w, r)
	if w.Code != 403 {
		t.Fatal(w.Code)
	}
}
func TestBridgeDisconnectFailsPending(t *testing.T) {
	b, s := setup(t)
	ws := dial(t, s, b.token)
	var ready any
	websocket.JSON.Receive(ws, &ready)
	done := make(chan error, 1)
	go func() { _, e := b.Call(context.Background(), "submit", nil); done <- e }()
	var req any
	websocket.JSON.Receive(ws, &req)
	ws.Close()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("missing disconnect error")
		}
	case <-time.After(time.Second):
		t.Fatal("request hung")
	}
}

func TestBridgeSessionIsOnWire(t *testing.T) {
	b, s := setup(t)
	ws := dial(t, s, b.token)
	var ready any
	if err := websocket.JSON.Receive(ws, &ready); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	done := make(chan error, 1)
	go func() {
		_, err := b.CallSession(ctx, "conversation-a", "browser", map[string]any{"action": "observe", "tab_id": 17})
		done <- err
	}()
	var req struct {
		ID      string `json:"id"`
		Session string `json:"session_id"`
		Action  string `json:"action"`
	}
	if err := websocket.JSON.Receive(ws, &req); err != nil {
		t.Fatal(err)
	}
	if req.Session != "conversation-a" || req.Action != "browser" {
		t.Fatalf("wrong routing: %+v", req)
	}
	if err := websocket.JSON.Send(ws, map[string]any{"id": req.ID, "result": map[string]any{"ok": true}}); err != nil {
		t.Fatal(err)
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}
