package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/plugins"
)

func pluginServer(t *testing.T) *Server {
	t.Helper()
	d, err := db.Open(filepath.Join(t.TempDir(), "talk.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { d.Close() })
	definition := plugins.Definition{Manifest: plugins.Manifest{ID: "fixture", Name: "Fixture", Version: "1.0.0", API: 1, DataVersion: 1, Permissions: []string{"tools"}, Operations: []plugins.Operation{{Name: "echo", Description: "Echo", Parameters: json.RawMessage(`{"type":"object"}`), Tool: true, TimeoutSeconds: 1}}, Panels: []plugins.Panel{{ID: "fixture", Title: "Fixture", Operations: []string{"echo"}}}}, Handle: func(_ context.Context, _ plugins.Host, _ string, r plugins.Request) (json.RawMessage, error) {
		return json.Marshal(map[string]any{"session": r.SessionID, "input": r.Input})
	}}
	m, err := plugins.New(context.Background(), d, nil, []plugins.Definition{definition})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close(context.Background()) })
	return &Server{db: d, plugins: m}
}
func pluginRequest(s *Server, method, path, body, origin string) *httptest.ResponseRecorder {
	r := httptest.NewRequest(method, path, strings.NewReader(body))
	r.Header.Set("Content-Type", "application/json")
	if origin != "" {
		r.Header.Set("Origin", origin)
	}
	w := httptest.NewRecorder()
	s.pluginAPI(w, r)
	return w
}
func TestPluginHTTPAndToolRegistryLifecycle(t *testing.T) {
	s := pluginServer(t)
	if w := pluginRequest(s, "GET", "/api/plugins", "", ""); w.Code != 200 || !strings.Contains(w.Body.String(), "fixture") {
		t.Fatal(w.Code, w.Body.String())
	}
	if w := pluginRequest(s, "POST", "/api/plugins/fixture/enable", "{}", ""); w.Code != 403 {
		t.Fatal("activation bypassed grants", w.Code)
	}
	if w := pluginRequest(s, "PUT", "/api/plugins/fixture/configure", `{"config":{},"grants":["tools"]}`, "https://untrusted.example"); w.Code != 403 {
		t.Fatal("cross-origin mutation accepted")
	}
	if w := pluginRequest(s, "PUT", "/api/plugins/fixture/configure", `{"config":{},"grants":["tools"]}`, ""); w.Code != 200 {
		t.Fatal(w.Body.String())
	}
	if w := pluginRequest(s, "POST", "/api/plugins/fixture/enable", "{}", ""); w.Code != 200 {
		t.Fatal(w.Body.String())
	}
	registry := newCompletionToolRegistry(s, "", config.ToolsConfig{}, false, nil)
	name := plugins.ToolName("fixture", "echo")
	got, err := registry.execute(context.Background(), llm.ToolCall{Function: llm.FunctionCall{Name: name, Arguments: `{"text":"hello"}`}}, nil, nil)
	if err != nil || !strings.Contains(got.Result, "hello") {
		t.Fatal(got, err)
	}
	if w := pluginRequest(s, "GET", "/api/plugins/fixture/runs", "", ""); w.Code != 200 || !strings.Contains(w.Body.String(), "completed") {
		t.Fatal(w.Body.String())
	}
	if w := pluginRequest(s, "POST", "/api/plugins/fixture/disable", "{}", ""); w.Code != 200 {
		t.Fatal(w.Body.String())
	}
	if _, err = registry.execute(context.Background(), llm.ToolCall{Function: llm.FunctionCall{Name: name, Arguments: `{}`}}, nil, nil); !errors.Is(err, plugins.ErrDisabled) {
		t.Fatal("stale registry can execute", err)
	}
	fresh := newCompletionToolRegistry(s, "", config.ToolsConfig{}, false, nil)
	if fresh.handlers[name] != nil {
		t.Fatal("disabled tool remains visible")
	}
}
func TestPluginHTTPRejectsUnknownAndOversizedInput(t *testing.T) {
	s := pluginServer(t)
	for _, body := range []string{`{"unknown":true}`, `{} {}`, `{"config":"` + strings.Repeat("x", plugins.MaxPayload+4096) + `"}`} {
		w := pluginRequest(s, http.MethodPut, "/api/plugins/fixture/configure", body, "")
		if w.Code < 400 {
			t.Fatal("bad input accepted")
		}
	}
	if w := pluginRequest(s, "POST", "/api/plugins/not_installed/enable", "{}", ""); w.Code != 404 {
		t.Fatal(w.Code)
	}
}

func TestPluginModelServiceUsesConfiguredClientWithoutToolEscalation(t *testing.T) {
	model := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if tools, ok := body["tools"]; ok && tools != nil {
			t.Error("model-only service exposed tools", tools)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"fixture reply\"}}]}\n\ndata: [DONE]\n\n"))
	}))
	defer model.Close()
	s := pluginServer(t)
	s.llm = llm.New(model.URL, "test-model", "")
	s.cfg.Model.DefaultModel = "test-model"
	f := plugins.Definition{Manifest: plugins.Manifest{ID: "client", Name: "Client", Version: "1.0.0", API: 1, DataVersion: 1, Permissions: []string{"model.complete"}, Operations: []plugins.Operation{{Name: "ask", Parameters: json.RawMessage(`{"type":"object"}`), TimeoutSeconds: 2}}}, Handle: func(c context.Context, h plugins.Host, _ string, r plugins.Request) (json.RawMessage, error) {
		return h.Call(c, "model.complete", r)
	}}
	m, err := plugins.New(context.Background(), s.db, s.pluginServices(), []plugins.Definition{f})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close(context.Background())
	if err = m.Configure(context.Background(), "client", json.RawMessage(`{}`), f.Manifest.Permissions); err != nil {
		t.Fatal(err)
	}
	if err = m.Enable(context.Background(), "client"); err != nil {
		t.Fatal(err)
	}
	out, err := m.Call(context.Background(), "client", "ask", plugins.Request{Input: json.RawMessage(`{"prompt":"hello"}`)})
	if err != nil || !strings.Contains(string(out), "fixture reply") {
		t.Fatal(string(out), err)
	}
	s.generationMu.Lock()
	left := len(s.generations)
	s.generationMu.Unlock()
	if left != 0 {
		t.Fatal("generation lease leaked")
	}
}

func TestPluginBackgroundToolReturnsRunAndSurvivesRequestContext(t *testing.T) {
	s := pluginServer(t)
	started := make(chan struct{})
	release := make(chan struct{})
	f := plugins.Definition{Manifest: plugins.Manifest{ID: "background", Name: "Background", Version: "1.0.0", API: 1, DataVersion: 1, Permissions: []string{"jobs", "tools"}, Operations: []plugins.Operation{{Name: "work", Parameters: json.RawMessage(`{"type":"object"}`), Tool: true, Background: true, TimeoutSeconds: 2}}}, Handle: func(c context.Context, _ plugins.Host, _ string, r plugins.Request) (json.RawMessage, error) {
		close(started)
		select {
		case <-release:
			return json.RawMessage(`{"ok":true}`), nil
		case <-c.Done():
			return nil, c.Err()
		}
	}}
	m, err := plugins.New(context.Background(), s.db, s.pluginServices(), []plugins.Definition{f})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close(context.Background())
	s.plugins = m
	if err = m.Configure(context.Background(), "background", json.RawMessage(`{}`), f.Manifest.Permissions); err != nil {
		t.Fatal(err)
	}
	if err = m.Enable(context.Background(), "background"); err != nil {
		t.Fatal(err)
	}
	registry := newCompletionToolRegistry(s, "", config.ToolsConfig{}, false, nil)
	request, cancel := context.WithCancel(context.Background())
	result, err := registry.execute(request, llm.ToolCall{ID: "fixed-call", Function: llm.FunctionCall{Name: plugins.ToolName("background", "work"), Arguments: `{}`}}, nil, nil)
	cancel()
	if err != nil {
		t.Fatal(err)
	}
	var run plugins.Run
	if err = json.Unmarshal([]byte(result.Result), &run); err != nil || run.Status != "running" {
		t.Fatal(result, err)
	}
	<-started
	close(release)
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		rows, _ := m.Runs(context.Background(), "background")
		if len(rows) > 0 && rows[0].Status == "completed" {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatal("background job did not complete after caller disconnected")
}

func TestPluginPackageUploadGuards(t *testing.T) {
	s := pluginServer(t)
	for _, tc := range []struct {
		method, contentType, origin string
		status                      int
	}{
		{"GET", "application/zip", "", 405},
		{"POST", "text/plain", "", 400},
		{"POST", "application/zip", "https://untrusted.example", 403},
		{"POST", "application/zip", "", 400},
	} {
		r := httptest.NewRequest(tc.method, "/api/plugins/install", strings.NewReader("not a zip"))
		r.Header.Set("Content-Type", tc.contentType)
		r.Header.Set("Origin", tc.origin)
		w := httptest.NewRecorder()
		s.pluginAPI(w, r)
		if w.Code != tc.status {
			t.Fatalf("%+v got %d %s", tc, w.Code, w.Body.String())
		}
	}
}
