package server

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
)

func TestCodeProjectLegacyImportToolEditAndRestore(t *testing.T) {
	s, _ := testImageServer(t)
	content := "```html\n<!doctype html><title>TETRIS</title><h1>RED</h1>\n```\n```css\nh1 {color:red}\n```"
	original, err := s.db.AddMessage("session", "assistant", content, "", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	reg := newCompletionToolRegistry(s, "session", config.ToolsConfig{}, false, nil)
	if reg.err != nil {
		t.Fatal(reg.err)
	}
	items, err := s.db.Artifacts("session")
	if err != nil || len(items) != 1 {
		t.Fatalf("legacy import %+v %v", items, err)
	}
	id := items[0].ID
	call := func(args string) registeredToolResult {
		t.Helper()
		r, e := reg.handlers["code_project"](context.Background(), llm.ToolCall{ID: "edit", Function: llm.FunctionCall{Name: "code_project", Arguments: args}}, nil, func(event string, payload any) error {
			if event == "artifact_updated" {
				a := payload.(db.Artifact)
				saved, e := s.db.Artifact("session", a.ID, 0)
				if e != nil || saved.Version != a.Version {
					t.Error("event emitted before persistence")
				}
			}
			return nil
		})
		if e != nil {
			t.Fatal(e)
		}
		return r
	}
	raw, _ := json.Marshal(map[string]any{"action": "edit", "project_id": id, "base_version": 1, "summary": "색상 개선", "edits": []db.ArtifactEdit{{Name: "index.html", Operation: "replace", Old: "RED", New: "BLUE"}, {Name: "style.css", Operation: "replace", Old: "red", New: "blue"}}})
	call(string(raw))
	again := newCompletionToolRegistry(s, "session", config.ToolsConfig{}, false, nil)
	if again.err != nil {
		t.Fatal(again.err)
	}
	items, _ = s.db.Artifacts("session")
	if len(items) != 1 || items[0].Version != 2 {
		t.Fatalf("pool identity lost: %+v", items)
	}
	messages, _ := s.db.Messages("session")
	if messages[len(messages)-1].ID != original.ID || messages[len(messages)-1].Content != content {
		t.Fatal("legacy message changed")
	}
	w := httptest.NewRecorder()
	s.artifactAPI(w, httptest.NewRequest("POST", "/api/artifacts/"+id+"/restore?session_id=session", strings.NewReader(`{"base_version":2,"version":1}`)))
	if w.Code != 200 {
		t.Fatalf("restore %d %s", w.Code, w.Body.String())
	}
	var restored db.Artifact
	json.Unmarshal(w.Body.Bytes(), &restored)
	if restored.Version != 3 || !strings.Contains(restored.Files[0].Source, "RED") {
		t.Fatalf("bad restore: %+v", restored)
	}
	w = httptest.NewRecorder()
	s.artifactAPI(w, httptest.NewRequest("POST", "/api/artifacts/"+id+"/restore?session_id=session", strings.NewReader(`{"base_version":2,"version":1}`)))
	if w.Code != 409 {
		t.Fatalf("stale restore status %d", w.Code)
	}
	s.db.CreateSession("other", "other", "", "")
	w = httptest.NewRecorder()
	s.artifactAPI(w, httptest.NewRequest("GET", "/api/artifacts/"+id+"?session_id=other", nil))
	if w.Code != 404 {
		t.Fatalf("cross-session read %d", w.Code)
	}
}

func TestCodeProjectRejectsMalformedWritesAndPreservesSource(t *testing.T) {
	s, _ := testImageServer(t)
	a, err := s.db.CreateArtifact("session", "p", "Game", "", []db.ArtifactFile{{Name: "index.html", Source: "<h1>original</h1>"}})
	if err != nil {
		t.Fatal(err)
	}
	reg := newCompletionToolRegistry(s, "session", config.ToolsConfig{}, false, nil)
	call := func(raw string) (registeredToolResult, error) {
		return reg.handlers["code_project"](context.Background(), llm.ToolCall{ID: "c", Function: llm.FunctionCall{Name: "code_project", Arguments: raw}}, nil, nil)
	}
	// Replay the exact shape responsible for the production data loss.
	result, err := call(`{"action":"edit","project_id":"p","base_version":1,"edits":[{"name":"index.html","operation":"delete"},{"name":"index.html","operation":"create","source":"<h1>recovered</h1>"}]}`)
	if err != nil {
		t.Fatal(err)
	}
	current, _ := s.db.Artifact("session", a.ID, 0)
	if current.Version != 2 || current.Files[0].Source != "<h1>recovered</h1>" {
		t.Fatalf("source discarded: %+v", current)
	}
	if !strings.Contains(result.Result, `"bytes":18`) || !strings.Contains(result.Result, `"sha256"`) {
		t.Fatalf("missing save receipt: %s", result.Result)
	}
	for _, edits := range []string{
		`[{"name":"index.html","operation":"delete"},{"name":"index.html","operation":"create"}]`,
		`[{"name":"index.html","operation":"delete"},{"name":"index.html","operation":"create","soruce":"lost"}]`,
		`[{"name":"index.html","operation":"write","source":null}]`,
		`[{"name":"index.html","operation":"write","source":"","new":"nonempty"}]`,
		`[{"name":"index.html","operation":"write","source":"   "}]`,
		`[{"name":"index.html","operation":"replace","old":"recovered"}]`,
		`[{"name":"index.html","old":"recovered","new":"new"}]`,
		`[{"name":"index.html","operation":"write","source":"<h1>recovered</h1>"}]`,
	} {
		_, err = call(`{"action":"edit","project_id":"p","base_version":2,"edits":` + edits + `}`)
		if err == nil {
			t.Fatalf("invalid write accepted: %s", edits)
		}
		saved, _ := s.db.Artifact("session", "p", 0)
		if saved.Version != 2 || saved.Files[0].Source != current.Files[0].Source {
			t.Fatalf("invalid write modified existing code: %+v", saved)
		}
	}
	_, err = call(`{"action":"edit","project_id":"p","base_version":2,"edits":[{"name":"index.html","operation":"write","source":"<h1>complete rewrite</h1>"}]}`)
	if err != nil {
		t.Fatal(err)
	}
	_, err = call(`{"action":"create","title":"missing","files":[{"name":"index.html"}]}`)
	if err == nil {
		t.Fatal("missing initial source accepted")
	}
	_, err = call(`{"action":"edit","project_id":"p","base_version":3,"files":[{"name":"index.html","source":"wrong location"}],"edits":[{"name":"index.html","operation":"delete"}]}`)
	if err == nil {
		t.Fatal("ignored top-level files")
	}
}
