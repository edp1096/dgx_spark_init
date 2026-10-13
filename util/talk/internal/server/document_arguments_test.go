package server

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"sparktalk/internal/llm"
)

func TestDocumentToolSingleSourceAliasAndCanonicalRetry(t *testing.T) {
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) { mockDocumentResponse(w, r, true) })
	id := fixtureSource(t, s, "A complete paragraph.")
	registry := completionToolRegistry{handlers: map[string]registeredToolHandler{}}
	s.registerDocumentTasks(&registry, "s")
	args, _ := json.Marshal(map[string]any{"action": "start", "kind": "translate", "source_id": id, "instructions": "전체 번역"})
	ctx := context.WithValue(context.Background(), sessionTitleRequestKey{}, "전체 번역")
	result, err := registry.execute(ctx, llm.ToolCall{Function: llm.FunctionCall{Name: "document_task", Arguments: string(args)}}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if result.DeferredDocument == nil || len(result.DeferredDocument.Document.Sources) != 1 || result.DeferredDocument.Document.Sources[0] != id {
		t.Fatal("single source alias not normalized")
	}
	if run := waitDocument(t, s, result.DeferredDocument.ID); run.Status != "completed" {
		t.Fatalf("alias did not process: %+v", run)
	}
	first, err := s.startDocument(context.Background(), "s", documentStart{Kind: "translate", SourceID: id, Key: "same"})
	if err != nil {
		t.Fatal(err)
	}
	second, err := s.startDocument(context.Background(), "s", documentStart{Kind: "translate", Sources: []string{id}, Key: "same"})
	if err != nil || first.ID != second.ID {
		t.Fatal("canonical retry created another run", err)
	}
	waitDocument(t, s, first.ID)
}

func TestDocumentToolArgumentErrorsExplainTheActualProblem(t *testing.T) {
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) { t.Error("invalid arguments reached inference") })
	id := fixtureSource(t, s, "Complete source.")
	registry := completionToolRegistry{handlers: map[string]registeredToolHandler{}}
	s.registerDocumentTasks(&registry, "s")
	for _, tc := range []struct {
		args map[string]any
		want string
	}{
		{map[string]any{"action": "start", "kind": "translate"}, "requires sources"},
		{map[string]any{"action": "start", "kind": "translate", "sources": []string{""}}, "nonempty"},
		{map[string]any{"action": "start", "kind": "translate", "sources": make([]string, 9)}, "at most 8"},
		{map[string]any{"action": "start", "kind": "translate", "source_id": id, "instructions": strings.Repeat("x", 16001)}, "instructions exceed"},
		{map[string]any{"action": "start", "kind": "translate", "source_id": id, "sources": []string{"other"}}, "conflict"},
		{map[string]any{"action": "start", "kind": "translate", "source_id": id, "targets": []string{"_none"}}, `unknown field "targets"`},
	} {
		args, _ := json.Marshal(tc.args)
		ctx := context.WithValue(context.Background(), sessionTitleRequestKey{}, "전체 번역")
		_, err := registry.execute(ctx, llm.ToolCall{Function: llm.FunctionCall{Name: "document_task", Arguments: string(args)}}, nil, nil)
		if err == nil || !strings.Contains(err.Error(), tc.want) {
			t.Fatalf("expected %q, got %v", tc.want, err)
		}
	}
	if _, err := s.db.CreateSession("other", "other", "model", "none"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.startDocument(context.Background(), "other", documentStart{Kind: "translate", SourceID: id}); err == nil {
		t.Fatal("alias bypassed source session ownership")
	}
}
