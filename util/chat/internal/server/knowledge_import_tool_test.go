package server

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/knowledge"
	"sparktalk/internal/llm"
)

func TestKnowledgeImportToolApprovesAndIndexesSources(t *testing.T) {
	collector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var input map[string]any
		if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
			t.Fatal(err)
		}
		var output bytes.Buffer
		archive := zip.NewWriter(&output)
		manifest, _ := archive.Create("manifest.json")
		_ = json.NewEncoder(manifest).Encode(map[string]any{
			"version": 1, "requested_url": input["url"], "final_url": input["url"],
			"title": "통풍 진료지침", "method": "direct", "content_type": "application/pdf",
			"raw_path": "raw/guideline.txt", "fetched_at": time.Now(),
		})
		raw, _ := archive.Create("raw/guideline.txt")
		_, _ = raw.Write([]byte("통풍 진단과 요산저하 치료에 관한 원문"))
		text, _ := archive.Create("normalized/text.txt")
		_, _ = text.Write([]byte("통풍 진단과 요산저하 치료에 관한 검색 본문"))
		_ = archive.Close()
		w.Header().Set("Content-Type", "application/zip")
		_, _ = w.Write(output.Bytes())
	}))
	defer collector.Close()

	databasePath := filepath.Join(t.TempDir(), "sparktalk.db")
	store, err := db.Open(databasePath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.CreateSession("session-1", "knowledge", "model", "low"); err != nil {
		t.Fatal(err)
	}
	fileStore, err := knowledge.New(databasePath)
	if err != nil {
		t.Fatal(err)
	}
	server := &Server{
		db: store, knowledge: fileStore, knowledgeIndex: &knowledge.Extractor{},
		collector: knowledge.NewCollectorClient(collector.URL), approvals: make(map[string]*toolApproval),
		cfg: config.Config{Extra: config.ExtraConfig{CollectorEndpoint: collector.URL}},
	}
	registry := newCompletionToolRegistry(server, "session-1", config.ToolsConfig{Enabled: true}, true, nil)
	if _, ok := registry.handlers["knowledge_import"]; !ok {
		t.Fatalf("knowledge import tool was not registered: %+v", registry.handlers)
	}

	call := llm.ToolCall{ID: "import-1", Function: llm.FunctionCall{Name: "knowledge_import", Arguments: `{"action":"import_urls","collection":"내 지식","urls":["https://example.com/gout.pdf"]}`}}
	result, err := executeApprovedKnowledgeImport(t, server, registry, call, approvalOnce)
	if err != nil || !strings.Contains(result, `"imported":1`) {
		t.Fatalf("result=%s err=%v", result, err)
	}
	documents, err := store.KnowledgeDocuments(1)
	if err != nil || len(documents) != 1 || documents[0].Status != "ready" {
		t.Fatalf("documents=%+v err=%v", documents, err)
	}
	results, err := store.SearchKnowledge("요산저하", 1, 5)
	if err != nil || len(results) != 1 {
		t.Fatalf("search=%+v err=%v", results, err)
	}

	rejected := llm.ToolCall{ID: "import-2", Function: llm.FunctionCall{Name: "knowledge_import", Arguments: `{"action":"import_urls","collection":"내 지식","urls":["https://example.com/rejected.pdf"]}`}}
	if _, err := executeApprovedKnowledgeImport(t, server, registry, rejected, approvalReject); err == nil {
		t.Fatal("rejected import unexpectedly succeeded")
	}
	documents, _ = store.KnowledgeDocuments(1)
	if len(documents) != 1 {
		t.Fatalf("rejected import changed stored documents: %+v", documents)
	}
}

func executeApprovedKnowledgeImport(t *testing.T, server *Server, registry completionToolRegistry, call llm.ToolCall, decision approvalDecision) (string, error) {
	t.Helper()
	type outcome struct {
		result string
		err    error
	}
	done := make(chan outcome, 1)
	go func() {
		execution, executeErr := registry.execute(context.Background(), call, nil, func(event string, payload any) error {
			if event != "tool_approval" {
				return nil
			}
			approval := payload.(map[string]any)
			if approval["approval_kind"] != "knowledge_import" || approval["collection_name"] != "내 지식" {
				t.Errorf("unexpected approval payload: %+v", approval)
			}
			body, _ := json.Marshal(map[string]string{"decision": string(decision)})
			response := httptest.NewRecorder()
			server.toolApproval(response, httptest.NewRequest(http.MethodPost, "/api/tool-approvals/"+approval["approval_id"].(string), strings.NewReader(string(body))))
			if response.Code != http.StatusOK {
				t.Errorf("approval status=%d body=%s", response.Code, response.Body.String())
			}
			return nil
		})
		done <- outcome{result: execution.Result, err: executeErr}
	}()
	value := <-done
	return value.result, value.err
}
