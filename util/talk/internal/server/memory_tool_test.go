package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
)

func TestMemoryManageSearchCreateUpdateAndDelete(t *testing.T) {
	store, err := db.Open(t.TempDir() + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	if _, err := store.CreateSession("session-1", "memory", "model", "low"); err != nil {
		t.Fatal(err)
	}
	server := &Server{db: store, approvals: make(map[string]*toolApproval)}

	createResult, err := executeApprovedMemoryTool(t, server, "session-1", "create-1", `{"action":"create","kind":"memory","priority":"preferred","title":"장비","content":"주력 장비는 DGX Spark다."}`, approvalOnce)
	if err != nil || !strings.Contains(createResult, `"action":"create"`) {
		t.Fatalf("create result=%s err=%v", createResult, err)
	}
	items, err := store.Memories()
	if err != nil || len(items) != 1 {
		t.Fatalf("created memories=%+v err=%v", items, err)
	}
	id := items[0].ID

	searchCall := llm.ToolCall{ID: "search-1", Function: llm.FunctionCall{Name: "memory_manage", Arguments: `{"action":"search","query":"DGX Spark 장비"}`}}
	searchResult, err := server.executeMemoryManage(context.Background(), "session-1", searchCall, func(string, any) error { return nil })
	if err != nil || !strings.Contains(searchResult, `"id":`+strconv.FormatInt(id, 10)) {
		t.Fatalf("search result=%s err=%v", searchResult, err)
	}

	updateArgs := `{"action":"update","memory_id":` + strconv.FormatInt(id, 10) + `,"kind":"user","priority":"reference","title":"주력 장비","enabled":false}`
	updateResult, err := executeApprovedMemoryTool(t, server, "session-1", "update-1", updateArgs, approvalOnce)
	if err != nil || !strings.Contains(updateResult, `"enabled":false`) {
		t.Fatalf("update result=%s err=%v", updateResult, err)
	}
	updated, err := store.Memory(id)
	if err != nil || updated.Kind != "user" || updated.Priority != "reference" || updated.Title != "주력 장비" || updated.Enabled {
		t.Fatalf("updated memory=%+v err=%v", updated, err)
	}

	deleteArgs := `{"action":"delete","memory_id":` + strconv.FormatInt(id, 10) + `}`
	if _, err := executeApprovedMemoryTool(t, server, "session-1", "delete-reject", deleteArgs, approvalReject); err == nil {
		t.Fatal("rejected delete unexpectedly succeeded")
	}
	if _, err := store.Memory(id); err != nil {
		t.Fatalf("rejected delete removed memory: %v", err)
	}
	deleteResult, err := executeApprovedMemoryTool(t, server, "session-1", "delete-1", deleteArgs, approvalOnce)
	if err != nil || !strings.Contains(deleteResult, `"action":"delete"`) {
		t.Fatalf("delete result=%s err=%v", deleteResult, err)
	}
	if _, err := store.Memory(id); !db.IsMemoryNotFound(err) {
		t.Fatalf("deleted memory still exists: %v", err)
	}
}

func TestMemoryManageIsAvailableWhenProactiveProposalsAreDisabled(t *testing.T) {
	server := &Server{cfg: config.Config{Memory: config.MemoryConfig{Enabled: true, AllowProposals: false}}}
	registry := newCompletionToolRegistry(server, "session", config.ToolsConfig{}, false, nil)
	if _, ok := registry.handlers["memory_manage"]; !ok {
		t.Fatal("explicit memory management was disabled with proactive proposals")
	}
	if len(registry.prompts) != 1 || !strings.Contains(registry.prompts[0], "explicitly asks") {
		t.Fatalf("unexpected memory prompt: %+v", registry.prompts)
	}
	_, err := registry.execute(context.Background(), llm.ToolCall{Function: llm.FunctionCall{Name: "memory_manage", Arguments: `{"action":"delete","memory_id":7}`}}, nil, func(string, any) error { return nil })
	if err == nil || !strings.Contains(err.Error(), "not returned by a memory search") {
		t.Fatalf("unsearched mutation was allowed: %v", err)
	}
}

func TestMemoryManageRequiresSearchResultIDForMutation(t *testing.T) {
	server := &Server{}
	call := llm.ToolCall{ID: "missing-id", Function: llm.FunctionCall{Name: "memory_manage", Arguments: `{"action":"delete","query":"장비"}`}}
	_, err := server.executeMemoryManage(context.Background(), "session", call, func(string, any) error { return nil })
	if err == nil || !strings.Contains(err.Error(), "memory_id") {
		t.Fatalf("expected exact ID error, got %v", err)
	}
}

func executeApprovedMemoryTool(t *testing.T, server *Server, sessionID, callID, arguments string, decision approvalDecision) (string, error) {
	t.Helper()
	call := llm.ToolCall{ID: callID, Function: llm.FunctionCall{Name: "memory_manage", Arguments: arguments}}
	type outcome struct {
		result string
		err    error
	}
	done := make(chan outcome, 1)
	go func() {
		result, executeErr := server.executeMemoryManage(context.Background(), sessionID, call, func(event string, payload any) error {
			if event != "tool_approval" {
				return nil
			}
			approvalID := payload.(map[string]any)["approval_id"].(string)
			body, _ := json.Marshal(map[string]string{"decision": string(decision)})
			response := httptest.NewRecorder()
			server.toolApproval(response, httptest.NewRequest(http.MethodPost, "/api/tool-approvals/"+approvalID, strings.NewReader(string(body))))
			if response.Code != http.StatusOK {
				t.Errorf("approval status=%d body=%s", response.Code, response.Body.String())
			}
			return nil
		})
		done <- outcome{result: result, err: executeErr}
	}()
	result := <-done
	return result.result, result.err
}
