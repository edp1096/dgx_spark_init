package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"sparktalk/internal/db"
	"sparktalk/internal/llm"
)

func TestMemoryProposalRequiresApprovalBeforeStore(t *testing.T) {
	store, err := db.Open(t.TempDir() + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	if _, err := store.CreateSession("session-1", "memory", "model", "low"); err != nil {
		t.Fatal(err)
	}
	server := &Server{db: store, approvals: make(map[string]*toolApproval)}
	call := llm.ToolCall{ID: "call-1"}
	call.Function.Name = "memory_propose"
	call.Function.Arguments = `{"kind":"user","title":"말투","content":"답변은 간결하게 작성한다."}`

	done := make(chan error, 1)
	go func() {
		_, executeErr := server.executeMemoryProposal(context.Background(), "session-1", call, func(event string, payload any) error {
			if event != "tool_approval" {
				return nil
			}
			approvalID := payload.(map[string]any)["approval_id"].(string)
			response := httptest.NewRecorder()
			server.toolApproval(response, httptest.NewRequest(http.MethodPost, "/api/tool-approvals/"+approvalID, strings.NewReader(`{"decision":"once"}`)))
			if response.Code != http.StatusOK {
				t.Errorf("approval status=%d body=%s", response.Code, response.Body.String())
			}
			return nil
		})
		done <- executeErr
	}()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	items, err := store.Memories()
	if err != nil || len(items) != 1 || items[0].Title != "말투" {
		t.Fatalf("memories=%+v err=%v", items, err)
	}
}
