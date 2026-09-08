package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestToolApprovalDecisionAndResolution(t *testing.T) {
	s := &Server{approvals: make(map[string]*toolApproval)}
	approvalID := make(chan string, 1)
	resolved := make(chan bool, 1)
	type result struct {
		decision approvalDecision
		err      error
	}
	done := make(chan result, 1)

	go func() {
		decision, err := s.awaitToolApproval(context.Background(), "call-1", map[string]any{}, func(event string, data any) error {
			payload := data.(map[string]any)
			switch event {
			case "tool_approval":
				approvalID <- payload["approval_id"].(string)
			case "tool_approval_resolved":
				resolved <- payload["approved"].(bool)
			}
			return nil
		})
		done <- result{decision: decision, err: err}
	}()

	id := <-approvalID
	request := httptest.NewRequest(http.MethodPost, "/api/tool-approvals/"+id, strings.NewReader(`{"decision":"conversation"}`))
	response := httptest.NewRecorder()
	s.toolApproval(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	approvalResult := <-done
	if approvalResult.err != nil || approvalResult.decision != approvalConversation {
		t.Fatalf("decision=%q err=%v", approvalResult.decision, approvalResult.err)
	}
	if approved := <-resolved; !approved {
		t.Fatal("expected resolved approval")
	}
	var body map[string]any
	if err := json.Unmarshal(response.Body.Bytes(), &body); err != nil || body["accepted"] != true {
		t.Fatalf("unexpected response: %s", response.Body.String())
	}
}

func TestToolApprovalRejectsUnknownID(t *testing.T) {
	s := &Server{approvals: make(map[string]*toolApproval)}
	request := httptest.NewRequest(http.MethodPost, "/api/tool-approvals/missing", strings.NewReader(`{"approved":false}`))
	response := httptest.NewRecorder()
	s.toolApproval(response, request)
	if response.Code != http.StatusNotFound {
		t.Fatalf("status=%d", response.Code)
	}
}
