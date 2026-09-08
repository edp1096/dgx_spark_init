package server

import (
	"strings"
	"testing"

	"sparktalk/internal/db"
)

func TestModelHistoryReplacesFailedMediaWithShortText(t *testing.T) {
	items := []db.Message{
		{ID: 1, Role: "user", Status: db.MessageCompleted, Content: "정상 질문"},
		{ID: 2, Role: "assistant", Status: db.MessageCompleted, Content: "정상 답변"},
		{ID: 3, Role: "user", Status: db.MessageFailed, Content: "이거 설명", Error: "decode data:video/mp4;base64,AAAA failed", Attachments: []db.Attachment{{Name: "bad.mp4", MIME: "video/mp4", Size: 5 << 20}}},
		{ID: 4, Role: "user", Status: db.MessagePending, Content: "새 이미지", Attachments: []db.Attachment{{Name: "new.png", MIME: "image/png"}}},
	}
	history := modelHistory(items, 4)
	if len(history) != 5 {
		t.Fatalf("model history length = %d: %+v", len(history), history)
	}
	failedRequest := history[2]
	if len(failedRequest.Attachments) != 0 || !strings.Contains(failedRequest.Content, "bad.mp4") || strings.Contains(failedRequest.Content, "AAAA") {
		t.Fatalf("failed media was not safely summarized: %+v", failedRequest)
	}
	if len(history[4].Attachments) != 1 || history[4].Attachments[0].Name != "new.png" {
		t.Fatalf("current pending attachment was removed: %+v", history[4])
	}
}

func TestModelHistoryDropsAbandonedPendingRequest(t *testing.T) {
	history := modelHistory([]db.Message{{ID: 1, Role: "user", Status: db.MessagePending, Content: "old"}}, 0)
	if len(history) != 0 {
		t.Fatalf("abandoned pending request leaked into history: %+v", history)
	}
}

func TestInterruptedCodeRemainsAvailableForContinuation(t *testing.T) {
	code := "```go\nfunc main() {\n\tprintln(\"hello\")\n"
	items := []db.Message{
		{ID: 1, Role: "user", Status: db.MessageFailed, Content: "write code", Error: "finish_reason=length"},
		{ID: 2, Role: "assistant", Status: db.MessageFailed, Content: code, Reasoning: "private reasoning", ToolTrace: []db.ToolEvent{{Name: "incomplete"}}},
		{ID: 3, Role: "user", Status: db.MessagePending, Content: "continue"},
	}
	got := modelHistory(items, 3)
	if len(got) != 3 || !strings.Contains(got[1].Content, code) || !strings.Contains(got[1].Content, "불완전한") || got[1].Reasoning != "" || len(got[1].ToolTrace) != 0 {
		t.Fatalf("continuation history lost or replayed incomplete tools: %+v", got)
	}
}
