package server

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
)

func TestChatTimingPersistsForSendRetryAndEdit(t *testing.T) {
	s, _ := testImageServer(t)
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(25 * time.Millisecond)
		emitAttachmentTestCompletion(w, nil)
	}))
	defer backend.Close()
	s.cfg = config.Config{Tools: config.ToolsConfig{MaxRounds: 3}, Context: config.ContextConfig{WindowTokens: 8192, OutputReserve: 512}, Model: config.ModelConfig{DefaultModel: "model", Endpoint: backend.URL}}
	s.llm = llm.New(backend.URL, "model", "")
	defer s.tasks.Wait(context.Background())
	for _, action := range []string{"send", "retry", "edit"} {
		path := "/api/chat"
		body := map[string]any{"session_id": "session", "content": "question", "submission_delay_ms": 1500}
		var parent, answer db.Message
		if action != "send" {
			messages, err := s.db.Messages("session")
			if err != nil {
				t.Fatal(err)
			}
			parent, answer = messages[len(messages)-2], messages[len(messages)-1]
			if action == "retry" {
				path = "/api/messages/" + strconv.FormatInt(answer.ID, 10) + "/retry"
			} else {
				path = "/api/messages/" + strconv.FormatInt(parent.ID, 10) + "/edit"
			}
		}
		raw, _ := json.Marshal(body)
		w := httptest.NewRecorder()
		req := httptest.NewRequest("POST", path, bytes.NewReader(raw))
		if action == "send" {
			s.chat(w, req)
		} else {
			s.messageAction(w, req)
		}
		if w.Code != 200 || !strings.Contains(w.Body.String(), "event: done") {
			t.Fatalf("%s: %d %s", action, w.Code, w.Body.String())
		}
		messages, err := s.db.Messages("session")
		if err != nil {
			t.Fatal(err)
		}
		result := messages[len(messages)-1].Performance
		if result == nil || result.ElapsedSeconds == nil || *result.ElapsedSeconds < 1.525 || result.FinishStatus != db.MessageCompleted {
			t.Fatalf("%s: %+v", action, result)
		}
		if _, err = time.Parse(time.RFC3339Nano, result.FinishedAt); err != nil {
			t.Fatal(err)
		}
		encoded, _ := json.Marshal(result)
		if !strings.Contains(w.Body.String(), string(encoded)) {
			t.Fatal("stream and saved timing differ")
		}
	}
}

func TestSubmissionStartRejectsInvalidClientDelays(t *testing.T) {
	now := time.Now()
	for _, ms := range []float64{-1, math.NaN(), math.Inf(1), 60001} {
		if !submissionStart(now, ms).Equal(now) {
			t.Fatal("invalid delay accepted")
		}
	}
	if now.Sub(submissionStart(now, 1500)) != 1500*time.Millisecond {
		t.Fatal("browser preparation omitted")
	}
}
