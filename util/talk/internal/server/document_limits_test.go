package server

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"sparktalk/internal/llm"
	"sparktalk/internal/workflows"
)

func TestDocumentIdleLimitResetsOnlyForActualProgress(t *testing.T) {
	for _, active := range []bool{false, true} {
		t.Run(fmt.Sprint(active), func(t *testing.T) {
			backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				w.(http.Flusher).Flush()
				if !active {
					<-r.Context().Done()
					return
				}
				for i := 0; i < 6; i++ {
					time.Sleep(20 * time.Millisecond)
					fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking\"}}]}\n\n")
					w.(http.Flusher).Flush()
				}
				fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
			}))
			defer backend.Close()
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			ctx = context.WithValue(ctx, documentLimitsKey{}, documentStreamLimits{idle: 80 * time.Millisecond})
			run := workflows.Run{Document: &workflows.DocumentTask{Model: "model", Effort: "xhigh"}}
			result, err := streamDocumentUnit(ctx, llm.New(backend.URL, "model", ""), run, nil, 4096, func(string, string) error { return nil })
			if active {
				if err != nil || result.Content != "answer" {
					t.Fatal("progressing stream was killed", err)
				}
			} else if !errors.Is(err, errDocumentIdle) {
				t.Fatal("idle stream not distinguished", err)
			}
		})
	}
}

func TestDocumentReasoningBackstopIsNotModelUnavailability(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"reasoning continues without an answer\"}}]}\n\n")
	}))
	defer backend.Close()
	ctx := context.WithValue(context.Background(), documentLimitsKey{}, documentStreamLimits{idle: time.Second, reasoningChars: 8})
	run := workflows.Run{Document: &workflows.DocumentTask{Model: "model", Effort: "xhigh"}}
	result, err := streamDocumentUnit(ctx, llm.New(backend.URL, "model", ""), run, nil, 4096, func(string, string) error { return nil })
	if !errors.Is(err, errDocumentReasoning) || result.Reasoning == "" || documentModelUnavailable(err) {
		t.Fatal(result, err)
	}
	if documentModelUnavailable(fmt.Errorf("wrapped: %w", context.DeadlineExceeded)) {
		t.Fatal("deadline is not a missing model")
	}
	if !documentModelUnavailable(&llm.HTTPError{StatusCode: 503}) || documentModelUnavailable(&llm.HTTPError{StatusCode: 400}) || documentModelUnavailable(&llm.HTTPError{StatusCode: 504}) {
		t.Fatal("HTTP availability classification")
	}
}

func TestDocumentResumeRetriesDisconnectedUnitPreservingEightCompletedUnits(t *testing.T) {
	var calls atomic.Int32
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"interrupted reasoning\"}}]}\n\n")
			return
		}
		mockDocumentResponse(w, r, true)
	})
	source := fixtureSource(t, s, strings.Repeat("A paragraph that must be preserved entirely with its checkpoint. ", 500))
	s.docInferMu.Lock()
	run, err := s.documentAction(context.Background(), "s", documentStart{Action: "start", Kind: "translate", Sources: []string{source}})
	if err != nil {
		s.docInferMu.Unlock()
		t.Fatal(err)
	}
	run, err = s.documentAction(context.Background(), "s", documentStart{Action: "pause", RunID: run.ID})
	s.docInferMu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	units, err := s.db.DocumentUnits("s", run.ID)
	if err != nil || len(units) <= 8 {
		t.Fatal("fixture too small", err, len(units))
	}
	for i := 0; i < 8; i++ {
		units[i].Status = "completed"
		units[i].Output = map[string]string{}
		for _, in := range units[i].Inputs {
			units[i].Output[in.ID] = "고정 완료 " + in.Text
		}
		units[i].Attempts = []workflows.DocumentAttempt{{FinishReason: "stop", OutputTokens: 12}}
	}
	before := append([]workflows.DocumentUnit(nil), units[:8]...)
	run.Document.Completed = 8
	run.Status = "waiting_model"
	if err = s.db.SaveDocumentProgress(&run, units...); err != nil {
		t.Fatal(err)
	}
	if _, err = s.documentAction(context.Background(), "s", documentStart{Action: "resume", RunID: run.ID}); err != nil {
		t.Fatal(err)
	}
	run = waitDocument(t, s, run.ID)
	if run.Status != "completed" || run.Document.Completed != len(units) {
		t.Fatalf("disconnection stopped the task: %+v", run.Document)
	}
	after, err := s.db.DocumentUnits("s", run.ID)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(before, after[:8]) {
		t.Fatal("completed checkpoints were modified")
	}
	if len(after[8].Attempts) != 2 || after[8].Attempts[0].Error == "" || after[8].Attempts[1].Error != "" || !after[8].Attempts[0].OutputTokensEstimated || after[8].Attempts[0].OutputTokens == 0 {
		t.Fatal("interrupted unit was not retried or accounted for", after[8].Attempts)
	}
	if int(calls.Load()) != len(units)-8+1 {
		t.Fatal("unexpected duplicate calls", calls.Load())
	}
}

func TestRepeatedStreamInterruptionDoesNotBecomeModelWaiting(t *testing.T) {
	var calls atomic.Int32
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"partial thinking\"}}]}\n\n")
	})
	source := fixtureSource(t, s, "A complete source paragraph.")
	run, err := s.documentAction(context.Background(), "s", documentStart{Action: "start", Kind: "translate", Sources: []string{source}})
	if err != nil {
		t.Fatal(err)
	}
	run = waitDocument(t, s, run.ID)
	if run.Status != "needs_review" || calls.Load() != 3 || !strings.Contains(run.Document.Error, "3회 실패") {
		t.Fatalf("wrong recovery state: %+v calls=%d", run, calls.Load())
	}
}
