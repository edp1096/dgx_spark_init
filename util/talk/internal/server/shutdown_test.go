package server

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sparktalk/internal/llm"
	"testing"
	"time"
)

func TestShutdownDrainsQueuedKnowledgeAndGenerations(t *testing.T) {
	s := &Server{knowledgeJobSem: make(chan struct{}, 1), knowledgeOCRSem: make(chan struct{}, 1)}
	s.knowledgeJobSem <- struct{}{}
	s.knowledgeOCRSem <- struct{}{}
	// No database: queued work must cancel without touching it.
	s.scheduleKnowledgeJob("queued")
	s.scheduleKnowledgeOCR("queued")
	gen, finish, err := s.trackGeneration(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	go func() { <-gen.Done(); finish() }()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err = s.Shutdown(ctx); err != nil {
		t.Fatal(err)
	}
	if len(s.knowledgeJobs) != 0 || len(s.knowledgeOCRJobs) != 0 || len(s.generations) != 0 {
		t.Fatal("shutdown returned with work alive")
	}
	s.scheduleKnowledgeJob("late")
	s.scheduleKnowledgeOCR("late")
	if len(s.knowledgeJobs) != 0 || len(s.knowledgeOCRJobs) != 0 {
		t.Fatal("accepted work after shutdown")
	}
	if _, _, err = s.trackGeneration(context.Background()); err == nil {
		t.Fatal("accepted generation after shutdown")
	}
	if err = s.Shutdown(ctx); err != nil {
		t.Fatal("second shutdown:", err)
	}
}

func TestShutdownCancelsAndDrainsTitleRequest(t *testing.T) {
	started := make(chan struct{})
	canceled := make(chan struct{})
	release := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		close(started)
		select {
		case <-r.Context().Done():
			close(canceled)
		case <-release:
		}
	}))
	defer upstream.Close()
	defer close(release)
	s := &Server{}
	s.scheduleTitle(llm.New(upstream.URL, "model", ""), "session", "model", "input")
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("title request did not start")
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := s.Shutdown(ctx); err != nil {
		t.Fatal(err)
	}
	select {
	case <-canceled:
	case <-ctx.Done():
		t.Fatal("upstream title not canceled")
	}
	s.scheduleTitle(nil, "late", "model", "input")
}
