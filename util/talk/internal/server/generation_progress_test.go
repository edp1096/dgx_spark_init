package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"sparktalk/internal/config"
	"sparktalk/internal/llm"
	"sparktalk/internal/orchestrator"
)

func TestVideoProgressPrecedesAttachmentAndPersistsWithTrace(t *testing.T) {
	s, _ := testImageServer(t)
	var requestID atomic.Value
	progressRead := make(chan struct{})
	var delivered sync.Once
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "POST" {
			id := r.Header.Get("X-SparkTalk-Request-ID")
			if len(id) != 32 {
				t.Error("missing request correlation ID")
			}
			requestID.Store(id)
			select {
			case <-progressRead:
			case <-time.After(5 * time.Second):
				t.Error("no progress poll")
			}
			w.Header().Set("Content-Type", "video/mp4")
			w.Write(importTestVideo)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		id, _ := requestID.Load().(string)
		if id != "" && strings.Contains(r.URL.Path, id) {
			if r.URL.Query().Get("after") == "0" {
				fmt.Fprint(w, `{"events":[{"event":"progress","stage":"sample","kind":"h3","step":1,"total":20,"eta_seconds":200,"eta_scope":"total"}],"next":1}`)
			} else {
				fmt.Fprint(w, `{"events":[],"next":1}`)
			}
			delivered.Do(func() { close(progressRead) })
		} else {
			fmt.Fprint(w, `{"events":[],"next":0}`)
		}
	}))
	defer backend.Close()
	catalog, _ := orchestrator.LoadCatalog()
	for i := range catalog.Components {
		if catalog.Components[i].ID == "qwim-mmh3" {
			catalog.Components[i].Endpoint = backend.URL
			catalog.Components[i].HealthURL = backend.URL + "/health"
		}
	}
	catalog, err := orchestrator.ValidateCatalog(catalog)
	if err != nil {
		t.Fatal(err)
	}
	s.cfg = config.Config{Runtime: config.RuntimeConfig{Mode: "external", Bundle: "qwen38fn_exl3", Catalog: &catalog}}
	var requests atomic.Int32
	model := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		if requests.Add(1) == 1 {
			fmt.Fprintln(w, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_video","type":"function","function":{"name":"video_generate","arguments":"{\"prompt\":\"red car\"}"}}]}}]}`)
		} else {
			fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":"saved"}}]}`)
		}
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer model.Close()
	history, _ := s.db.Messages("session")
	sawSample := false
	result, err := runCompletionLoopForSessionWithMedia(s, "session", context.Background(), llm.New(model.URL, "test", ""), []llm.Message{{Role: "user", Content: "make video"}}, "test", "none", "", config.ToolsConfig{Enabled: true, MaxRounds: 3}, true, func(event string, payload any) error {
		if event == "tool_output" {
			raw, _ := json.Marshal(payload)
			if strings.Contains(string(raw), "샘플링") {
				sawSample = true
			}
		}
		if event == "media_attached" && !sawSample {
			t.Error("video attachment arrived before progress")
		}
		return nil
	}, s.persistMediaAttachments(history[0].ID, 0, nil, nil))
	if err != nil {
		t.Fatal(err)
	}
	if len(result.ToolTrace) != 1 || !strings.Contains(result.ToolTrace[0].Output, "샘플링 1/20") || !strings.Contains(result.ToolTrace[0].Output, "전체 예상 3:20 남음") {
		t.Fatalf("progress not archived: %+v", result.ToolTrace)
	}
	if _, err = s.db.AddMessage("session", "assistant", result.Content, result.Reasoning, result.ToolTrace, nil); err != nil {
		t.Fatal(err)
	}
	messages, _ := s.db.Messages("session")
	if messages[len(messages)-1].ToolTrace[0].Output != result.ToolTrace[0].Output {
		t.Fatal("progress log lost on DB reload")
	}
}

func TestGenerationProgressStopCancelsInFlightPoll(t *testing.T) {
	entered := make(chan struct{})
	exited := make(chan struct{})
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { close(entered); <-r.Context().Done(); close(exited) }))
	defer backend.Close()
	ctx, cancel := context.WithCancel(context.Background())
	_, stop := startGenerationProgress(ctx, backend.URL, "call", "h3", func(string, any) error { return nil })
	select {
	case <-entered:
	case <-time.After(3 * time.Second):
		t.Fatal("poll did not start")
	}
	cancel()
	stop()
	stop()
	select {
	case <-exited:
	case <-time.After(time.Second):
		t.Fatal("poll leaked after cancellation")
	}
}
