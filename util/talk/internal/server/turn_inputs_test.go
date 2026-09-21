package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
)

func waitSignal(t *testing.T, ch <-chan struct{}) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(5 * time.Second):
		t.Fatal("timed out")
	}
}

func TestSteeringHTTPInterruptsInferenceAndPersists(t *testing.T) {
	s, _ := testImageServer(t)
	first, second, finish := make(chan struct{}), make(chan struct{}), make(chan struct{})
	var calls atomic.Int32
	var aborted atomic.Bool
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/server_info" {
			fmt.Fprint(w, `{"max_total_num_tokens":32768,"mamba_radix_cache_strategy":"auto"}`)
			return
		}
		if r.URL.Path == "/abort_request" {
			aborted.Store(true)
			w.WriteHeader(200)
			return
		}
		if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
			http.NotFound(w, r)
			return
		}
		data, _ := io.ReadAll(r.Body)
		n := calls.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		if n == 1 {
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking\"}}]}\n\n")
			w.(http.Flusher).Flush()
			close(first)
			<-r.Context().Done()
			return
		}
		if n != 2 || !bytes.Contains(data, []byte("original-task")) || !bytes.Contains(data, []byte("new-direction")) {
			t.Errorf("bad resumed request: %s", data)
		}
		close(second)
		select {
		case <-finish:
		case <-r.Context().Done():
			return
		}
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"updated answer\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	defer backend.Close()
	s.cfg = config.Config{Runtime: config.RuntimeConfig{Mode: "external"}, Model: config.ModelConfig{Endpoint: backend.URL, DefaultModel: "test"}, Context: config.ContextConfig{WindowTokens: 32768}, Tools: config.ToolsConfig{MaxRounds: 3}}
	s.llm = llm.New(backend.URL, "test", "")
	recorder := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		defer close(done)
		s.chat(recorder, httptest.NewRequest("POST", "/api/chat", strings.NewReader(`{"session_id":"session","content":"original-task"}`)))
	}()
	waitSignal(t, first)
	s.turnMu.Lock()
	turn := s.turns["session"]
	s.turnMu.Unlock()
	if turn == nil {
		t.Fatal("turn missing")
	}
	body, _ := json.Marshal(map[string]string{"session_id": "session", "turn_id": turn.token, "input_id": "stable-input-id", "content": "new-direction"})
	send := func() int {
		w := httptest.NewRecorder()
		s.steerChat(w, httptest.NewRequest("POST", "/api/chat/steer", bytes.NewReader(body)))
		return w.Code
	}
	if code := send(); code != 202 {
		t.Fatal(code)
	}
	waitSignal(t, second)
	if code := send(); code != 202 {
		t.Fatalf("idempotent retry: %d", code)
	}
	if !aborted.Load() {
		t.Fatal("native abort was not used by the actual inference client")
	}
	if turn.ctx.Err() != nil {
		t.Fatal("steering cancelled parent/tool context")
	}
	close(finish)
	waitSignal(t, done)
	if !strings.Contains(recorder.Body.String(), "event: steering_applied") || !strings.Contains(recorder.Body.String(), "event: done") || strings.Contains(recorder.Body.String(), "event: error") {
		t.Fatal(recorder.Body.String())
	}
	messages, err := s.db.Messages("session")
	if err != nil {
		t.Fatal(err)
	}
	if len(messages) < 2 || len(messages[len(messages)-2].TurnInputs) != 1 || messages[len(messages)-1].Content != "updated answer" {
		t.Fatalf("bad history: %+v", messages)
	}
	if code := send(); code != 409 {
		t.Fatalf("closed turn accepted input: %d", code)
	}
}

func TestSteeringKeepsToolContextAndResults(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	turn := &activeTurn{ctx: ctx, accepting: true}
	ctx = context.WithValue(ctx, activeTurnKey{}, turn)
	// Simulate a tool already running with the parent context.
	toolDone := make(chan struct{})
	releaseTool := make(chan struct{})
	go func() {
		select {
		case <-releaseTool:
		case <-ctx.Done():
			t.Error("tool was killed by new input")
		}
		close(toolDone)
	}()
	turn.mu.Lock()
	turn.pending = append(turn.pending, db.TurnInput{ID: "direction", Content: "new direction"})
	turn.mu.Unlock()
	close(releaseTool)
	waitSignal(t, toolDone)
	conversation := []llm.Message{{Role: "user", Content: "original"}, {Role: "assistant", ToolCalls: []llm.ToolCall{{ID: "completed"}}}, {Role: "tool", ToolCallID: "completed", Content: "download completed"}}
	result, err := steeredRequest(ctx, &conversation, false, func(string, any) error { return nil }, func(child context.Context, m []llm.Message) (llm.StreamResult, error) {
		if len(m) != 4 || m[2].Content != "download completed" || m[3].Content != "new direction" {
			t.Fatalf("tool result lost: %+v", m)
		}
		return llm.StreamResult{Content: "done"}, nil
	})
	if err != nil || result.Content != "done" || turn.accepting {
		t.Fatal(result, err, turn.accepting)
	}
}

func TestSteeringAtFinalBoundaryAndExplicitStop(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	turn := &activeTurn{ctx: ctx, accepting: true}
	ctx = context.WithValue(ctx, activeTurnKey{}, turn)
	conversation := []llm.Message{{Role: "user", Content: "original"}}
	calls := 0
	_, err := steeredRequest(ctx, &conversation, false, func(string, any) error { return nil }, func(child context.Context, m []llm.Message) (llm.StreamResult, error) {
		calls++
		if calls == 1 {
			turn.mu.Lock()
			turn.pending = append(turn.pending, db.TurnInput{Content: "late input"})
			turn.mu.Unlock()
			return llm.StreamResult{Content: "old answer"}, nil
		}
		if m[len(m)-1].Content != "late input" {
			t.Fatal(m)
		}
		cancel()
		return llm.StreamResult{}, child.Err()
	})
	if err != context.Canceled || calls != 2 {
		t.Fatal(err, calls)
	}
}

func TestSteeringDoesNotCancelRunningToolOrStartStaleNextTool(t *testing.T) {
	root, cancel := context.WithCancel(context.Background())
	defer cancel()
	turn := &activeTurn{ctx: root, accepting: true}
	ctx := context.WithValue(root, activeTurnKey{}, turn)
	started, release, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
	calls := 0
	handler := func(toolCtx context.Context, _ llm.ToolCall, _ []llm.Message, _ eventEmitter) (registeredToolResult, error) {
		calls++
		close(started)
		select {
		case <-toolCtx.Done():
			return registeredToolResult{}, toolCtx.Err()
		case <-release:
			return registeredToolResult{Result: "download finished"}, nil
		}
	}
	go func() {
		defer close(done)
		r, err := executeTurnTool(ctx, llm.ToolCall{}, nil, nil, handler)
		if err != nil || r.Result != "download finished" {
			t.Errorf("tool interrupted: %+v %v", r, err)
		}
	}()
	waitSignal(t, started)
	turn.mu.Lock()
	turn.pending = append(turn.pending, db.TurnInput{ID: "new-input", Content: "new instruction"})
	if turn.cancelInference != nil {
		turn.cancelInference()
	}
	turn.mu.Unlock()
	close(release)
	waitSignal(t, done)
	_, err := executeTurnTool(ctx, llm.ToolCall{}, nil, nil, handler)
	if err == nil || calls != 1 {
		t.Fatalf("stale tool was executed: calls=%d err=%v", calls, err)
	}
}

func TestTurnInputIncludedInFutureContextAndSummary(t *testing.T) {
	s := &Server{}
	item := db.Message{Role: "user", Content: "original", TurnInputs: []db.TurnInput{{ID: "a", Content: "must keep this direction"}}}
	messages, err := s.llmMessages(context.Background(), []db.Message{item}, config.Config{})
	if err != nil || !strings.Contains(fmt.Sprint(messages[0].Content), "must keep this direction") {
		t.Fatal(messages, err)
	}
	if !strings.Contains(s.contextTranscript([]db.Message{item}, config.Config{}), "must keep this direction") {
		t.Fatal("summary dropped follow-up")
	}
	if estimateMessage(item, 0) <= estimateMessage(db.Message{Role: "user", Content: "original"}, 0) {
		t.Fatal("follow-up was not budgeted")
	}
}

func TestExplicitStopCancelsTool(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	started, done := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(done)
		_, err := executeTurnTool(ctx, llm.ToolCall{}, nil, nil, func(ctx context.Context, _ llm.ToolCall, _ []llm.Message, _ eventEmitter) (registeredToolResult, error) {
			close(started)
			<-ctx.Done()
			return registeredToolResult{}, ctx.Err()
		})
		if err != context.Canceled {
			t.Errorf("explicit stop did not cancel tool: %v", err)
		}
	}()
	waitSignal(t, started)
	cancel()
	waitSignal(t, done)
}
