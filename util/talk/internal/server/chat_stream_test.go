package server

import (
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

func TestBrowserDisconnectDoesNotCancelAndReplayCompletes(t *testing.T) {
	for _, tc := range []struct {
		operation string
		stop      bool
	}{{"chat", false}, {"chat", true}, {"retry", false}, {"edit", false}} {
		t.Run(fmt.Sprint(tc.operation, tc.stop), func(t *testing.T) {
			explicitCancel := tc.stop
			s, _ := testImageServer(t)
			started, release, canceled := make(chan struct{}), make(chan struct{}), make(chan struct{})
			var calls atomic.Int32
			backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
					http.NotFound(w, r)
					return
				}
				var request struct {
					Stream bool `json:"stream"`
				}
				_ = json.NewDecoder(r.Body).Decode(&request)
				if !request.Stream {
					io.WriteString(w, `{"choices":[{"message":{"content":"title"}}]}`)
					return
				}
				calls.Add(1)
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"first part\"}}]}\n\n")
				w.(http.Flusher).Flush()
				close(started)
				select {
				case <-release:
				case <-r.Context().Done():
					close(canceled)
					return
				}
				fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\" last part\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
			}))
			s.cfg = config.Config{Runtime: config.RuntimeConfig{Mode: "external"}, Model: config.ModelConfig{Endpoint: backend.URL, DefaultModel: "test"}, Context: config.ContextConfig{WindowTokens: 32768}, Tools: config.ToolsConfig{MaxRounds: 3}}
			s.llm = llm.New(backend.URL, "test", "")
			mux := http.NewServeMux()
			mux.HandleFunc("/api/chat", s.chat)
			mux.HandleFunc("/api/chat/active", s.activeChat)
			mux.HandleFunc("/api/chat/active/", s.activeChat)
			mux.HandleFunc("/api/messages/", s.messageAction)
			app := httptest.NewServer(mux)
			t.Cleanup(func() {
				s.tasks.Stop()
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer cancel()
				s.tasks.Wait(ctx)
				app.Close()
				backend.Close()
			})
			path := "/api/chat"
			if tc.operation != "chat" {
				previous, err := s.db.AddMessage("session", "assistant", "old answer", "", nil, nil)
				if err != nil {
					t.Fatal(err)
				}
				id := previous.ID
				if tc.operation == "edit" {
					messages, _ := s.db.Messages("session")
					id = messages[0].ID
				}
				path = fmt.Sprintf("/api/messages/%d/%s", id, tc.operation)
			}
			response, err := http.Post(app.URL+path, "application/json", strings.NewReader(`{"session_id":"session","content":"keep working after disconnect"}`))
			if err != nil {
				t.Fatal(err)
			}
			waitSignal(t, started)
			response.Body.Close()
			select {
			case <-canceled:
				t.Fatal("browser disconnect canceled generation")
			case <-time.After(100 * time.Millisecond):
			}
			status, err := http.Get(app.URL + "/api/chat/active?session_id=session")
			if err != nil {
				t.Fatal(err)
			}
			var state struct {
				Active bool   `json:"active"`
				TurnID string `json:"turn_id"`
			}
			json.NewDecoder(status.Body).Decode(&state)
			status.Body.Close()
			if !state.Active || state.TurnID == "" {
				t.Fatal("detached task not discoverable", state)
			}
			replay, err := http.Get(app.URL + "/api/chat/active/events?session_id=session&turn_id=" + state.TurnID)
			if err != nil {
				t.Fatal(err)
			}
			defer replay.Body.Close()
			if explicitCancel {
				body := fmt.Sprintf(`{"session_id":"session","turn_id":%q}`, state.TurnID)
				cancelReply, err := http.Post(app.URL+"/api/chat/active/cancel", "application/json", strings.NewReader(body))
				if err != nil {
					t.Fatal(err)
				}
				cancelReply.Body.Close()
				if cancelReply.StatusCode != 202 {
					t.Fatal(cancelReply.StatusCode)
				}
				waitSignal(t, canceled)
			} else {
				close(release)
			}
			data, err := io.ReadAll(replay.Body)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(data), "first part") {
				t.Fatal("replay lost partial answer", string(data))
			}
			messages, err := s.db.Messages("session")
			if err != nil {
				t.Fatal(err)
			}
			last := messages[len(messages)-1]
			if explicitCancel {
				if !strings.Contains(string(data), "event: error") || last.Status != db.MessageCancelled {
					t.Fatal("explicit stop not persisted", string(data), last)
				}
			} else {
				if !strings.Contains(string(data), "event: done") || last.Content != "first part last part" || last.Status != db.MessageCompleted {
					t.Fatal("detached completion lost", string(data), last)
				}
			}
			if calls.Load() != 1 {
				t.Fatal("reconnection restarted generation", calls.Load())
			}
		})
	}
}
