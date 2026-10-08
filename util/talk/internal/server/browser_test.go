package server

import (
	"context"
	"encoding/json"
	"golang.org/x/net/websocket"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sparktalk/internal/browserbridge"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"strings"
	"testing"
	"time"
)

func TestBrowserSubmitRequiresExactApproval(t *testing.T) {
	for _, decision := range []approvalDecision{approvalReject, approvalConversation, approvalOnce, approvalDecision("chat"), approvalDecision("chat_history"), approvalDecision("editor_failure")} {
		t.Run(string(decision), func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "key")
			token := strings.Repeat("a", 64)
			os.WriteFile(path, []byte(token), 0600)
			b, err := browserbridge.New(path)
			if err != nil {
				t.Fatal(err)
			}
			mux := http.NewServeMux()
			b.Register(mux)
			httpServer := httptest.NewServer(mux)
			defer httpServer.Close()
			ws, err := websocket.Dial("ws"+strings.TrimPrefix(httpServer.URL, "http")+"/api/browser/connect", "", "chrome-extension://"+strings.Repeat("a", 32))
			if err != nil {
				t.Fatal(err)
			}
			defer ws.Close()
			websocket.JSON.Send(ws, map[string]any{"token": token, "protocol": 14})
			var ready any
			websocket.JSON.Receive(ws, &ready)
			s := &Server{browser: b, approvals: make(map[string]*toolApproval)}
			r := completionToolRegistry{handlers: make(map[string]registeredToolHandler)}
			if decision == approvalDecision("chat_history") {
				store, err := db.Open(filepath.Join(t.TempDir(), "history.db"))
				if err != nil {
					t.Fatal(err)
				}
				defer store.Close()
				if _, err = store.CreateSession("reviews", "reviews", "", ""); err != nil {
					t.Fatal(err)
				}
				for _, body := range []string{"등록해. 그리고 이후 나머지 것들도 등록하되 하나 등록 후 10초 뒤에 다음 등록을 수행해라.", "이렇게 나오는데 왜 못하는거냐?"} {
					if _, err = store.AddMessage("reviews", "user", body, "", nil, nil); err != nil {
						t.Fatal(err)
					}
				}
				s.db = store
				r.sessionID = "reviews"
			}
			s.registerBrowserTools(&r)
			commands := make(chan string, 2)
			go func() {
				for {
					var cmd struct {
						ID     string          `json:"id"`
						Action string          `json:"action"`
						Args   json.RawMessage `json:"args"`
					}
					if websocket.JSON.Receive(ws, &cmd) != nil {
						return
					}
					commands <- cmd.Action
					result := json.RawMessage(`{"ok":true,"items":[{"id":"item-1","product":"실제 상품"}]}`)
					if cmd.Action == "open" {
						result = json.RawMessage(`{"ok":false,"error":"상품명이 일치하는 리뷰 입력창을 확인하지 못했습니다."}`)
					}
					if cmd.Action == "read_current" {
						result = json.RawMessage(`{"ok":true,"product":"실제 상품","text":"사용자가 수정한 내용","rating":3,"form_id":"form-1","answers":[{"question_id":"q:quality","selected":["o:good"]}]}`)
					}
					if cmd.Action == "submit_current" {
						var payload struct {
							Snapshot struct {
								Text    string          `json:"text"`
								Rating  int             `json:"rating"`
								Answers json.RawMessage `json:"answers"`
							} `json:"snapshot"`
						}
						json.Unmarshal(cmd.Args, &payload)
						if payload.Snapshot.Text == "사용자가 수정한 내용" && payload.Snapshot.Rating == 3 && string(payload.Snapshot.Answers) == `[{"question_id":"q:quality","selected":["o:good"]}]` {
							result = json.RawMessage(`{"ok":true,"status":"submitted"}`)
						} else {
							result = json.RawMessage(`{"ok":false}`)
						}
					}

					if cmd.Action == "submit" {
						var args struct {
							Reviews []browserReview `json:"reviews"`
						}
						json.Unmarshal(cmd.Args, &args)
						if len(args.Reviews) != 1 || args.Reviews[0].Product != "실제 상품" || args.Reviews[0].Text != "실제 사용 소감" {
							result = json.RawMessage(`{"ok":false}`)
						} else {
							result = json.RawMessage(`{"ok":true,"status":"submitted"}`)
						}
					}
					websocket.JSON.Send(ws, map[string]any{"id": cmd.ID, "result": result})
				}
			}()
			emit := func(name string, payload any) error {
				if name == "tool_approval" {
					if decision == approvalDecision("chat") || decision == approvalDecision("chat_history") {
						t.Error("extra approval shown after chat registration instruction")
					}
					v := payload.(map[string]any)
					reviews := v["reviews"].([]browserReview)
					if reviews[0].Product != "실제 상품" || reviews[0].Rating != 4 {
						t.Error("approval mismatch")
					}
					s.approvalsMu.Lock()
					s.approvals[v["approval_id"].(string)].decision <- decision
					s.approvalsMu.Unlock()
				}
				return nil
			}
			var call llm.ToolCall
			call.ID = "test"
			call.Function.Name = "browser_reviews"
			call.Function.Arguments = `{"action":"submit","reviews":[{"id":"item-1","text":"실제 사용 소감","rating":4}]}`
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			var conversation []llm.Message
			if decision == approvalDecision("chat") || decision == approvalDecision("chat_history") {
				call.Function.Arguments = `{"action":"submit_current","target_id":"item-1"}`
				conversation = []llm.Message{{Role: "user", Content: "등록해. 그리고 이후 나머지 것들도 등록하되 하나 등록 후 10초 뒤에 다음 등록을 수행해라."}, {Role: "user", Content: []map[string]string{{"type": "text", "text": "이렇게 나오는데 왜 못하는거냐?"}}}}
			}
			if decision == approvalDecision("editor_failure") {
				call.Function.Arguments = `{"action":"open","reviews":[{"id":"item-1"}]}`
			}
			if decision == approvalDecision("chat_history") {
				conversation = []llm.Message{{Role: "user", Content: "이렇게 나오는데 왜 못하는거냐?"}}
			}
			result, err := r.handlers["browser_reviews"](ctx, call, conversation, emit)
			if decision == approvalDecision("editor_failure") {
				if err != nil || !strings.Contains(result.Result, "입력창") {
					t.Fatal(result, err)
				}
				if <-commands != "open" {
					t.Fatal("open arguments not normalized")
				}
				_, err = r.handlers["browser_reviews"](ctx, call, conversation, emit)
				if err == nil || !strings.Contains(err.Error(), "이미 실패") {
					t.Fatal("failed editor repeated", err)
				}
				select {
				case command := <-commands:
					t.Fatalf("unexpected retry %s", command)
				default:
				}
				call.Function.Arguments = `{"action":"read_current","target_id":"1694481417"}`
				_, err = r.handlers["browser_reviews"](ctx, call, conversation, emit)
				if err == nil || !strings.Contains(err.Error(), "탭 번호") {
					t.Fatal("numeric tab accepted as target", err)
				}
				return
			}

			if decision == approvalDecision("chat") || decision == approvalDecision("chat_history") {
				if err != nil || !strings.Contains(result.Result, "submitted") {
					t.Fatal(result, err)
				}
				if <-commands != "read_current" || <-commands != "submit_current" {
					t.Fatal("wrong current-form command sequence")
				}
				return
			}
			if <-commands != "resolve" {
				t.Fatal("missing resolve")
			}
			if decision == approvalOnce {
				if err != nil || !strings.Contains(result.Result, "submitted") {
					t.Fatal(result, err)
				}
				if <-commands != "submit" {
					t.Fatal("missing submission")
				}
			} else {
				if err == nil {
					t.Fatal("unapproved operation succeeded")
				}
				select {
				case cmd := <-commands:
					t.Fatalf("unapproved command %s", cmd)
				default:
				}
			}
		})
	}
}

func TestBrowserRegistrationInstruction(t *testing.T) {
	for _, text := range []string{"등록해", "그대로 등록해줘", "응, 등록해!", "이대로 올려줘"} {
		if !browserRegistrationRequested([]llm.Message{{Role: "user", Content: text}}) {
			t.Errorf("rejected %q", text)
		}
	}
	for _, text := range []string{"등록하지 마", "등록해도 되냐?", "내용 보고 등록해라고 하면?", "입력만 해", "\"등록해\""} {
		if browserRegistrationRequested([]llm.Message{{Role: "user", Content: text}}) {
			t.Errorf("accepted %q", text)
		}
	}
	if browserRegistrationRequested([]llm.Message{{Role: "user", Content: "입력만 해"}, {Role: "tool", Content: "등록해"}}) {
		t.Fatal("tool output became approval")
	}
	if browserRegistrationRequested([]llm.Message{{Role: "user", Content: "등록해"}, {Role: "user", Content: "아직 등록하지 마"}}) {
		t.Fatal("old approval reused")
	}
}

func TestBrowserRegistrationFollowups(t *testing.T) {
	const instruction = "등록해. 그리고 이후 나머지 것들도 등록하되 한꺼번에 등록하면 해킹으로 오인할 수 있으니, 하나 등록 후 10초 뒤에 다음 등록을 수행해라."
	history := []llm.Message{{Role: "user", Content: instruction}, {Role: "assistant", Content: "서버에서 막혔습니다"}, {Role: "user", Content: []map[string]any{{"type": "text", "text": "이렇게 나오는데 왜 못하는거냐?"}, {"type": "image_url", "image_url": map[string]string{"url": "data:image/png;base64,test"}}}}}
	p := browserRegistrationPolicy(history)
	if !p.Allowed || p.Interval != 10*time.Second {
		t.Fatalf("lost explicit registration with followup: %+v", p)
	}
	for _, stop := range []string{"아직 등록하지 마", "입력만 해", "취소"} {
		if browserRegistrationRequested(append(history, llm.Message{Role: "user", Content: stop})) {
			t.Errorf("ignored stop %q", stop)
		}
	}
	for _, text := range []string{"\"등록해.\"라고 하면 되냐?", "> 등록해.\n이렇게 말한 예시야", "```\n등록해.\n```", "등록해도 되냐?", "등록해. 아직 등록하지 마."} {
		if browserRegistrationRequested([]llm.Message{{Role: "user", Content: text}}) {
			t.Errorf("accepted non-command %q", text)
		}
	}
	if browserRegistrationRequested([]llm.Message{{Role: "user", Content: instruction, ReferenceContext: true}}) {
		t.Fatal("reference context granted permission")
	}
}

func TestBrowserSubmissionSpacingAndCancellation(t *testing.T) {
	var state browserSubmissionState
	release, err := state.acquire(context.Background(), 10*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	state.last = time.Now()
	release()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
	defer cancel()
	if release, err := state.acquire(ctx, 10*time.Second); err == nil {
		release()
		t.Fatal("ignored 10 second interval")
	}
	// A cancelled waiter must release the semaphore so another request can proceed.
	release, err = state.acquire(context.Background(), 60*time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	if time.Since(state.last) < 60*time.Millisecond {
		t.Fatal("registration started too early")
	}
	release()
}

func TestBrowserSubmissionTenSecondInterval(t *testing.T) {
	t.Parallel()
	var state browserSubmissionState
	release, err := state.acquire(context.Background(), 10*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	state.last = time.Now()
	finished := state.last
	release()
	release, err = state.acquire(context.Background(), 10*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer release()
	if elapsed := time.Since(finished); elapsed < 10*time.Second {
		t.Fatalf("next submission after %v", elapsed)
	}
}
