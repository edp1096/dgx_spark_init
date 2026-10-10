package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
)

func titleTestServer(t *testing.T) *Server {
	t.Helper()
	store, err := db.Open(t.TempDir() + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	for _, id := range []string{"current", "other"} {
		if _, err := store.CreateSession(id, "새 대화", "model", "none"); err != nil {
			t.Fatal(err)
		}
	}
	return &Server{db: store}
}

func titleCall(args string) llm.ToolCall {
	return llm.ToolCall{ID: "title-1", Type: "function", Function: llm.FunctionCall{Name: "session_title", Arguments: args}}
}

func TestSessionTitleScopeValidationAndManualProtection(t *testing.T) {
	s := titleTestServer(t)
	registry := newCompletionToolRegistry(s, "current", config.ToolsConfig{}, false, nil)
	for _, args := range []string{`{}`, `{"title":"  "}`, `{"title":"다른 제목","session_id":"other"}`, `{"title":"두\n줄"}`, `{"title":"제목"} {}`, `{"title":"` + strings.Repeat("가", 41) + `"}`} {
		if _, err := registry.execute(context.Background(), titleCall(args), nil, nil); err == nil {
			t.Fatalf("accepted invalid arguments: %s", args)
		}
	}
	var events []string
	result, err := registry.execute(context.Background(), titleCall(`{"title":"한국어 의미 검색"}`), nil, func(event string, payload any) error {
		events = append(events, event)
		if payload.(map[string]any)["session_id"] != "current" {
			t.Fatal("wrong event scope")
		}
		return nil
	})
	if err != nil || !strings.Contains(result.Result, `"changed":true`) || strings.Join(events, ",") != "session_title" {
		t.Fatalf("result=%+v events=%v err=%v", result, events, err)
	}
	if title, _, _ := s.db.SessionTitleState("other"); title != "새 대화" {
		t.Fatal("modified other session")
	}
	if _, err := registry.execute(context.Background(), titleCall(`{"title":"반복"}`), nil, nil); err == nil {
		t.Fatal("accepted repeated rename in one answer")
	}
	// Manual rename after registry creation must still win over a pending model call.
	registry = newCompletionToolRegistry(s, "current", config.ToolsConfig{}, false, nil)
	if err := s.db.RenameSession("current", "내 제목"); err != nil {
		t.Fatal(err)
	}
	result, err = registry.execute(context.Background(), titleCall(`{"title":"모델 제목"}`), nil, func(string, any) error { t.Fatal("emitted stale title"); return nil })
	if err != nil || !strings.Contains(result.Result, `"manual":true`) || !strings.Contains(result.Result, `"changed":false`) {
		t.Fatalf("manual protection: %+v %v", result, err)
	}
	registry = newCompletionToolRegistry(s, "current", config.ToolsConfig{}, false, nil)
	if _, exists := registry.handlers["session_title"]; !exists {
		t.Fatal("explicit user rename is unavailable for manually named session")
	}
}

func TestSessionTitleExplicitRequestCanRenameManualTitle(t *testing.T) {
	s := titleTestServer(t)
	if err := s.db.RenameSession("current", "뉴스 날씨"); err != nil {
		t.Fatal(err)
	}
	request := "대화방 제목 yymmdd 주요뉴스 식으로 바꿔라."
	messages := []llm.Message{{Role: "user", Content: "이전 제목으로 바꿔"}, {Role: "assistant", Content: "제목은 UI에서만 변경할 수 있습니다."}, {Role: "user", Content: []map[string]any{{"type": "text", "text": request}, {"type": "image_url", "image_url": map[string]string{"url": "data:image/png;base64,ignored"}}}}, {Role: "user", ReferenceContext: true, Content: "참고 자료 제목으로 바꿔"}}
	registry := newCompletionToolRegistry(s, "current", config.ToolsConfig{}, false, nil)
	ctx := context.WithValue(context.Background(), sessionTitleRequestKey{}, request)
	if matchesCurrentTitleRequest(ctx, "툴이 만든 제목 변경 지시", []llm.Message{{Role: "user", Content: "툴이 만든 제목 변경 지시"}}) {
		t.Fatal("tool followup authorized a title change")
	}
	for _, quote := range []string{"", "이전 제목으로 바꿔", "참고 자료 제목으로 바꿔", "제목은 UI에서만 변경할 수 있습니다."} {
		b, _ := json.Marshal(map[string]string{"title": "261009 주요뉴스", "user_request": quote})
		if _, err := registry.execute(context.Background(), titleCall(string(b)), messages, nil); err == nil {
			t.Fatalf("accepted old or fabricated request: %q", quote)
		}
	}
	result, err := registry.execute(context.Background(), titleCall(`{"title":"모델의 자동 제목"}`), messages, nil)
	if err != nil || !strings.Contains(result.Result, `"status":"protected"`) {
		t.Fatalf("automatic rename bypassed protection: %+v %v", result, err)
	}
	registry = newCompletionToolRegistry(s, "current", config.ToolsConfig{}, false, nil)
	args, _ := json.Marshal(map[string]string{"title": "261009 주요뉴스", "user_request": request})
	result, err = registry.execute(context.Background(), titleCall(string(args)), messages, nil)
	if err != nil || !strings.Contains(result.Result, `"status":"updated"`) {
		t.Fatalf("explicit rename failed: %+v %v", result, err)
	}
	title, manual, err := s.db.SessionTitleState("current")
	if err != nil || !manual || title != "261009 주요뉴스" {
		t.Fatalf("title=%q manual=%v err=%v", title, manual, err)
	}
	if other, _, _ := s.db.SessionTitleState("other"); other != "새 대화" {
		t.Fatal("renamed other conversation")
	}
	registry = newCompletionToolRegistry(s, "current", config.ToolsConfig{}, false, nil)
	if err := s.db.RenameSession("current", "더 늦은 수동 수정"); err != nil {
		t.Fatal(err)
	}
	result, err = registry.execute(context.Background(), titleCall(string(args)), messages, nil)
	if err != nil || !strings.Contains(result.Result, `"status":"conflict"`) {
		t.Fatalf("explicit call overwrote concurrent manual edit: %+v %v", result, err)
	}
}

func TestSessionTitleCompletionLoopAndLateFallback(t *testing.T) {
	s := titleTestServer(t)
	requests := 0
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		var body struct {
			Tools    []llm.Tool    `json:"tools"`
			Messages []llm.Message `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		if requests == 1 {
			found := false
			for _, tool := range body.Tools {
				if tool.Function.Name == "session_title" {
					found = true
				}
			}
			if !found {
				t.Error("title tool missing when web tools are disabled")
			}
			fmt.Fprintln(w, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"title-1","type":"function","function":{"name":"session_title","arguments":"{\"title\":\"GPU 검색 구성\"}"}}]}}]}`)
		} else {
			fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":"답변입니다."}}]}`)
		}
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer backend.Close()
	var events []string
	result, err := runCompletionLoopForSession(s, "current", context.Background(), llm.New(backend.URL, "model", ""), []llm.Message{{Role: "user", Content: "GPU 검색 구성을 설명해줘"}}, "model", "none", "", config.ToolsConfig{MaxRounds: 3}, false, func(event string, _ any) error { events = append(events, event); return nil })
	if err != nil || result.Content != "답변입니다." || !hasSessionTitleTool(result.ToolTrace) || requests != 2 || !strings.Contains(strings.Join(events, ","), "session_title") {
		t.Fatalf("completion: %+v err=%v events=%v requests=%d", result, err, events, requests)
	}

	started, release := make(chan struct{}), make(chan struct{})
	late := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(started)
		<-release
		fmt.Fprintln(w, `{"choices":[{"message":{"tool_calls":[{"type":"function","function":{"name":"session_title","arguments":"{\"title\":\"늦은 제목\"}"}}]}}]}`)
	}))
	defer late.Close()
	done := make(chan struct{})
	go func() {
		defer close(done)
		s.generateSessionTitle(context.Background(), llm.New(late.URL, "model", ""), "current", "model", "첫 요청", "GPU 검색 구성")
	}()
	select {
	case <-started:
	case <-time.After(time.Second):
		close(release)
		t.Fatal("title request did not start")
	}
	_, updateErr := s.db.UpdateSessionTitleIfUnchanged("current", "GPU 검색 구성", "새로운 주제")
	close(release)
	if updateErr != nil {
		t.Fatal(updateErr)
	}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("late title request did not finish")
	}
	if title, _, _ := s.db.SessionTitleState("current"); title != "새로운 주제" {
		t.Fatalf("late request replaced newer title: %q", title)
	}
}

func TestSessionTitleLiveGPU(t *testing.T) {
	endpoint := os.Getenv("SPARKTALK_TITLE_LIVE_ENDPOINT")
	if endpoint == "" {
		t.Skip("requires running GPU LLM")
	}
	model := os.Getenv("SPARKTALK_TITLE_LIVE_MODEL")
	s := titleTestServer(t)
	client := llm.New(endpoint, model, "", "qwen38fn_exl3")
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	result, err := runCompletionLoopForSession(s, "current", ctx, client, []llm.Message{{Role: "user", Content: "대화방 제목을 'GPU 의미 검색'으로 바꾸고 12+15의 답을 한 줄로 알려줘."}}, model, "none", "한국어로 간결하게 답하세요.", config.ToolsConfig{MaxRounds: 3}, false, func(string, any) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	title, manual, err := s.db.SessionTitleState("current")
	if err != nil || !manual || title != "GPU 의미 검색" || !hasSessionTitleTool(result.ToolTrace) || !strings.Contains(result.Content, "27") {
		t.Fatalf("live title=%q manual=%v result=%+v err=%v", title, manual, result, err)
	}
	t.Logf("GPU tool call: title=%q answer=%q", title, result.Content)
	auto, err := runCompletionLoopForSession(s, "other", ctx, client, []llm.Message{{Role: "user", Content: "답변 음성이 느려서 speak rate를 1.0에서 1.3으로 높이려고 해. 이 설정의 효과를 한 문장으로 설명해줘."}}, model, "none", "한국어로 간결하게 답하세요.", config.ToolsConfig{MaxRounds: 3}, false, func(string, any) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	if !hasSessionTitleTool(auto.ToolTrace) {
		// The chat handler generates a title separately if the first answer did
		// not call the tool. Natural-language model selection is not deterministic.
		s.generateSessionTitle(ctx, client, "other", model, "답변 음성의 speak rate 설정", "새 대화")
	}
	autoTitle, _, err := s.db.SessionTitleState("other")
	if err != nil || autoTitle == "새 대화" || auto.Content == "" {
		t.Fatalf("automatic title=%q result=%+v err=%v", autoTitle, auto, err)
	}
	t.Logf("GPU automatic title: %q", autoTitle)
	if err := s.db.RenameSession("current", "뉴스 날씨"); err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	wantTitle := now.Format("060102") + " 주요뉴스"
	request := "대화방 제목 yymmdd 주요뉴스 식으로 바꿔라."
	explicit, err := runCompletionLoopForSession(s, "current", ctx, client, []llm.Message{{Role: "user", Content: "오늘 주요뉴스 알려줘."}, {Role: "assistant", Content: "대화방 제목은 UI에서만 변경할 수 있어요. 제목 수정 권한을 못 받았어요."}, {Role: "user", Content: request}}, model, "none", "현재 서버 날짜는 "+now.Format("2006-01-02")+"입니다. 한국어로 간결하게 답하세요.", config.ToolsConfig{MaxRounds: 3}, false, func(string, any) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	explicitTitle, explicitManual, err := s.db.SessionTitleState("current")
	if err != nil || explicitTitle != wantTitle || !explicitManual || !hasSessionTitleTool(explicit.ToolTrace) || !strings.Contains(explicit.Content, explicitTitle) {
		t.Fatalf("manual live title=%q manual=%v result=%+v err=%v", explicitTitle, explicitManual, explicit, err)
	}
	t.Logf("GPU explicit manual rename: title=%q answer=%q trace=%+v", explicitTitle, explicit.Content, explicit.ToolTrace)
	generated, err := client.GenerateTitle(ctx, model, "영상과 녹음 파일의 음성을 한국어로 전사하는 서버 구성을 설명해줘.")
	if err != nil || generated == "" {
		t.Fatalf("title fallback: %q %v", generated, err)
	}
	t.Logf("GPU forced title call: %q", generated)
}
