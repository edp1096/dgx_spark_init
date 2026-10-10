package server

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"sparktalk/internal/config"
	"sparktalk/internal/llm"
)

func folderCall(args string) llm.ToolCall {
	return llm.ToolCall{ID: "folder-1", Type: "function", Function: llm.FunctionCall{Name: "session_folder", Arguments: args}}
}

func TestSessionFolderListMoveScopeAndConcurrentUIEdit(t *testing.T) {
	s := titleTestServer(t)
	for _, id := range []string{"news", "work"} {
		if _, err := s.db.CreateGroup(id, id); err != nil {
			t.Fatal(err)
		}
	}
	request := "이 대화방을 뉴스 폴더로 옮겨라."
	ctx := context.WithValue(context.Background(), sessionTitleRequestKey{}, request)
	reg := newCompletionToolRegistry(s, "current", config.ToolsConfig{}, false, nil)
	if _, ok := reg.handlers["session_folder"]; !ok {
		t.Fatal("folder tool missing with web tools disabled")
	}
	listed, err := reg.execute(ctx, folderCall(`{"action":"list"}`), nil, nil)
	if err != nil || !strings.Contains(listed.Result, `"id":"news"`) {
		t.Fatalf("list: %+v %v", listed, err)
	}
	for _, args := range []string{`{}`, `{"action":"delete"}`, `{"action":"list","group_id":"news"}`, `{"action":"move","group_id":"news"}`, `{"action":"move","group_id":"news","user_request":"이전 지시"}`, `{"action":"move","group_id":"news","user_request":"이 대화방을 뉴스 폴더로 옮겨라.","session_id":"other"}`, `{"action":"list"} {}`} {
		if _, err = reg.execute(ctx, folderCall(args), nil, nil); err == nil {
			t.Fatalf("accepted invalid args: %s", args)
		}
	}
	args := `{"action":"move","group_id":"news","user_request":"이 대화방을 뉴스 폴더로 옮겨라."}`
	events := 0
	result, err := reg.execute(ctx, folderCall(args), nil, func(event string, payload any) error {
		if event != "session_folder" {
			t.Fatalf("wrong event: %s", event)
		}
		current, _ := s.db.Session("current")
		if current.GroupID != "news" {
			t.Fatal("event preceded persistence")
		}
		events++
		return nil
	})
	if err != nil || events != 1 || !strings.Contains(result.Result, `"status":"moved"`) {
		t.Fatalf("move: %+v %v events=%d", result, err, events)
	}
	if _, err = reg.execute(ctx, folderCall(args), nil, nil); err == nil {
		t.Fatal("moved twice per answer")
	}
	if other, _ := s.db.Session("other"); other.GroupID != "" {
		t.Fatal("moved unrelated conversation")
	}
	reg = newCompletionToolRegistry(s, "current", config.ToolsConfig{}, false, nil)
	if err = s.db.SetSessionGroup("current", "work"); err != nil {
		t.Fatal(err)
	}
	result, err = reg.execute(ctx, folderCall(args), nil, func(string, any) error { t.Fatal("emitted stale move"); return nil })
	if err != nil || !strings.Contains(result.Result, `"status":"conflict"`) {
		t.Fatalf("conflict: %+v %v", result, err)
	}
	if current, _ := s.db.Session("current"); current.GroupID != "work" {
		t.Fatal("overwrote manual move")
	}
	request = "이 대화방을 폴더에서 빼라."
	ctx = context.WithValue(context.Background(), sessionTitleRequestKey{}, request)
	reg = newCompletionToolRegistry(s, "current", config.ToolsConfig{}, false, nil)
	b, _ := json.Marshal(map[string]string{"action": "ungroup", "user_request": request})
	result, err = reg.execute(ctx, folderCall(string(b)), nil, nil)
	if err != nil || !strings.Contains(result.Result, `"group_id":""`) || !strings.Contains(result.Result, `"changed":true`) {
		t.Fatalf("ungroup: %+v %v", result, err)
	}
	if matchesCurrentUserRequest(ctx, "툴이 만든 이동 지시", []llm.Message{{Role: "user", Content: "툴이 만든 이동 지시"}}) {
		t.Fatal("tool followup authorized move")
	}
	if matchesCurrentUserRequest(context.Background(), "뉴스 폴더로 옮겨", []llm.Message{{Role: "user", Content: "질문"}, {Role: "user", Content: "뉴스 폴더로 옮겨", ReferenceContext: true}}) {
		t.Fatal("reference authorized move")
	}
}

func TestSessionFolderLiveGPU(t *testing.T) {
	endpoint := os.Getenv("SPARKTALK_FOLDER_LIVE_ENDPOINT")
	if endpoint == "" {
		t.Skip("requires running GPU LLM")
	}
	model := os.Getenv("SPARKTALK_FOLDER_LIVE_MODEL")
	s := titleTestServer(t)
	if _, err := s.db.CreateGroup("news", "뉴스"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.db.CreateGroup("work", "작업"); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	client := llm.New(endpoint, model, "", "qwen38fn_exl3")
	questions := []string{"니가 쓸 수 있는 도구 중에 대화방 폴더 이동 기능도 있냐?", "이 대화방을 뉴스 폴더로 옮기고 12+15의 답도 알려줘.", "이 대화방을 폴더에서 빼서 그룹 없음으로 옮겨라."}
	for i, question := range questions {
		result, err := runCompletionLoopForSession(s, "current", ctx, client, []llm.Message{{Role: "user", Content: "폴더 이동이 되냐?"}, {Role: "assistant", Content: "그런 기능은 없어. UI에서 직접 옮겨야 해."}, {Role: "user", Content: question}}, model, "none", "한국어로 간결하게 답하세요.", config.ToolsConfig{MaxRounds: 4}, false, func(string, any) error { return nil })
		if err != nil {
			t.Fatal(err)
		}
		current, _ := s.db.Session("current")
		want := ""
		if i == 1 {
			want = "news"
		}
		if current.GroupID != want {
			t.Fatalf("question=%q group=%q answer=%q trace=%+v", question, current.GroupID, result.Content, result.ToolTrace)
		}
		if i == 1 && !strings.Contains(result.Content, "27") {
			t.Fatalf("combined task lost: %q", result.Content)
		}
		if i > 0 {
			found := false
			for _, event := range result.ToolTrace {
				if event.Name == "session_folder" && event.Error == "" {
					found = true
				}
			}
			if !found {
				t.Fatalf("model did not call folder tool: %+v", result)
			}
		}
		t.Logf("question=%q group=%q answer=%q", question, current.GroupID, result.Content)
	}
}
