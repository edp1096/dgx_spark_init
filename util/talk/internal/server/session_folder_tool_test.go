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

func TestSessionFolderCreateMoveRenameDeleteKeepsChats(t *testing.T) {
	s := titleTestServer(t)
	request := "번역 폴더를 만들고 이 방을 넣은 다음 이름을 독서로 바꿔라."
	ctx := context.WithValue(context.Background(), sessionTitleRequestKey{}, request)
	reg := newCompletionToolRegistry(s, "current", config.ToolsConfig{}, false, nil)
	events := 0
	call := func(args map[string]string) map[string]any {
		t.Helper()
		args["user_request"] = request
		raw, _ := json.Marshal(args)
		result, err := reg.execute(ctx, folderCall(string(raw)), nil, func(event string, payload any) error {
			if event != "session_folder" {
				t.Fatal(event)
			}
			events++
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		var out map[string]any
		if err = json.Unmarshal([]byte(result.Result), &out); err != nil {
			t.Fatal(err)
		}
		return out
	}
	created := call(map[string]string{"action": "create", "name": "번역"})
	if created["status"] != "created" {
		t.Fatal(created)
	}
	id := created["folder"].(map[string]any)["id"].(string)
	if reused := call(map[string]string{"action": "create", "name": "번역"}); reused["status"] != "exists" || reused["folder"].(map[string]any)["id"] != id {
		t.Fatal(reused)
	}
	if moved := call(map[string]string{"action": "move", "group_id": id}); moved["status"] != "moved" {
		t.Fatal(moved)
	}
	if renamed := call(map[string]string{"action": "rename", "group_id": id, "name": "독서"}); renamed["status"] != "renamed" || renamed["group_name"] != "독서" {
		t.Fatal(renamed)
	}
	if events != 3 {
		t.Fatal("wrong change events", events)
	}
	if err := s.db.SetSessionGroup("other", id); err != nil {
		t.Fatal(err)
	}
	for _, session := range []string{"current", "other"} {
		if _, err := s.db.AddMessage(session, "user", "보존할 대화", "", nil, nil); err != nil {
			t.Fatal(err)
		}
	}
	request = "독서 폴더만 삭제해라."
	ctx = context.WithValue(context.Background(), sessionTitleRequestKey{}, request)
	reg = newCompletionToolRegistry(s, "current", config.ToolsConfig{}, false, nil)
	deleted := call(map[string]string{"action": "delete", "group_id": id})
	if deleted["status"] != "deleted" || deleted["ungrouped_chats"] != float64(2) {
		t.Fatal(deleted)
	}
	groups, err := s.db.Groups()
	if err != nil || len(groups) != 0 {
		t.Fatal(groups, err)
	}
	for _, id := range []string{"current", "other"} {
		session, err := s.db.Session(id)
		if err != nil || session.GroupID != "" {
			t.Fatal(session, err)
		}
		messages, err := s.db.Messages(id)
		if err != nil || len(messages) != 1 || messages[0].Content != "보존할 대화" {
			t.Fatal("chat content lost", messages, err)
		}
	}
}

func TestSessionFolderCRUDRejectsInvalidRequestsAndPreservesConcurrentRename(t *testing.T) {
	s := titleTestServer(t)
	s.db.CreateGroup("g", "번역")
	request := "번역 폴더 이름을 독서로 바꿔라."
	ctx := context.WithValue(context.Background(), sessionTitleRequestKey{}, request)
	reg := newCompletionToolRegistry(s, "current", config.ToolsConfig{}, false, nil)
	for _, args := range []map[string]string{
		{"action": "create", "name": ""}, {"action": "create", "name": "a\nb"}, {"action": "create", "name": strings.Repeat("가", 61)},
		{"action": "create", "name": "독서", "group_id": "g"}, {"action": "delete", "group_id": "g", "name": "독서"},
		{"action": "rename", "name": "독서", "group_id": "unknown"}, {"action": "list", "name": "독서"},
	} {
		args["user_request"] = request
		raw, _ := json.Marshal(args)
		if _, err := reg.execute(ctx, folderCall(string(raw)), nil, nil); err == nil {
			t.Fatal("accepted invalid mutation", args)
		}
	}
	for _, action := range []string{"create", "rename", "delete"} {
		args := map[string]string{"action": action, "user_request": "이전 사용자의 지시"}
		if action == "create" {
			args["name"] = "새폴더"
		} else {
			args["group_id"] = "g"
		}
		if action == "rename" {
			args["name"] = "독서"
		}
		raw, _ := json.Marshal(args)
		if _, err := reg.execute(ctx, folderCall(string(raw)), []llm.Message{{Role: "user", Content: "이전 사용자의 지시"}}, nil); err == nil {
			t.Fatal("old message authorized mutation", action)
		}
	}
	if err := s.db.RenameGroup("g", "수동 변경"); err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal(map[string]string{"action": "rename", "group_id": "g", "name": "독서", "user_request": request})
	result, err := reg.execute(ctx, folderCall(string(raw)), nil, func(string, any) error { t.Fatal("emitted conflicted rename"); return nil })
	if err != nil || !strings.Contains(result.Result, `"status":"conflict"`) {
		t.Fatal(result, err)
	}
	if _, err = reg.execute(ctx, folderCall(`{"action":"list"}`), nil, nil); err != nil {
		t.Fatal(err)
	}
	if _, err = reg.execute(ctx, folderCall(string(raw)), nil, nil); err == nil {
		t.Fatal("retried over concurrent UI change")
	}
	groups, _ := s.db.Groups()
	if groups[0].Name != "수동 변경" {
		t.Fatal(groups)
	}
}

func TestSessionFolderDeleteConflictPreservesAssignments(t *testing.T) {
	s := titleTestServer(t)
	s.db.CreateGroup("g", "번역")
	s.db.SetSessionGroup("current", "g")
	reg := newCompletionToolRegistry(s, "current", config.ToolsConfig{}, false, nil)
	s.db.RenameGroup("g", "수동 이름")
	ctx := context.WithValue(context.Background(), sessionTitleRequestKey{}, "번역 폴더 삭제해라.")
	result, err := reg.execute(ctx, folderCall(`{"action":"delete","group_id":"g","user_request":"번역 폴더 삭제해라."}`), nil, nil)
	if err != nil || !strings.Contains(result.Result, `"status":"conflict"`) {
		t.Fatal(result, err)
	}
	session, err := s.db.Session("current")
	if err != nil || session.GroupID != "g" {
		t.Fatal("conflicted deletion ungrouped chat", session, err)
	}
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
	questions := []string{"니가 쓸 수 있는 도구 중에 대화방 폴더 이동 기능도 있냐?", "이 대화방을 뉴스 폴더로 옮기고 12+15의 답도 알려줘.", "이 대화방을 폴더에서 빼서 그룹 없음으로 옮겨라.", "번역 폴더를 만들고 이 대화방을 그 폴더로 옮겨라.", "번역 폴더 이름을 독서로 바꿔라.", "독서 폴더를 삭제해라. 안의 대화는 보존해라."}
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
		if i == 3 || i == 4 {
			name := "번역"
			if i == 4 {
				name = "독서"
			}
			folders, e := s.db.Groups()
			if e != nil {
				t.Fatal(e)
			}
			for _, folder := range folders {
				if folder.Name == name {
					want = folder.ID
				}
			}
			if want == "" {
				t.Fatalf("requested folder missing after %q: %+v answer=%q trace=%+v", question, folders, result.Content, result.ToolTrace)
			}
		}
		if i == 5 {
			folders, e := s.db.Groups()
			if e != nil {
				t.Fatal(e)
			}
			for _, folder := range folders {
				if folder.Name == "독서" {
					t.Fatalf("folder not deleted: %q trace=%+v", result.Content, result.ToolTrace)
				}
			}
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
