package db

import (
	"database/sql"
	"path/filepath"
	"testing"
	"time"
)

func TestRetryReplacesAssistantAndTruncatesLaterBranch(t *testing.T) {
	store, err := Open(filepath.Join(t.TempDir(), "retry.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.CreateSession("s1", "test", "model", "low"); err != nil {
		t.Fatal(err)
	}
	image := Attachment{ID: "image-1", Name: "diagram.png", MIME: "image/png", Size: 123, URL: "/api/images/image-1"}
	question, err := store.AddMessage("s1", "user", "first", "", nil, []Attachment{image})
	if err != nil {
		t.Fatal(err)
	}
	answer, err := store.AddMessage("s1", "assistant", "old", "old reasoning", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, _ = store.AddMessage("s1", "user", "later", "", nil, nil)
	_, _ = store.AddMessage("s1", "assistant", "later answer", "", nil, nil)

	target, history, err := store.RetryContext(answer.ID, 0)
	if err != nil {
		t.Fatal(err)
	}
	if target.Content != "old" || len(history) != 1 || history[0].Content != "first" {
		t.Fatalf("unexpected retry context: target=%+v history=%+v", target, history)
	}
	if len(history[0].Attachments) != 1 || history[0].Attachments[0].ID != image.ID {
		t.Fatalf("image attachment was not restored in retry history: %+v", history[0].Attachments)
	}
	referenced, err := store.ReferencedAttachmentIDs()
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := referenced[image.ID]; !ok {
		t.Fatalf("attachment reference was not found: %+v", referenced)
	}
	trace := []ToolEvent{{Name: "web_search", Arguments: `{"query":"test"}`, Result: `{"results":[]}`}}
	if err := store.ReplaceAssistant(answer.ID, "new", "new reasoning", trace, 0); err != nil {
		t.Fatal(err)
	}
	messages, err := store.Messages("s1")
	if err != nil {
		t.Fatal(err)
	}
	if len(messages) != 2 || messages[1].Content != "new" || messages[1].Reasoning != "new reasoning" || len(messages[1].ToolTrace) != 1 {
		t.Fatalf("retry did not replace/truncate atomically: %+v", messages)
	}
	if len(messages[1].Variants) != 2 || messages[1].Variants[0].Content != "old" || messages[1].Variants[1].Content != "new" {
		t.Fatalf("retry variants were not preserved: %+v", messages[1].Variants)
	}
	if err := store.ReplaceAssistant(answer.ID, "newest", "third reasoning", nil, 0); err != nil {
		t.Fatal(err)
	}
	messages, err = store.Messages("s1")
	if err != nil {
		t.Fatal(err)
	}
	if len(messages[1].Variants) != 3 || messages[1].Variants[2].Content != "newest" {
		t.Fatalf("subsequent retry was not appended: %+v", messages[1].Variants)
	}
	userTarget, assistantTarget, editHistory, err := store.EditContext(question.ID)
	if err != nil {
		t.Fatal(err)
	}
	if userTarget.Content != "first" || assistantTarget == nil || assistantTarget.ID != answer.ID || len(editHistory) != 0 {
		t.Fatalf("unexpected edit context: user=%+v assistant=%+v history=%+v", userTarget, assistantTarget, editHistory)
	}
	if err := store.AppendEditedBranch(question.ID, "first revised", nil, "edited answer", "edited reasoning", nil); err != nil {
		t.Fatal(err)
	}
	messages, err = store.Messages("s1")
	if err != nil {
		t.Fatal(err)
	}
	if len(messages) != 2 || len(messages[0].Variants) != 2 || messages[0].Content != "first revised" {
		t.Fatalf("question variants were not preserved: %+v", messages)
	}
	answerVariants := messages[1].Variants
	if len(answerVariants) != 4 || answerVariants[3].Content != "edited answer" || answerVariants[3].ParentVariant != 1 {
		t.Fatalf("edited answer was not linked to the revised question: %+v", answerVariants)
	}
	_, oldHistory, err := store.RetryContext(answer.ID, 0)
	if err != nil || oldHistory[len(oldHistory)-1].Content != "first" {
		t.Fatalf("old question branch was not restored for retry: history=%+v err=%v", oldHistory, err)
	}
	if err := store.ReplaceAssistant(answer.ID, "old branch retry", "", nil, 0); err != nil {
		t.Fatal(err)
	}
	messages, err = store.Messages("s1")
	if err != nil {
		t.Fatal(err)
	}
	latest := messages[1].Variants[len(messages[1].Variants)-1]
	if latest.Content != "old branch retry" || latest.ParentVariant != 0 {
		t.Fatalf("retry was linked to the wrong question branch: %+v", latest)
	}
}

func TestOpenMigratesLegacyMessagesToVisibleVariant(t *testing.T) {
	path := filepath.Join(t.TempDir(), "legacy.db")
	legacy, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	_, err = legacy.Exec(`
		CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT NOT NULL, model TEXT NOT NULL DEFAULT '', reasoning_effort TEXT NOT NULL DEFAULT '', created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL);
		CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, reasoning_content TEXT NOT NULL DEFAULT '', tool_trace TEXT NOT NULL DEFAULT '[]', created_at DATETIME NOT NULL);
		INSERT INTO sessions(id,title,model,reasoning_effort,created_at,updated_at) VALUES('s1','legacy','','',?,?);
		INSERT INTO messages(session_id,role,content,reasoning_content,tool_trace,created_at) VALUES('s1','assistant','legacy answer','legacy reasoning','[]',?);
	`, now, now, now)
	if err != nil {
		t.Fatal(err)
	}
	if err := legacy.Close(); err != nil {
		t.Fatal(err)
	}

	store, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	messages, err := store.Messages("s1")
	if err != nil {
		t.Fatal(err)
	}
	if len(messages) != 1 || len(messages[0].Variants) != 1 || messages[0].Variants[0].Content != "legacy answer" {
		t.Fatalf("legacy response was not exposed as its first variant: %+v", messages)
	}
}

func TestManualSessionTitleIsNotOverwrittenByGeneratedTitle(t *testing.T) {
	store, err := Open(filepath.Join(t.TempDir(), "title.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.CreateSession("s1", "새 대화", "model", "medium"); err != nil {
		t.Fatal(err)
	}
	if err := store.UpdateSessionTitle("s1", "자동 제목"); err != nil {
		t.Fatal(err)
	}
	if err := store.RenameSession("s1", "내 제목"); err != nil {
		t.Fatal(err)
	}
	if err := store.UpdateSessionTitle("s1", "뒤늦은 자동 제목"); err != nil {
		t.Fatal(err)
	}
	sessions, err := store.Sessions()
	if err != nil {
		t.Fatal(err)
	}
	if len(sessions) != 1 || sessions[0].Title != "내 제목" {
		t.Fatalf("manual title was overwritten: %+v", sessions)
	}
}

func TestGroupLifecycleAndSessionMembership(t *testing.T) {
	store, err := Open(filepath.Join(t.TempDir(), "groups.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.CreateSession("s1", "첫 대화", "model", "medium"); err != nil {
		t.Fatal(err)
	}
	first, err := store.CreateGroup("g1", "업무")
	if err != nil {
		t.Fatal(err)
	}
	second, err := store.CreateGroup("g2", "개인")
	if err != nil {
		t.Fatal(err)
	}
	if first.Position != 0 || second.Position != 1 {
		t.Fatalf("unexpected group positions: %+v %+v", first, second)
	}
	if err := store.SetSessionGroup("s1", "g1"); err != nil {
		t.Fatal(err)
	}
	if err := store.RenameGroup("g1", "프로젝트"); err != nil {
		t.Fatal(err)
	}
	if err := store.MoveGroup("g2", "up"); err != nil {
		t.Fatal(err)
	}
	groups, err := store.Groups()
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 2 || groups[0].ID != "g2" || groups[1].Name != "프로젝트" {
		t.Fatalf("unexpected groups after rename/move: %+v", groups)
	}
	sessions, err := store.Sessions()
	if err != nil {
		t.Fatal(err)
	}
	if len(sessions) != 1 || sessions[0].GroupID != "g1" {
		t.Fatalf("session was not assigned to group: %+v", sessions)
	}
	if err := store.DeleteGroup("g1"); err != nil {
		t.Fatal(err)
	}
	sessions, err = store.Sessions()
	if err != nil {
		t.Fatal(err)
	}
	if sessions[0].GroupID != "" {
		t.Fatalf("deleting a group should preserve and ungroup its sessions: %+v", sessions)
	}
}
