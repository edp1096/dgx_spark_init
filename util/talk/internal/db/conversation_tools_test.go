package db

import (
	"sparktalk/internal/workflows"
	"strings"
	"testing"
)

func conversationToolFixture(t *testing.T) *DB {
	t.Helper()
	d, err := Open(t.TempDir() + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { d.Close() })
	for _, id := range []string{"current", "a", "b", "c"} {
		if _, err = d.CreateSession(id, id, "model", "low"); err != nil {
			t.Fatal(err)
		}
	}
	if _, err = d.CreateGroup("folder", "정리"); err != nil {
		t.Fatal(err)
	}
	return d
}

func TestConversationSearchDistinctPagesAndDocumentKind(t *testing.T) {
	d := conversationToolFixture(t)
	for i := 0; i < 30; i++ {
		if _, err := d.AddMessage("a", "user", "번역 translation 내용", "", nil, nil); err != nil {
			t.Fatal(err)
		}
	}
	for _, id := range []string{"current", "b", "c"} {
		d.AddMessage(id, "user", "번역 translation 요청", "", nil, nil)
	}
	for _, q := range []string{"번역", "translation"} {
		first, next, err := d.FindConversations(q, "", "current", 0, 2)
		if err != nil || len(first) != 2 || next == 0 {
			t.Fatal(first, next, err)
		}
		last, end, err := d.FindConversations(q, "", "current", next, 2)
		if err != nil || len(last) != 1 || end != 0 {
			t.Fatal(last, end, err)
		}
		seen := map[string]bool{}
		for _, s := range append(first, last...) {
			if s.ID == "current" || seen[s.ID] {
				t.Fatal("wrong distinct scope", s)
			}
			seen[s.ID] = true
		}
	}
	d.CreateSession("d", "제목 없는 책", "model", "low")
	d.AddMessage("d", "user", "이 책 전체 작업을 계속해라.", "", nil, nil)
	if err := d.SaveWorkflowRun(&workflows.Run{ID: "document", SessionID: "d", Status: "completed", Document: &workflows.DocumentTask{Kind: "translate"}}); err != nil {
		t.Fatal(err)
	}
	items, _, err := d.FindConversations("", "translate", "current", 0, 20)
	if err != nil || len(items) != 1 || items[0].ID != "d" {
		t.Fatal(items, err)
	}
	union, _, err := d.FindConversations("번역", "translate", "current", 0, 20)
	if err != nil || len(union) != 4 {
		t.Fatal("keyword and task matches were not combined", union, err)
	}
}

func TestConversationReadBoundedUnicodeAndMessageScope(t *testing.T) {
	d := conversationToolFixture(t)
	m, err := d.AddMessage("a", "user", strings.Repeat("가", 3201), "private reasoning", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	d.AddMessage("a", "assistant", "답변", "", nil, nil)
	page, next, err := d.ConversationMessages("a", 0, 0, 0, 1)
	if err != nil || len(page) != 1 || len([]rune(page[0].Text)) != 1600 || page[0].NextOffset != 1600 || next != m.ID {
		t.Fatal(page, next, err)
	}
	part, _, err := d.ConversationMessages("a", 0, m.ID, 3200, 1)
	if err != nil || len(part) != 1 || part[0].Text != "가" || part[0].NextOffset != 0 {
		t.Fatal(part, err)
	}
	wrong, _, err := d.ConversationMessages("b", 0, m.ID, 0, 1)
	if err != nil || len(wrong) != 0 {
		t.Fatal("cross-conversation message ID read", wrong, err)
	}
}

func TestConversationBatchMoveProtectsConcurrentEditsAndReportsMissing(t *testing.T) {
	d := conversationToolFixture(t)
	var targets []ConversationSummary
	for _, id := range []string{"a", "b", "c"} {
		s, err := d.ConversationSummary(id)
		if err != nil {
			t.Fatal(err)
		}
		targets = append(targets, s)
	}
	d.AddMessage("b", "user", "new content after review", "", nil, nil)
	d.DeleteSession("c")
	result, err := d.MoveConversationsFromTool(targets, "folder", "정리")
	if err != nil || result[0].Status != "moved" || result[1].Status != "conflict" || result[2].Status != "not_found" {
		t.Fatal(result, err)
	}
	for _, id := range []string{"current", "b"} {
		s, _ := d.Session(id)
		if s.GroupID != "" {
			t.Fatal("unselected/conflicted session moved", s)
		}
	}
	repeat, err := d.MoveConversationsFromTool(targets[:1], "folder", "정리")
	if err != nil || repeat[0].Status != "unchanged" {
		t.Fatal(repeat, err)
	}
}

func TestConversationBatchMoveRollsBackOnDatabaseFailure(t *testing.T) {
	d := conversationToolFixture(t)
	a, _ := d.ConversationSummary("a")
	b, _ := d.ConversationSummary("b")
	if _, err := d.conn.Exec(`CREATE TRIGGER deny_second_move BEFORE UPDATE OF group_id ON sessions WHEN OLD.id='b' BEGIN SELECT RAISE(ABORT,'write failure'); END`); err != nil {
		t.Fatal(err)
	}
	if _, err := d.MoveConversationsFromTool([]ConversationSummary{a, b}, "folder", "정리"); err == nil {
		t.Fatal("write failure ignored")
	}
	s, _ := d.Session("a")
	if s.GroupID != "" {
		t.Fatal("partial batch committed", s)
	}
}
