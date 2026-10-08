package db

import (
	"database/sql"
	"fmt"
	"path/filepath"
	"testing"
	"time"
)

func TestLegacyMemoriesGainExplicitPriority(t *testing.T) {
	path := filepath.Join(t.TempDir(), "legacy-memory.db")
	legacy, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	_, err = legacy.Exec(`
		CREATE TABLE memories (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			kind TEXT NOT NULL,
			title TEXT NOT NULL DEFAULT '',
			content TEXT NOT NULL,
			enabled INTEGER NOT NULL DEFAULT 1,
			source_session_id TEXT NOT NULL DEFAULT '',
			source_message_id INTEGER NOT NULL DEFAULT 0,
			created_at DATETIME NOT NULL,
			updated_at DATETIME NOT NULL
		);
		INSERT INTO memories(kind,title,content,enabled,source_session_id,source_message_id,created_at,updated_at)
		VALUES ('memory','직접 기억','직접 입력',1,'',0,?,?),
		       ('memory','대화 저장','대화 원문',1,'session',17,?,?);
	`, time.Now(), time.Now(), time.Now(), time.Now())
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
	items, err := store.Memories()
	if err != nil || len(items) != 2 {
		t.Fatalf("memories=%+v err=%v", items, err)
	}
	priorities := map[string]string{}
	for _, item := range items {
		priorities[item.Title] = item.Priority
	}
	if priorities["직접 기억"] != "preferred" || priorities["대화 저장"] != "reference" {
		t.Fatalf("migrated priorities=%+v", priorities)
	}
}

func TestMemoryCRUDAndCrossSessionSearch(t *testing.T) {
	store, err := Open(filepath.Join(t.TempDir(), "memory.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.CreateSession("old", "DGX Spark 운영", "model", "low"); err != nil {
		t.Fatal(err)
	}
	if _, err := store.CreateSession("current", "현재 대화", "model", "low"); err != nil {
		t.Fatal(err)
	}
	if _, err := store.CreateSession("empty", "비어 있는 검색방", "model", "low"); err != nil {
		t.Fatal(err)
	}
	if _, err := store.AddMessage("old", "user", "DGX Spark 메모리 최적화 방법", "", nil, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := store.AddMessage("old", "assistant", "통합 메모리는 호스트 가용량을 함께 확인한다.", "", nil, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := store.CreateSession("irrelevant", "일반 확인", "model", "low"); err != nil {
		t.Fatal(err)
	}
	if _, err := store.AddMessage("irrelevant", "assistant", "요청한 내용을 확인하고 답하세요.", "", nil, nil); err != nil {
		t.Fatal(err)
	}
	failed, err := store.AddPendingMessage("old", "검색되면 안 되는 실패 문장", nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.FailPendingTurn(failed.ID, MessageFailed, "failed", "", "", nil); err != nil {
		t.Fatal(err)
	}

	profile, err := store.AddMemory("user", "preferred", "응답 형식", "답변은 짧고 간결하게 작성", "", 0)
	if err != nil {
		t.Fatal(err)
	}
	durable, err := store.AddMemory("memory", "reference", "메모리 기준", "DGX Spark에서는 시스템 가용 메모리를 기준으로 판단", "old", 1)
	if err != nil {
		t.Fatal(err)
	}
	duplicate, err := store.AddMemory("memory", "preferred", "다른 제목", "중복 내용", "old", 1)
	if err != nil || duplicate.ID != durable.ID {
		t.Fatalf("source message duplicate=%+v original=%+v err=%v", duplicate, durable, err)
	}
	profiles, err := store.UserMemories(5)
	if err != nil || len(profiles) != 1 || profiles[0].Content != profile.Content {
		t.Fatalf("profiles=%+v err=%v", profiles, err)
	}
	memories, err := store.SearchMemories("Spark 메모리 상태", 5)
	if err != nil || len(memories) != 1 || memories[0].Content != durable.Content {
		t.Fatalf("memories=%+v err=%v", memories, err)
	}
	hits, err := store.SearchMessages("DGX Spark 메모리", "current", 5)
	if err != nil || len(hits) == 0 || hits[0].SessionID != "old" {
		t.Fatalf("hits=%+v err=%v", hits, err)
	}
	for _, hit := range hits {
		if hit.Content == "검색되면 안 되는 실패 문장" {
			t.Fatalf("failed message was indexed: %+v", hits)
		}
	}
	precise, err := store.SearchMessages("테스트용 사실입니다. 오로라복숭아 번호 내용을 확인하고 답하세요.", "current", 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(precise) != 0 {
		t.Fatalf("generic request wording produced unrelated recalls: %+v", precise)
	}
	short, err := store.SearchConversations("메모리", 5)
	if err != nil || len(short) != 1 || short[0].SessionID != "old" {
		t.Fatalf("short conversation search=%+v err=%v", short, err)
	}
	empty, err := store.SearchConversations("검색방", 5)
	if err != nil || len(empty) != 1 || empty[0].SessionID != "empty" || empty[0].MessageID != 0 {
		t.Fatalf("empty conversation search=%+v err=%v", empty, err)
	}
	none, err := store.SearchConversations("존재하지않는검색어", 5)
	if err != nil || none == nil || len(none) != 0 {
		t.Fatalf("empty result must be []: %#v err=%v", none, err)
	}
	if _, err := store.UpdateMemory(durable.ID, durable.Kind, durable.Priority, durable.Title, durable.Content, false); err != nil {
		t.Fatal(err)
	}
	memories, err = store.SearchMemories("Spark 메모리 상태", 5)
	if err != nil || len(memories) != 0 {
		t.Fatalf("disabled memory remained searchable: %+v err=%v", memories, err)
	}
}

func TestConversationSearchUsesBoundedCursorPages(t *testing.T) {
	store, err := Open(filepath.Join(t.TempDir(), "search-page.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.CreateSession("search", "대규모 검색", "model", "low"); err != nil {
		t.Fatal(err)
	}
	for index := 0; index < 25; index++ {
		if _, err := store.AddMessage("search", "assistant", fmt.Sprintf("대규모검색키워드 결과 %02d", index), "", nil, nil); err != nil {
			t.Fatal(err)
		}
	}
	first, cursor, err := store.SearchConversationPage("대규모검색키워드", ConversationSearchOptions{Limit: 20, Sort: "recent", Scope: "content"})
	if err != nil || len(first) != 20 || cursor == nil {
		t.Fatalf("first=%d cursor=%+v err=%v", len(first), cursor, err)
	}
	second, next, err := store.SearchConversationPage("대규모검색키워드", ConversationSearchOptions{Limit: 20, Sort: "recent", Scope: "content", CursorID: cursor.MessageID})
	if err != nil || len(second) != 5 || next != nil {
		t.Fatalf("second=%d cursor=%+v err=%v", len(second), next, err)
	}
	if first[len(first)-1].MessageID == second[0].MessageID {
		t.Fatal("cursor repeated the last item from the previous page")
	}
	relevantFirst, relevantCursor, err := store.SearchConversationPage("대규모검색키워드", ConversationSearchOptions{Limit: 10, Sort: "relevance", Scope: "content"})
	if err != nil || len(relevantFirst) != 10 || relevantCursor == nil {
		t.Fatalf("relevance first=%d cursor=%+v err=%v", len(relevantFirst), relevantCursor, err)
	}
	relevantSecond, _, err := store.SearchConversationPage("대규모검색키워드", ConversationSearchOptions{Limit: 10, Sort: "relevance", Scope: "content", CursorID: relevantCursor.MessageID, CursorRank: relevantCursor.Rank})
	if err != nil || len(relevantSecond) != 10 || relevantFirst[len(relevantFirst)-1].MessageID == relevantSecond[0].MessageID {
		t.Fatalf("relevance second=%d err=%v", len(relevantSecond), err)
	}
}

func TestMemoryRecallRanksExactTitleAboveIncidentalContent(t *testing.T) {
	store, err := Open(filepath.Join(t.TempDir(), "memory-rank.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.AddMemory("memory", "reference", "일반 메모", "DFlash2 튜닝은 나중에 확인한다.", "", 0); err != nil {
		t.Fatal(err)
	}
	exact, err := store.AddMemory("memory", "preferred", "DFlash2 튜닝", "draft token 8을 기준으로 사용한다.", "", 0)
	if err != nil {
		t.Fatal(err)
	}
	items, err := store.SearchMemories("DFlash2 튜닝", 2)
	if err != nil || len(items) != 2 || items[0].Title != exact.Title {
		t.Fatalf("ranked items=%+v err=%v", items, err)
	}
}

func TestFindMemoriesIncludesDisabledItemsAndIgnoresCommandWords(t *testing.T) {
	store, err := Open(filepath.Join(t.TempDir(), "memory-manage.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	item, err := store.AddMemory("memory", "preferred", "주력 장비", "주력 장비는 DGX Spark다.", "", 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.UpdateMemory(item.ID, item.Kind, item.Priority, item.Title, item.Content, false); err != nil {
		t.Fatal(err)
	}
	items, err := store.FindMemories("DGX Spark 기억을 보여줘", 8)
	if err != nil || len(items) != 1 || items[0].ID != item.ID || items[0].Enabled {
		t.Fatalf("managed search=%+v err=%v", items, err)
	}
	items, err = store.FindMemories("뭘 기억하고 있지", 8)
	if err != nil || len(items) != 1 || items[0].ID != item.ID {
		t.Fatalf("recent memory list=%+v err=%v", items, err)
	}
}
