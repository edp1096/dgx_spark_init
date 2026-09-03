package server

import (
	"path/filepath"
	"strings"
	"testing"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
)

func TestRecallCombinesProfileMemoryAndPastSessionWithinBudget(t *testing.T) {
	store, err := db.Open(filepath.Join(t.TempDir(), "recall.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.CreateSession("old", "Spark 메모리", "model", "low"); err != nil {
		t.Fatal(err)
	}
	if _, err := store.CreateSession("current", "현재", "model", "low"); err != nil {
		t.Fatal(err)
	}
	_, _ = store.AddMessage("old", "assistant", "DGX Spark 통합 메모리는 시스템 가용량으로 확인한다.", "", nil, nil)
	_, _ = store.AddMemory("user", "preferred", "말투", "항상 짧고 명확하게 답한다.", "", 0)
	_, _ = store.AddMemory("memory", "reference", "Spark", "DGX Spark 메모리 측정은 free 명령도 함께 본다.", "old", 1)

	server := &Server{db: store}
	cfg := config.MemoryConfig{Enabled: true, RecallSessions: true, AlwaysMaxResults: 1, AlwaysTokenBudget: 256, MaxResults: 2, TokenBudget: 512}
	items, prompt, tokens, err := server.buildRecallContext("current", []db.Message{{Role: "user", Content: "DGX Spark 메모리 측정 방법"}}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 3 || tokens <= 0 || tokens > cfg.AlwaysTokenBudget+cfg.TokenBudget {
		t.Fatalf("items=%+v tokens=%d", items, tokens)
	}
	for _, want := range []string{"Preferred memory", "Related memory", "Past conversation"} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("missing %q in %s", want, prompt)
		}
	}
	if !strings.Contains(prompt, "user-authoritative facts") {
		t.Fatalf("preferred-memory authority missing: %s", prompt)
	}
}

func TestRecallUsesRecentTurnToResolveDeicticQuestion(t *testing.T) {
	store, err := db.Open(filepath.Join(t.TempDir(), "deictic.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.CreateSession("current", "현재", "model", "low"); err != nil {
		t.Fatal(err)
	}
	_, _ = store.AddMemory("memory", "preferred", "DFlash2 설정", "Qwen DFlash2의 draft token은 8을 기준으로 측정했다.", "", 0)
	messages := []db.Message{
		{Role: "user", Content: "Qwen DFlash2 draft token 설정을 이야기했었지?"},
		{Role: "assistant", Content: "네, draft token 8을 기준으로 확인했습니다."},
		{Role: "user", Content: "아까 그 설정 다시 알려줘."},
	}
	cfg := config.MemoryConfig{Enabled: true, AlwaysMaxResults: 1, AlwaysTokenBudget: 256, MaxResults: 3, TokenBudget: 512}
	items, prompt, _, err := (&Server{db: store}).buildRecallContext("current", messages, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 1 || items[0].Kind != "memory" || !strings.Contains(prompt, "draft token") {
		t.Fatalf("deictic recall failed: items=%+v prompt=%s", items, prompt)
	}
}

func TestRecallMarksPreferredMemoryAsUserAuthoritative(t *testing.T) {
	store, err := db.Open(filepath.Join(t.TempDir(), "preferred-recall.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.CreateSession("current", "현재", "model", "low"); err != nil {
		t.Fatal(err)
	}
	if _, err := store.AddMemory("memory", "preferred", "우리나라 대통령 이름은?", "이명박", "", 0); err != nil {
		t.Fatal(err)
	}
	cfg := config.MemoryConfig{Enabled: true, RecallSessions: false, AlwaysMaxResults: 1, AlwaysTokenBudget: 128, MaxResults: 3, TokenBudget: 512}
	items, prompt, _, err := (&Server{db: store}).buildRecallContext("current", []db.Message{{Role: "user", Content: "우리나라 대통령 이름은?"}}, cfg)
	if err != nil || len(items) != 1 || items[0].Priority != "preferred" {
		t.Fatalf("preferred recall=%+v err=%v", items, err)
	}
	for _, want := range []string{"Preferred memory", "이명박", "general knowledge or web sources disagree", "never mention or imply retrieved memory", "Do not offer to verify"} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("preferred recall prompt missing %q: %s", want, prompt)
		}
	}
}
