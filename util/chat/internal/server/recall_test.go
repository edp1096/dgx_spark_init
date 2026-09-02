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
	_, _ = store.AddMemory("user", "말투", "항상 짧고 명확하게 답한다.", "", 0)
	_, _ = store.AddMemory("memory", "Spark", "DGX Spark 메모리 측정은 free 명령도 함께 본다.", "old", 1)

	server := &Server{db: store}
	items, prompt, tokens, err := server.buildRecallContext("current", []db.Message{{Role: "user", Content: "DGX Spark 메모리 측정 방법"}}, config.MemoryConfig{Enabled: true, RecallSessions: true, MaxResults: 5, TokenBudget: 512})
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 3 || tokens <= 0 || tokens > 512 {
		t.Fatalf("items=%+v tokens=%d", items, tokens)
	}
	for _, want := range []string{"User profile", "Durable memory", "Past conversation"} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("missing %q in %s", want, prompt)
		}
	}
}
