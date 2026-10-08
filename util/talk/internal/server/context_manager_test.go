package server

import (
	"strings"
	"testing"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/media"
)

func TestContextEstimatorAndCutKeepCompleteRecentTurns(t *testing.T) {
	items := []db.Message{
		{ID: 1, Role: "user", Content: strings.Repeat("a", 120)},
		{ID: 2, Role: "assistant", Content: strings.Repeat("b", 120)},
		{ID: 3, Role: "user", Content: "두 번째 질문"},
		{ID: 4, Role: "assistant", Content: "두 번째 답변"},
		{ID: 5, Role: "user", Content: "최근 질문"},
		{ID: 6, Role: "assistant", Content: "최근 답변"},
	}
	cut := selectCompactionCut(items, 40, 2048, false)
	if cut <= 0 || items[cut-1].Role != "assistant" || len(items)-cut < 2 {
		t.Fatalf("invalid compaction cut %d", cut)
	}
	if estimateTextTokens("abcd") != 1 || estimateTextTokens("한글") != 2 {
		t.Fatal("token estimator did not apply conservative mixed-language accounting")
	}
}

func TestContextTranscriptUsesAttachmentDescriptorsNotPayloads(t *testing.T) {
	store, err := media.New(t.TempDir() + "/context.db")
	if err != nil {
		t.Fatal(err)
	}
	s := &Server{media: store}
	text := s.contextTranscript([]db.Message{{ID: 7, Role: "user", Content: "설명", Attachments: []db.Attachment{{ID: "img", Name: "photo.png", MIME: "image/png", Size: 123}}}}, config.Config{})
	if !strings.Contains(text, "photo.png") || !strings.Contains(text, "image/png") || strings.Contains(text, "base64") {
		t.Fatalf("unexpected transcript: %s", text)
	}
}
