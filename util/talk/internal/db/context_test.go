package db

import (
	"path/filepath"
	"testing"
)

func TestContextSegmentsPreserveTranscriptAndCascadeWithSession(t *testing.T) {
	store, err := Open(filepath.Join(t.TempDir(), "context.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	if _, err := store.CreateSession("s1", "context", "model", "medium"); err != nil {
		t.Fatal(err)
	}
	first, _ := store.AddMessage("s1", "user", "first", "", nil, nil)
	last, _ := store.AddMessage("s1", "assistant", "answer", "", nil, nil)
	if _, err := store.AddContextSegment("s1", first.ID, last.ID, "segment summary", "cumulative checkpoint", "model", 42); err != nil {
		t.Fatal(err)
	}
	segments, err := store.ContextSegments("s1")
	if err != nil || len(segments) != 1 || segments[0].Checkpoint != "cumulative checkpoint" {
		t.Fatalf("unexpected segments: %+v err=%v", segments, err)
	}
	messages, err := store.Messages("s1")
	if err != nil || len(messages) != 2 || messages[0].Content != "first" {
		t.Fatalf("compaction changed the transcript: %+v err=%v", messages, err)
	}
	if err := store.DeleteSession("s1"); err != nil {
		t.Fatal(err)
	}
	segments, err = store.ContextSegments("s1")
	if err != nil || len(segments) != 0 {
		t.Fatalf("segments did not cascade: %+v err=%v", segments, err)
	}
	messages, err = store.Messages("s1")
	if err != nil || len(messages) != 0 {
		t.Fatalf("messages did not delete with session: %+v err=%v", messages, err)
	}
	hits, err := store.SearchMessages("first answer", "other", 5)
	if err != nil || len(hits) != 0 {
		t.Fatalf("deleted messages remained searchable: %+v err=%v", hits, err)
	}
}
