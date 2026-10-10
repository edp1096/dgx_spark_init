package db

import (
	"context"
	"math"
	"path/filepath"
	"testing"
)

func unitEmbedding(axis int) []float32 { v := make([]float32, 768); v[axis] = 1; return v }
func encodePendingForTest(t *testing.T, d *DB, profile string) {
	t.Helper()
	sources, err := d.PendingEmbeddings(100)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range sources {
		if err = d.CompleteEmbedding(s, profile, []EmbeddingSegment{{Text: s.Content, Vector: unitEmbedding(0)}}); err != nil {
			t.Fatal(err)
		}
	}
}

func TestEmbeddingInvalidationAndStaleWriter(t *testing.T) {
	d, err := Open(filepath.Join(t.TempDir(), "retrieval.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	m, err := d.AddMemory("memory", "preferred", "설정", "예전 내용", "", 0)
	if err != nil {
		t.Fatal(err)
	}
	sources, err := d.PendingEmbeddings(10)
	if err != nil || len(sources) != 1 {
		t.Fatalf("pending=%+v %v", sources, err)
	}
	old := sources[0]
	if err = d.CompleteEmbedding(old, "profile", []EmbeddingSegment{{Text: old.Content, Vector: unitEmbedding(0)}}); err != nil {
		t.Fatal(err)
	}
	search := func() []SemanticMatch {
		t.Helper()
		rows, e := d.SearchSemantic(context.Background(), "profile", "memory", "", 0, 0, unitEmbedding(0), .5, 50)
		if e != nil {
			t.Fatal(e)
		}
		return rows
	}
	if len(search()) != 1 {
		t.Fatal("ready memory missing")
	}
	if _, err = d.UpdateMemory(m.ID, "memory", "preferred", "설정", "새 내용", true); err != nil {
		t.Fatal(err)
	}
	if len(search()) != 0 {
		t.Fatal("stale vector remained searchable")
	}
	if err = d.CompleteEmbedding(old, "profile", []EmbeddingSegment{{Text: "예전 내용", Vector: unitEmbedding(0)}}); err == nil {
		t.Fatal("stale worker published")
	}
	encodePendingForTest(t, d, "profile")
	if got := search(); len(got) != 1 || got[0].Source.Content != "새 내용" {
		t.Fatalf("updated=%+v", got)
	}
	if _, err = d.UpdateMemory(m.ID, "memory", "preferred", "설정", "새 내용", false); err != nil {
		t.Fatal(err)
	}
	if len(search()) != 0 {
		t.Fatal("disabled memory returned")
	}
	if err = d.DeleteMemory(m.ID); err != nil {
		t.Fatal(err)
	}
	var n int
	_ = d.conn.QueryRow("SELECT count(*) FROM retrieval_vectors").Scan(&n)
	if n != 0 {
		t.Fatal("deleted memory retained vectors")
	}
}

func TestEmbeddingScopeProfileAndSegmentDedup(t *testing.T) {
	d, err := Open(filepath.Join(t.TempDir(), "scope.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	for _, id := range []string{"old", "current"} {
		_, err = d.CreateSession(id, id, "model", "none")
		if err != nil {
			t.Fatal(err)
		}
	}
	old, _ := d.AddMessage("old", "assistant", "옛 대화의 GPU 설정", "", nil, nil)
	current, _ := d.AddMessage("current", "assistant", "현재 대화의 설정", "", nil, nil)
	_, _ = d.AddPendingMessage("old", "실패·미완료 결과", nil)
	document, _, err := d.AddKnowledgeDocument(KnowledgeDocument{ID: "doc-a", CollectionID: 1, Title: "원문", SHA256: "hash-a", StoragePath: "objects/a", Status: "processing"})
	if err != nil {
		t.Fatal(err)
	}
	if err = d.ReplaceKnowledgeChunks(document.ID, []KnowledgeChunk{{Ordinal: 0, PageStart: 3, PageEnd: 3, Content: "문서 설정"}}, 3, "ready", ""); err != nil {
		t.Fatal(err)
	}
	encodePendingForTest(t, d, "profile")
	matches, err := d.SearchSemantic(context.Background(), "profile", "message", "current", 0, 0, unitEmbedding(0), .5, 50)
	if err != nil || len(matches) != 1 || matches[0].Source.ID != old.ID {
		t.Fatalf("cross-session scope=%+v %v", matches, err)
	}
	matches, err = d.SearchSemantic(context.Background(), "profile", "message", "current", current.ID, 0, unitEmbedding(0), .5, 50)
	if err != nil || len(matches) != 1 || matches[0].Source.ID != current.ID {
		t.Fatalf("compacted scope=%+v %v", matches, err)
	}
	matches, err = d.SearchSemantic(context.Background(), "wrong-profile", "message", "current", 0, 0, unitEmbedding(0), .5, 50)
	if err != nil || len(matches) != 0 {
		t.Fatal("mixed model profiles")
	}
	matches, err = d.SearchSemantic(context.Background(), "profile", "knowledge", "", 0, 999, unitEmbedding(0), .5, 50)
	if err != nil || len(matches) != 0 {
		t.Fatal("escaped requested collection")
	}
	matches, err = d.SearchSemantic(context.Background(), "profile", "knowledge", "", 0, 1, unitEmbedding(0), .5, 50)
	if err != nil || len(matches) != 1 || matches[0].Source.PageStart != 3 {
		t.Fatalf("source provenance=%+v %v", matches, err)
	}
	_, _ = d.conn.Exec("UPDATE knowledge_collections SET enabled=0 WHERE id=1")
	matches, err = d.SearchSemantic(context.Background(), "profile", "knowledge", "", 0, 0, unitEmbedding(0), .5, 50)
	if err != nil || len(matches) != 0 {
		t.Fatal("disabled collection returned")
	}
}

func TestEmbeddingRejectsCorruptVectors(t *testing.T) {
	for _, v := range [][]float32{nil, make([]float32, 768)} {
		if _, err := normalizeVector(v); err == nil {
			t.Fatal("invalid vector accepted")
		}
	}
	v := unitEmbedding(0)
	v[1] = float32(math.NaN())
	if _, err := normalizeVector(v); err == nil {
		t.Fatal("NaN accepted")
	}
}
