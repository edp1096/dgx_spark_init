package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/embedding"
	"sparktalk/internal/orchestrator"
	"testing"
)

func retrievalUnitVector(axis int) []float32 { v := make([]float32, 768); v[axis] = 1; return v }
func TestRankFusionUsesUnionAndStableIDs(t *testing.T) {
	result := fuseRanked([]string{"a", "b", "c"}, []string{"d", "b", "a"}, func(s string) string { return s }, 4)
	if len(result) != 4 || result[0] != "a" || result[1] != "b" || result[2] != "d" || result[3] != "c" {
		t.Fatalf("fusion=%v", result)
	}
	result = fuseRanked([]string{"a", "a"}, []string{"b"}, func(s string) string { return s }, 2)
	if len(result) != 2 {
		t.Fatalf("duplicates=%v", result)
	}
}

func TestQADIgnoresCachedSemanticQueriesAndKeepsFTS(t *testing.T) {
	d, err := db.Open(filepath.Join(t.TempDir(), "qad.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	m, err := d.AddMemory("memory", "reference", "QAD 설정", "KV 용량은 512K를 사용한다.", "", 0)
	if err != nil {
		t.Fatal(err)
	}
	sources, _ := d.PendingEmbeddings(1)
	if err = d.CompleteEmbedding(sources[0], embedding.Profile, []db.EmbeddingSegment{{Text: m.Content, Vector: retrievalUnitVector(0)}}); err != nil {
		t.Fatal(err)
	}
	calls := 0
	endpoint := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		_ = json.NewEncoder(w).Encode(embedding.Response{Profile: embedding.Profile, Dimensions: 768, Data: []embedding.Segment{{Text: "질문", Vector: retrievalUnitVector(0)}}})
	}))
	defer endpoint.Close()
	cat, _ := orchestrator.LoadCatalog()
	s := &Server{db: d, cfg: config.Config{Runtime: config.RuntimeConfig{Mode: "managed", ActiveBundle: "qwen38fn_exl3", Catalog: &cat}, Embedding: config.EmbeddingConfig{Enabled: true, Endpoint: endpoint.URL, Timeout: "1s"}}}
	if v, err := s.queryVector(context.Background(), "QAD 설정"); err != nil || len(v) != 768 {
		t.Fatal(err)
	}
	s.cfg.Runtime.ActiveBundle = "flash-next"
	if v, err := s.queryVector(context.Background(), "QAD 설정"); err != nil || len(v) != 0 || calls != 1 {
		t.Fatal("QAD used cached semantic query", err)
	}
	results, err := s.hybridRecall(context.Background(), "memory", "512K", "current", 0, 5)
	if err != nil || len(results) != 1 || results[0].MemoryID != m.ID || calls != 1 {
		t.Fatal("QAD FTS fallback", results, err)
	}
}
func TestHybridRecallIndependentSemanticCandidatesAndFailureFallback(t *testing.T) {
	d, err := db.Open(filepath.Join(t.TempDir(), "hybrid.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	m, err := d.AddMemory("memory", "preferred", "환경 설정", "그래픽 장치의 작업 공간 확보를 위해 KV 용량을 조정했다.", "", 0)
	if err != nil {
		t.Fatal(err)
	}
	sources, err := d.PendingEmbeddings(10)
	if err != nil {
		t.Fatal(err)
	}
	if err = d.CompleteEmbedding(sources[0], embedding.Profile, []db.EmbeddingSegment{{Text: m.Content, Vector: retrievalUnitVector(0)}}); err != nil {
		t.Fatal(err)
	}
	endpoint := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(embedding.Response{Profile: embedding.Profile, Dimensions: 768, Data: []embedding.Segment{{Text: "질문", Vector: retrievalUnitVector(0)}}})
	}))
	defer endpoint.Close()
	s := &Server{db: d, cfg: config.Config{Runtime: config.RuntimeConfig{Mode: "external"}, Embedding: config.EmbeddingConfig{Enabled: true, Endpoint: endpoint.URL, Timeout: "1s", MinSimilarity: .5}}}
	result, err := s.hybridRecall(context.Background(), "memory", "그림을 못 그렸던 원인은?", "current", 0, 5)
	if err != nil || len(result) != 1 || result[0].MemoryID != m.ID || result[0].Priority != "preferred" {
		t.Fatalf("independent semantic recall=%+v %v", result, err)
	}
	s.cfg.Embedding.Endpoint = endpoint.URL + "/missing"
	endpoint.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { http.Error(w, "unavailable", 503) })
	result, err = s.hybridRecall(context.Background(), "memory", "그래픽 장치 작업 공간", "current", 0, 5)
	if err != nil || len(result) != 1 || result[0].MemoryID != m.ID {
		t.Fatalf("FTS fallback=%+v %v", result, err)
	}
}

func TestRetrievalPreviewDoesNotStartGPUAndQuotedNamesRemainExact(t *testing.T) {
	d, err := db.Open(filepath.Join(t.TempDir(), "preview.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	_, _ = d.AddMemory("memory", "reference", "Q8_0", "검증한 본체 Q8_0", "", 0)
	source, _ := d.PendingEmbeddings(1)
	if err = d.CompleteEmbedding(source[0], embedding.Profile, []db.EmbeddingSegment{{Text: source[0].Content, Vector: retrievalUnitVector(0)}}); err != nil {
		t.Fatal(err)
	}
	calls := 0
	endpoint := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls++; http.Error(w, "should not run", 500) }))
	defer endpoint.Close()
	s := &Server{db: d, cfg: config.Config{Embedding: config.EmbeddingConfig{Enabled: true, Endpoint: endpoint.URL, Timeout: "1s"}}}
	vector, err := s.queryVector(context.WithValue(context.Background(), retrievalPreviewKey{}, true), "설정")
	if err != nil || len(vector) != 0 || calls != 0 {
		t.Fatal("preview initiated embedding request")
	}
	got := filterExact([]string{"Q4_0 본체", "Q8_0 본체"}, `"Q8_0" 본체`, func(s string) string { return s })
	if len(got) != 1 || got[0] != "Q8_0 본체" {
		t.Fatalf("quoted identity=%v", got)
	}
}
