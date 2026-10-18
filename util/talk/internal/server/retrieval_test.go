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
	"strings"
	"testing"
	"unicode/utf8"
)

func retrievalUnitVector(axis int) []float32 { v := make([]float32, 768); v[axis] = 1; return v }

func TestLongNewConversationUsesBoundedSemanticQuery(t *testing.T) {
	d, err := db.Open(filepath.Join(t.TempDir(), "long-query.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	_, _ = d.AddMemory("memory", "reference", "검색 설정", "EmbeddingGemma 2는 의미 검색에 사용한다.", "", 0)
	sources, _ := d.PendingEmbeddings(1)
	if err = d.CompleteEmbedding(sources[0], embedding.Profile, []db.EmbeddingSegment{{Text: sources[0].Content, Vector: retrievalUnitVector(0)}}); err != nil {
		t.Fatal(err)
	}
	const topic = "검색할 주제: EmbeddingGemma 2\n"
	const question = "마지막 질문: 의미 검색 모델을 어떻게 실행하지?"
	query := topic + strings.Repeat("긴 참고 자료🙂 ", 50000) + question
	original := query
	calls := 0
	endpoint := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		var in struct {
			Task   string            `json:"task"`
			Inputs []embedding.Input `json:"inputs"`
		}
		if r.ContentLength > 256*1024 {
			http.Error(w, "invalid request size", 400)
			return
		}
		if err := json.NewDecoder(r.Body).Decode(&in); err != nil || in.Task != "query" || len(in.Inputs) != 1 {
			http.Error(w, "invalid query", 400)
			return
		}
		text := in.Inputs[0].Text
		if len(text) > semanticQueryMaxBytes || !utf8.ValidString(text) || !strings.HasPrefix(text, topic) || !strings.HasSuffix(text, question) {
			t.Error("semantic excerpt lost the topic, final question or UTF-8 boundary")
		}
		_ = json.NewEncoder(w).Encode(embedding.Response{Profile: embedding.Profile, Dimensions: 768, Data: []embedding.Segment{{Text: text, Vector: retrievalUnitVector(0)}}})
	}))
	defer endpoint.Close()
	s := &Server{db: d, cfg: config.Config{Runtime: config.RuntimeConfig{Mode: "external"}, Embedding: config.EmbeddingConfig{Enabled: true, Endpoint: endpoint.URL, Timeout: "1s"}}}
	for range 2 {
		v, err := s.queryVector(context.Background(), query)
		if err != nil || len(v) != embedding.Dimensions {
			t.Fatal("large new-conversation query failed", err)
		}
	}
	if calls != 1 || query != original {
		t.Fatal("semantic cache missed or original conversation changed")
	}
	short := "임베딩 모델을 실행해 줘"
	if semanticQueryExcerpt(short) != short {
		t.Fatal("short query changed")
	}
}
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
