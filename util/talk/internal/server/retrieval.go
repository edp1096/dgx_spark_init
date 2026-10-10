package server

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"sparktalk/internal/db"
	"sparktalk/internal/embedding"
)

const retrievalCandidates = 50
const rrfConstant = 60.0

type retrievalPreviewKey struct{}

type queryEmbedding struct {
	vector  []float32
	expires time.Time
}
type retrievalState struct {
	queries atomic.Int32
	error   string
	cache   map[[32]byte]queryEmbedding
}

func (s *Server) encodeEmbedding(ctx context.Context, task string, inputs []embedding.Input) (embedding.Response, error) {
	cfg, _ := s.snapshot()
	if !cfg.SemanticSearchEnabled() {
		return embedding.Response{}, fmt.Errorf("semantic search is unavailable in the current set")
	}
	release, err := s.acquireWorkload(ctx, "extra-embedding")
	if err != nil {
		return embedding.Response{}, err
	}
	result, err := embedding.Encode(ctx, cfg.Embedding.Endpoint, task, inputs)
	// Transport cancellation need not mean CUDA has stopped; the lifecycle API
	// refuses reclamation while the worker is still busy.
	if cleanup := release(err); err == nil && cleanup != nil {
		err = cleanup
	}
	return result, err
}

func (s *Server) queryVector(ctx context.Context, query string) ([]float32, error) {
	cfg, _ := s.snapshot()
	if !cfg.SemanticSearchEnabled() {
		return nil, nil
	}
	ready, err := s.db.HasEmbeddingVectors(embedding.Profile)
	if err != nil || !ready {
		return nil, err
	}
	key := sha256.Sum256([]byte(cfg.Embedding.Endpoint + "\x00" + query))
	s.retrievalMu.Lock()
	cached, ok := s.retrieval.cache[key]
	s.retrievalMu.Unlock()
	if ok && time.Now().Before(cached.expires) {
		return cached.vector, nil
	}
	if preview, _ := ctx.Value(retrievalPreviewKey{}).(bool); preview {
		return nil, nil
	}
	s.retrieval.queries.Add(1)
	defer s.retrieval.queries.Add(-1)
	timeout, _ := time.ParseDuration(cfg.Embedding.Timeout)
	if timeout <= 0 {
		timeout = 20 * time.Second
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	result, err := s.encodeEmbedding(ctx, "query", []embedding.Input{{Text: query}})
	s.retrievalMu.Lock()
	defer s.retrievalMu.Unlock()
	if err != nil {
		s.retrieval.error = err.Error()
		return nil, err
	}
	s.retrieval.error = ""
	if s.retrieval.cache == nil || len(s.retrieval.cache) >= 64 {
		s.retrieval.cache = make(map[[32]byte]queryEmbedding)
	}
	vector := result.Data[0].Vector
	s.retrieval.cache[key] = queryEmbedding{vector, time.Now().Add(5 * time.Minute)}
	return vector, nil
}

func (s *Server) startEmbeddingIndexer() {
	ctx, finish, err := s.tasks.Track(s.tasks.Context())
	if err != nil {
		return
	}
	go func() {
		defer finish()
		ticker := time.NewTicker(250 * time.Millisecond)
		defer ticker.Stop()
		profile := ""
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}
			cfg, _ := s.snapshot()
			if !cfg.SemanticSearchEnabled() || s.retrieval.queries.Load() > 0 {
				continue
			}
			if profile != embedding.Profile {
				if err := s.db.EnsureEmbeddingJobs(embedding.Profile); err != nil {
					continue
				}
				profile = embedding.Profile
			}
			sources, err := s.db.PendingEmbeddings(1)
			if err != nil || len(sources) == 0 {
				continue
			}
			source := sources[0]
			jobCtx, cancel := context.WithTimeout(ctx, 5*time.Minute)
			segments := []db.EmbeddingSegment{}
			runes := []rune(source.Content)
			for start := 0; start < len(runes); start += 7872 {
				// Yield between bounded inputs so foreground queries have admission priority.
				for s.retrieval.queries.Load() > 0 && jobCtx.Err() == nil {
					select {
					case <-jobCtx.Done():
					case <-time.After(50 * time.Millisecond):
					}
				}
				cfg, _ = s.snapshot()
				if !cfg.SemanticSearchEnabled() {
					err = fmt.Errorf("semantic search disabled")
					break
				}
				response, e := s.encodeEmbedding(jobCtx, "document", []embedding.Input{{Title: source.Title, Text: string(runes[start:min(start+8000, len(runes))])}})
				if e != nil {
					err = e
					break
				}
				for _, p := range response.Data {
					segments = append(segments, db.EmbeddingSegment{Text: p.Text, Vector: p.Vector})
				}
			}
			if err == nil {
				err = s.db.CompleteEmbedding(source, embedding.Profile, segments)
			}
			cancel()
			if err != nil {
				_ = s.db.FailEmbedding(source, err.Error())
			}
			s.retrievalMu.Lock()
			if err != nil {
				s.retrieval.error = err.Error()
			} else {
				s.retrieval.error = ""
			}
			s.retrievalMu.Unlock()
		}
	}()
}

// Stable reciprocal rank fusion combines unrelated score scales. FTS is the
// deterministic tie breaker; source identifiers deduplicate either result list.
func fuseRanked[T any](lexical, semantic []T, key func(T) string, limit int) []T {
	type entry struct {
		item  T
		score float64
		order int
	}
	entries := map[string]*entry{}
	order := 0
	for _, list := range [][]T{lexical, semantic} {
		seen := map[string]bool{}
		for rank, item := range list {
			k := key(item)
			if seen[k] {
				continue
			}
			seen[k] = true
			e := entries[k]
			if e == nil {
				e = &entry{item: item, order: order}
				entries[k] = e
				order++
			}
			e.score += 1 / (rrfConstant + float64(rank+1))
		}
	}
	sorted := make([]*entry, 0, len(entries))
	for _, e := range entries {
		sorted = append(sorted, e)
	}
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].score == sorted[j].score {
			return sorted[i].order < sorted[j].order
		}
		return sorted[i].score > sorted[j].score
	})
	out := make([]T, 0, min(limit, len(sorted)))
	for _, e := range sorted {
		if len(out) == limit {
			break
		}
		out = append(out, e.item)
	}
	return out
}

func (s *Server) hybridRecall(ctx context.Context, kind, query, session string, through int64, limit int) ([]db.RecallItem, error) {
	if limit <= 0 {
		return nil, nil
	}
	cfg, _ := s.snapshot()
	candidates := limit
	if cfg.SemanticSearchEnabled() {
		candidates = retrievalCandidates
	}
	var lexical []db.RecallItem
	var err error
	switch {
	case kind == "memory":
		lexical, err = s.db.SearchMemories(query, candidates)
	case through > 0:
		lexical, err = s.db.SearchCompactedMessages(query, session, through, candidates)
	default:
		lexical, err = s.db.SearchMessages(query, session, candidates)
	}
	if err != nil {
		return nil, err
	}
	vector, semanticErr := s.queryVector(ctx, query)
	if semanticErr != nil || len(vector) == 0 {
		if len(lexical) > limit {
			lexical = lexical[:limit]
		}
		return lexical, nil
	}
	matches, semanticErr := s.db.SearchSemantic(ctx, embedding.Profile, kind, session, through, 0, vector, cfg.Embedding.MinSimilarity, retrievalCandidates)
	if semanticErr != nil {
		if len(lexical) > limit {
			lexical = lexical[:limit]
		}
		return lexical, nil
	}
	semantic := make([]db.RecallItem, 0, len(matches))
	for _, m := range matches {
		x := m.Source
		item := db.RecallItem{Kind: "session", Title: x.Title, Content: x.Content, Priority: x.Priority, SessionID: x.SessionID, Role: x.Role, MessageID: x.ReferenceMessageID, CreatedAt: x.CreatedAt}
		if kind == "memory" {
			item.Kind = "memory"
			item.MemoryID = x.ID
			item.MessageID = x.ReferenceMessageID
		}
		semantic = append(semantic, item)
	}
	key := func(x db.RecallItem) string {
		if x.Kind == "memory" {
			return fmt.Sprintf("memory:%d", x.MemoryID)
		}
		return fmt.Sprintf("message:%d", x.MessageID)
	}
	lexical = filterExact(lexical, query, func(x db.RecallItem) string { return x.Title + "\n" + x.Content })
	semantic = filterExact(semantic, query, func(x db.RecallItem) string { return x.Title + "\n" + x.Content })
	merged := fuseRanked(lexical, semantic, key, retrievalCandidates)
	out := make([]db.RecallItem, 0, limit)
	seenSessions := map[string]bool{}
	for _, x := range merged {
		if kind == "message" && through == 0 {
			if seenSessions[x.SessionID] {
				continue
			}
			seenSessions[x.SessionID] = true
		}
		out = append(out, x)
		if len(out) == limit {
			break
		}
	}
	return out, nil
}

func (s *Server) hybridKnowledge(ctx context.Context, query string, collection int64, limit int) ([]db.KnowledgeSearchResult, error) {
	if limit < 1 || limit > 50 {
		limit = 12
	}
	cfg, _ := s.snapshot()
	candidates := limit
	if cfg.SemanticSearchEnabled() {
		candidates = retrievalCandidates
	}
	lexical, err := s.db.SearchKnowledge(query, collection, candidates)
	if err != nil {
		return nil, err
	}
	vector, semanticErr := s.queryVector(ctx, query)
	if semanticErr != nil || len(vector) == 0 {
		if len(lexical) > limit {
			lexical = lexical[:limit]
		}
		return lexical, nil
	}
	matches, semanticErr := s.db.SearchSemantic(ctx, embedding.Profile, "knowledge", "", 0, collection, vector, cfg.Embedding.MinSimilarity, retrievalCandidates)
	if semanticErr != nil {
		if len(lexical) > limit {
			lexical = lexical[:limit]
		}
		return lexical, nil
	}
	semantic := make([]db.KnowledgeSearchResult, 0, len(matches))
	for _, m := range matches {
		x := m.Source
		semantic = append(semantic, db.KnowledgeSearchResult{ChunkID: x.ID, Ordinal: x.Ordinal, DocumentID: x.DocumentID, CollectionID: x.CollectionID, Title: x.Title, SourceURL: x.SourceURL, Content: x.Content, PageStart: x.PageStart, PageEnd: x.PageEnd})
	}
	lexical = filterExact(lexical, query, func(x db.KnowledgeSearchResult) string { return x.Title + "\n" + x.Content })
	semantic = filterExact(semantic, query, func(x db.KnowledgeSearchResult) string { return x.Title + "\n" + x.Content })
	return fuseRanked(lexical, semantic, func(x db.KnowledgeSearchResult) string { return fmt.Sprintf("%d", x.ChunkID) }, limit), nil
}

func (s *Server) retrievalStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	cfg, _ := s.snapshot()
	counts, err := s.db.EmbeddingCounts(embedding.Profile)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	s.retrievalMu.Lock()
	detail := s.retrieval.error
	s.retrievalMu.Unlock()
	data := map[string]any{"enabled": cfg.SemanticSearchEnabled(), "profile": embedding.Profile, "counts": counts, "error": detail}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(data)
}

var quotedAnchor = regexp.MustCompile(`"([^"\n]{2,200})"`)

func filterExact[T any](items []T, query string, text func(T) string) []T {
	anchors := quotedAnchor.FindAllStringSubmatch(query, -1)
	if len(anchors) == 0 {
		return items
	}
	out := make([]T, 0, len(items))
	for _, item := range items {
		body := strings.ToLower(text(item))
		match := true
		for _, a := range anchors {
			if !strings.Contains(body, strings.ToLower(a[1])) {
				match = false
				break
			}
		}
		if match {
			out = append(out, item)
		}
	}
	return out
}
