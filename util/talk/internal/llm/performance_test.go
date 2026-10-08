package llm

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"sparktalk/internal/performance"
)

func TestStreamPerformanceUsesUsageAndDecodeTimeNotChunksOrOverallTPS(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, `data: {"choices":[{"delta":{"role":"assistant","content":""}}]}

data: {"choices":[{"delta":{"reasoning":"multi token reasoning burst"}}]}

data: {"choices":[{"delta":{"content":"answer burst"},"finish_reason":"stop"}]}

data: {"choices":[],"usage":{"prompt_tokens":1000,"completion_tokens":21,"total_tokens":1021,"prompt_tokens_details":{"cached_tokens":800}},"metrics":{"time_to_first_token_ms":200,"generation_time_ms":1000,"queue_time_ms":300,"tokens_per_second":17.5}}

data: [DONE]

`)
	}))
	defer backend.Close()
	var accumulated performance.Accumulator
	ctx := WithPerformanceObserver(context.Background(), accumulated.Start)
	result, err := New(backend.URL, "model", "").Stream(ctx, []Message{{Role: "user", Content: "test"}}, "model", "none", nil, func(string, string) error { return nil })
	if err != nil || result.Reasoning == "" || result.Content == "" {
		t.Fatalf("stream failed: %+v %v", result, err)
	}
	s := accumulated.Summary(false)
	if *s.PP != 1000 || *s.TG != 20 || *s.TTFT != .5 || s.PPEstimated || s.TGEstimated || s.TTFTEstimated || s.OutputTokens != 21 {
		t.Fatalf("incorrect native metrics: %+v", s)
	}
}

func TestFallbackIgnoresRoleChunksAndIncludesReasoning(t *testing.T) {
	start := time.Unix(100, 0)
	p := &streamPerformance{started: start}
	p.generated("", start.Add(time.Second))
	if !p.first.IsZero() {
		t.Fatal("empty role chunk counted as generation")
	}
	p.generated("thinking", start.Add(2*time.Second))
	p.generated("answer", start.Add(3*time.Second))
	s := p.sample()
	if s.TTFT == nil || *s.TTFT != 2 || s.DecodeSeconds != 1 || !s.TGEstimated || !s.TTFTEstimated {
		t.Fatalf("incorrect fallback: %+v", s)
	}
}

func TestLlamaTimingsAndSingleTokenDoNotInventDecodeRate(t *testing.T) {
	p := &streamPerformance{timings: &responseTimings{PromptN: 120, PromptMS: 200, PredictedN: 30, PredictedMS: 1500}}
	s := p.sample()
	if s.PrefillTokens/s.PrefillSeconds != 600 || s.DecodeTokens/s.DecodeSeconds != 20 || s.PPEstimated || s.TGEstimated {
		t.Fatal(s)
	}
	zero := 0.0
	p = &streamPerformance{usage: Usage{CompletionTokens: 1}, metrics: &responseMetrics{GenerationTimeMS: &zero}}
	s = p.sample()
	if s.DecodeTokens != 0 || s.DecodeSeconds != 0 {
		t.Fatal(s)
	}
}
