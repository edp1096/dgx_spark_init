package llm

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"sparktalk/internal/performance"
)

func TestTensorFoldNativeRatesExcludePrefillAndCachedTokens(t *testing.T) {
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, `data: {"choices":[{"delta":{"content":"one burst, many tokens"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1000,"completion_tokens":81,"total_tokens":1081,"prompt_tokens_details":{"cached_tokens":800}},"tensorfold":{"seconds":5,"prefill_seconds":0.1,"time_to_first_token":0.4,"tokens_per_second":80}}

data: [DONE]

`)
	}))
	defer api.Close()
	var accumulated performance.Accumulator
	ctx := WithPerformanceObserver(context.Background(), accumulated.Start)
	_, err := New(api.URL, "DeepSeek-V4.1-Flash-TF", "", "deepseek-v4").Stream(ctx, []Message{{Role: "user", Content: "test"}}, "DeepSeek-V4.1-Flash-TF", "off", nil, func(string, string) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	s := accumulated.Summary(false)
	if s.PP == nil || *s.PP != 2000 || s.TG == nil || *s.TG != 80 || s.TTFT == nil || *s.TTFT != .4 || s.PPEstimated || s.TGEstimated || s.TTFTEstimated {
		t.Fatalf("TensorFold native metrics lost or wall time used as decode: %+v", s)
	}
}

func TestTensorFoldMissingOrZeroMetricsKeepFallback(t *testing.T) {
	zero := 0.0
	p := &streamPerformance{usage: Usage{CompletionTokens: 1}, tensorfold: &tensorFoldTimings{TokensPerSecond: &zero}}
	s := p.sample()
	if s.DecodeTokens != 0 || s.DecodeSeconds != 0 || s.TTFT != nil || !s.PPEstimated {
		t.Fatal("invented a measurement for an absent/one-token interval", s)
	}
}

func TestTensorFoldSparkSurfaceUsesMeasuredRoundTimes(t *testing.T) {
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, `data: {"choices":[{"delta":{"content":"counting burst"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1000,"completion_tokens":81,"prompt_tokens_details":{"cached_tokens":800}},"tensorfold":{"prompt":1000,"cached":800,"completion":81,"prefill_s":0.1,"decode_s":1,"ttft_s":0.4}}

data: [DONE]

`)
	}))
	defer api.Close()
	var accumulated performance.Accumulator
	ctx := WithPerformanceObserver(context.Background(), accumulated.Start)
	_, err := New(api.URL, "model", "").Stream(ctx, []Message{{Role: "user", Content: "test"}}, "model", "off", nil, func(string, string) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	s := accumulated.Summary(false)
	if s.PP == nil || *s.PP != 2000 || s.TG == nil || *s.TG != 80 || s.TTFT == nil || *s.TTFT != .4 || s.PPEstimated || s.TGEstimated || s.TTFTEstimated {
		t.Fatalf("Spark native metrics lost: %+v", s)
	}
}

func TestTensorFoldSparkFirstTokenIsNotCountedAsDecode(t *testing.T) {
	decode := 0.5
	for _, n := range []int{1, 2, 239} {
		p := &streamPerformance{usage: Usage{CompletionTokens: n}, tensorfold: &tensorFoldTimings{SparkCompletion: &n, SparkDecodeSeconds: &decode}}
		var accumulated performance.Accumulator
		accumulated.Start()(p.sample())
		s := accumulated.Summary(false)
		if n == 1 {
			if s.TG != nil {
				t.Fatal("a prefill-only first token invented a decode rate", s)
			}
		} else if s.TG == nil || *s.TG != float64(n-1)/decode || s.TGEstimated {
			t.Fatalf("first token counted in decode: n=%d sample=%+v", n, s)
		}
	}
}
