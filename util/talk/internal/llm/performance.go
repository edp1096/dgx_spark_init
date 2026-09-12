package llm

import (
	"context"
	"math"
	"time"

	"sparktalk/internal/performance"
)

type performanceObserverKey struct{}

// Each model request owns one replaceable measurement. Tool execution,
// approvals and other work between requests are outside these intervals.
func WithPerformanceObserver(ctx context.Context, begin func() func(performance.Sample)) context.Context {
	return context.WithValue(ctx, performanceObserverKey{}, begin)
}

type responseMetrics struct {
	// In the pinned vLLM this is scheduling-to-first-token, excluding queue.
	TimeToFirstTokenMS *float64 `json:"time_to_first_token_ms"`
	GenerationTimeMS   *float64 `json:"generation_time_ms"`
	QueueTimeMS        *float64 `json:"queue_time_ms"`
}

type responseTimings struct {
	PromptN     int     `json:"prompt_n"`
	PromptMS    float64 `json:"prompt_ms"`
	PredictedN  int     `json:"predicted_n"`
	PredictedMS float64 `json:"predicted_ms"`
}

type streamPerformance struct {
	started, first, last, published time.Time
	estimated, firstEstimated       float64
	usage                           Usage
	metrics                         *responseMetrics
	timings                         *responseTimings
	report                          func(performance.Sample)
}

func newStreamPerformance(ctx context.Context) *streamPerformance {
	p := &streamPerformance{started: time.Now()}
	if begin, ok := ctx.Value(performanceObserverKey{}).(func() func(performance.Sample)); ok {
		p.report = begin()
	}
	return p
}

func estimateOutput(text string) float64 {
	var total float64
	for _, r := range text {
		if r < 128 {
			total += .25
		} else {
			total++
		}
	}
	return total
}

func (p *streamPerformance) generated(text string, now time.Time) {
	if text == "" {
		return
	}
	// Count text approximately, never SSE events: DSpark can emit many tokens
	// in one event. Final server usage replaces this estimate when available.
	p.estimated += estimateOutput(text)
	if p.first.IsZero() {
		p.first = now
		p.firstEstimated = p.estimated
	}
	p.last = now
}

func finitePositive(v float64) bool { return v > 0 && !math.IsNaN(v) && !math.IsInf(v, 0) }

func (p *streamPerformance) sample() performance.Sample {
	s := performance.Sample{PromptTokens: p.usage.PromptTokens,
		OutputTokens: p.usage.CompletionTokens, PPEstimated: true, TGEstimated: true, TTFTEstimated: true}
	if s.OutputTokens == 0 {
		s.OutputTokens = int(math.Ceil(p.estimated))
	}
	cacheKnown := p.usage.PromptTokensDetails != nil && p.usage.PromptTokensDetails.CachedTokens != nil
	if cacheKnown {
		s.CachedTokens = max(0, min(s.PromptTokens, *p.usage.PromptTokensDetails.CachedTokens))
	}
	if !p.first.IsZero() {
		seconds := p.first.Sub(p.started).Seconds()
		s.TTFT = &seconds
		s.PrefillSeconds = seconds
		s.PrefillTokens = float64(max(0, s.PromptTokens-s.CachedTokens))
		s.DecodeSeconds = p.last.Sub(p.first).Seconds()
		s.DecodeTokens = math.Max(0, float64(s.OutputTokens)-math.Max(1, p.firstEstimated))
	}
	if m := p.metrics; m != nil {
		if m.TimeToFirstTokenMS != nil && finitePositive(*m.TimeToFirstTokenMS) {
			s.PrefillSeconds = *m.TimeToFirstTokenMS / 1000
			s.PrefillTokens = float64(max(0, s.PromptTokens-s.CachedTokens))
			s.PPEstimated = !cacheKnown
			if m.QueueTimeMS != nil && *m.QueueTimeMS >= 0 && !math.IsInf(*m.QueueTimeMS, 0) && !math.IsNaN(*m.QueueTimeMS) {
				seconds := (*m.TimeToFirstTokenMS + *m.QueueTimeMS) / 1000
				s.TTFT = &seconds
				s.TTFTEstimated = false
			}
		}
		if m.GenerationTimeMS != nil && *m.GenerationTimeMS >= 0 && !math.IsNaN(*m.GenerationTimeMS) && !math.IsInf(*m.GenerationTimeMS, 0) && p.usage.CompletionTokens > 0 {
			// vLLM's tokens_per_second includes prefill and must NOT be used
			// as tg. Its decode interval starts at the first generated token.
			s.DecodeSeconds = *m.GenerationTimeMS / 1000
			s.DecodeTokens = float64(max(0, p.usage.CompletionTokens-1))
			s.TGEstimated = false
		}
	}
	if t := p.timings; t != nil {
		if t.PromptN > 0 && finitePositive(t.PromptMS) {
			s.PrefillTokens, s.PrefillSeconds = float64(t.PromptN), t.PromptMS/1000
			s.PPEstimated = false
		}
		if t.PredictedN > 0 && finitePositive(t.PredictedMS) {
			s.DecodeTokens, s.DecodeSeconds = float64(t.PredictedN), t.PredictedMS/1000
			s.TGEstimated = false
		}
	}
	return s
}

func (p *streamPerformance) publish(force bool) {
	if p.report == nil {
		return
	}
	now := time.Now()
	if !force && !p.published.IsZero() && now.Sub(p.published) < 250*time.Millisecond {
		return
	}
	p.published = now
	p.report(p.sample())
}
