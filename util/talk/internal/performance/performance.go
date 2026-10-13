// Package performance describes inference measurements and whole-response timing,
// shared by the streaming client, response storage and UI. Durations are in seconds.
package performance

import (
	"math"
	"sync"
	"time"
)

type Summary struct {
	Totals         *Totals  `json:"totals,omitempty"`
	ElapsedSeconds *float64 `json:"elapsed_seconds,omitempty"`
	FinishedAt     string   `json:"finished_at,omitempty"`
	FinishStatus   string   `json:"finish_status,omitempty"`
	PP             *float64 `json:"pp,omitempty"`
	TG             *float64 `json:"tg,omitempty"`
	TTFT           *float64 `json:"ttft,omitempty"`
	PPEstimated    bool     `json:"pp_estimated,omitempty"`
	TGEstimated    bool     `json:"tg_estimated,omitempty"`
	TTFTEstimated  bool     `json:"ttft_estimated,omitempty"`
	PromptTokens   int      `json:"prompt_tokens"`
	CachedTokens   int      `json:"cached_tokens"`
	OutputTokens   int      `json:"output_tokens"`
	Calls          int      `json:"calls"`
	Live           bool     `json:"live,omitempty"`
}

// Keep the measured numerators and denominators for durable aggregation.
// A stored rate alone cannot reconstruct speculative decode or cached prefill.
type Totals struct {
	PrefillTokens  float64 `json:"prefill_tokens"`
	PrefillSeconds float64 `json:"prefill_seconds"`
	DecodeTokens   float64 `json:"decode_tokens"`
	DecodeSeconds  float64 `json:"decode_seconds"`
}

// Sample belongs to one model request. Updates replace the same sample, so
// usage chunks and speculative multi-token events never double-count tokens.
type Sample struct {
	PromptTokens, CachedTokens, OutputTokens int
	PrefillTokens, PrefillSeconds            float64
	DecodeTokens, DecodeSeconds              float64
	TTFT                                     *float64
	PPEstimated, TGEstimated, TTFTEstimated  bool
}

type Accumulator struct {
	mu      sync.Mutex
	samples []Sample
}

func (a *Accumulator) Start() func(Sample) {
	a.mu.Lock()
	index := len(a.samples)
	a.samples = append(a.samples, Sample{})
	a.mu.Unlock()
	return func(s Sample) {
		a.mu.Lock()
		a.samples[index] = s
		a.mu.Unlock()
	}
}

func (a *Accumulator) Summary(live bool) *Summary {
	a.mu.Lock()
	defer a.mu.Unlock()
	if len(a.samples) == 0 {
		return nil
	}
	result := &Summary{Calls: len(a.samples), Live: live}
	var ppTokens, ppTime, tgTokens, tgTime float64
	for _, s := range a.samples {
		result.PromptTokens += max(0, s.PromptTokens)
		result.CachedTokens += max(0, s.CachedTokens)
		result.OutputTokens += max(0, s.OutputTokens)
		if result.TTFT == nil && s.TTFT != nil && valid(*s.TTFT) {
			value := *s.TTFT
			result.TTFT = &value
			result.TTFTEstimated = s.TTFTEstimated
		}
		if s.PrefillTokens > 0 && s.PrefillSeconds > 0 && valid(s.PrefillSeconds) {
			ppTokens += s.PrefillTokens
			ppTime += s.PrefillSeconds
			result.PPEstimated = result.PPEstimated || s.PPEstimated
		}
		if s.DecodeTokens > 0 && s.DecodeSeconds > 0 && valid(s.DecodeSeconds) {
			tgTokens += s.DecodeTokens
			tgTime += s.DecodeSeconds
			result.TGEstimated = result.TGEstimated || s.TGEstimated
		}
	}
	result.PP = rate(ppTokens, ppTime)
	result.TG = rate(tgTokens, tgTime)
	result.Totals = &Totals{ppTokens, ppTime, tgTokens, tgTime}
	return result
}

// Combine only summaries that retain measured totals. Older records remain
// readable, but their rates must not be averaged or turned into invented times.
func Combine(summaries []*Summary) *Summary {
	var a Accumulator
	calls := 0
	for _, s := range summaries {
		if s == nil || s.Totals == nil {
			continue
		}
		t := s.Totals
		calls += max(1, s.Calls)
		a.Start()(Sample{PromptTokens: s.PromptTokens, CachedTokens: s.CachedTokens, OutputTokens: s.OutputTokens,
			PrefillTokens: t.PrefillTokens, PrefillSeconds: t.PrefillSeconds,
			DecodeTokens: t.DecodeTokens, DecodeSeconds: t.DecodeSeconds,
			TTFT: s.TTFT, PPEstimated: s.PPEstimated, TGEstimated: s.TGEstimated, TTFTEstimated: s.TTFTEstimated})
	}
	result := a.Summary(false)
	if result != nil {
		result.Calls = calls
	}
	return result
}

func valid(v float64) bool { return v >= 0 && !math.IsNaN(v) && !math.IsInf(v, 0) }

func rate(tokens, seconds float64) *float64 {
	if tokens <= 0 || seconds <= 0 {
		return nil
	}
	v := tokens / seconds
	if !valid(v) {
		return nil
	}
	return &v
}

// Optional keeps older callers and stored response variants compatible.
func Optional(values []*Summary) *Summary {
	if len(values) == 0 {
		return nil
	}
	return values[0]
}

// Finish includes request preparation, tools and waiting. Elapsed time uses the
// monotonic clock; the wall clock timestamp is only used for display.
func Finish(summary *Summary, started time.Time, status string) *Summary {
	result := Summary{}
	if summary != nil {
		result = *summary
	}
	now := time.Now()
	elapsed := max(0, now.Sub(started).Seconds())
	result.ElapsedSeconds = &elapsed
	result.FinishedAt = now.UTC().Format(time.RFC3339Nano)
	result.FinishStatus = status
	result.Live = false
	return &result
}
