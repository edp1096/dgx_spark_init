// Package performance describes inference-only measurements, shared by the
// streaming client, response storage and UI. Durations are in seconds.
package performance

import (
	"math"
	"sync"
)

type Summary struct {
	PP            *float64 `json:"pp,omitempty"`
	TG            *float64 `json:"tg,omitempty"`
	TTFT          *float64 `json:"ttft,omitempty"`
	PPEstimated   bool     `json:"pp_estimated,omitempty"`
	TGEstimated   bool     `json:"tg_estimated,omitempty"`
	TTFTEstimated bool     `json:"ttft_estimated,omitempty"`
	PromptTokens  int      `json:"prompt_tokens"`
	CachedTokens  int      `json:"cached_tokens"`
	OutputTokens  int      `json:"output_tokens"`
	Calls         int      `json:"calls"`
	Live          bool     `json:"live,omitempty"`
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
