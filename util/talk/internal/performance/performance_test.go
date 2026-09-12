package performance

import (
	"math"
	"testing"
)

func TestAggregateWeightsRequestsAndReplacesLiveEstimates(t *testing.T) {
	var a Accumulator
	first := a.Start()
	ttft := .5
	first(Sample{PromptTokens: 1000, CachedTokens: 800, OutputTokens: 10,
		PrefillTokens: 200, PrefillSeconds: .2, DecodeTokens: 9, DecodeSeconds: 1, TTFT: &ttft, TGEstimated: true})
	first(Sample{PromptTokens: 1000, CachedTokens: 800, OutputTokens: 21,
		PrefillTokens: 200, PrefillSeconds: .2, DecodeTokens: 20, DecodeSeconds: 1, TTFT: &ttft})
	// No timestamps from the time spent executing a tool enter this sample.
	second := a.Start()
	secondTTFT := 12.0
	second(Sample{PromptTokens: 400, OutputTokens: 61, PrefillTokens: 400, PrefillSeconds: .8,
		DecodeTokens: 60, DecodeSeconds: 3, TTFT: &secondTTFT})
	s := a.Summary(false)
	if s.Calls != 2 || s.PromptTokens != 1400 || s.OutputTokens != 82 || s.CachedTokens != 800 {
		t.Fatalf("updates counted as new requests: %+v", s)
	}
	if *s.PP != 600 || *s.TG != 20 || *s.TTFT != .5 || s.Live || s.TGEstimated {
		t.Fatalf("wrong weighted rates or first-request TTFT: %+v", s)
	}
}

func TestNoRateForSingleTokenOrInvalidIntervals(t *testing.T) {
	var a Accumulator
	ttft := 0.0
	a.Start()(Sample{OutputTokens: 1, TTFT: &ttft, DecodeTokens: 0, DecodeSeconds: 0,
		PrefillTokens: 100, PrefillSeconds: math.Inf(1)})
	s := a.Summary(false)
	if s.PP != nil || s.TG != nil || s.TTFT == nil || *s.TTFT != 0 {
		t.Fatalf("invalid rate or valid zero TTFT lost: %+v", s)
	}
}
