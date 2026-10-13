package performance

import (
	"encoding/json"
	"math"
	"testing"
	"time"
)

func TestStoredTotalsCombineByMeasuredTimeNotMeanRate(t *testing.T) {
	var a, b Accumulator
	a.Start()(Sample{PromptTokens: 1000, CachedTokens: 900, OutputTokens: 21,
		PrefillTokens: 100, PrefillSeconds: 1, DecodeTokens: 20, DecodeSeconds: 2})
	b.Start()(Sample{PromptTokens: 300, OutputTokens: 11,
		PrefillTokens: 300, PrefillSeconds: 2, DecodeTokens: 10, DecodeSeconds: 2, PPEstimated: true})
	raw, err := json.Marshal([]*Summary{a.Summary(false), b.Summary(false)})
	if err != nil {
		t.Fatal(err)
	}
	var saved []*Summary
	if err = json.Unmarshal(raw, &saved); err != nil {
		t.Fatal(err)
	}
	legacyRate := 999.0
	legacy := &Summary{PP: &legacyRate, TG: &legacyRate, PromptTokens: 9000}
	saved = append(saved, legacy)
	result := Combine(saved)
	if result.Calls != 2 || result.PromptTokens != 1300 || result.CachedTokens != 900 ||
		*result.PP != 400.0/3 || *result.TG != 7.5 || !result.PPEstimated || result.TGEstimated {
		t.Fatalf("wrong durable weighted rates: %+v", result)
	}
	if Combine([]*Summary{nil, legacy}) != nil {
		t.Fatal("invented totals for legacy rates")
	}
	if nested := Combine([]*Summary{result}); nested.Calls != 2 || *nested.PP != *result.PP {
		t.Fatal("aggregation lost request count", nested)
	}
}

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

func TestFinishMeasuresWholeRequestWithoutChangingInference(t *testing.T) {
	started := time.Now().Add(-83 * time.Second)
	tg := 20.0
	source := &Summary{TG: &tg, Calls: 2, Live: true}
	result := Finish(source, started, "completed")
	if result.ElapsedSeconds == nil || *result.ElapsedSeconds < 83 || result.Live || result.FinishStatus != "completed" || *result.TG != 20 || result.Calls != 2 {
		t.Fatalf("unexpected timing: %+v", result)
	}
	if _, err := time.Parse(time.RFC3339Nano, result.FinishedAt); err != nil {
		t.Fatal(err)
	}
	if source.FinishedAt != "" || !source.Live {
		t.Fatal("mutated previously emitted summary")
	}
	if Finish(nil, started, "failed").ElapsedSeconds == nil {
		t.Fatal("missing timing before first model call")
	}
}
