package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestSamplingParametersAreRequestLocalAndExplicitValuesWin(t *testing.T) {
	var bodies []map[string]any
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		bodies = append(bodies, body)
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer backend.Close()
	client := New(backend.URL, "model", "")
	seed := int64(31337)
	p := SamplingParameters{TopP: 0.95, TopK: 20, MinP: 0, RepetitionPenalty: 1, Seed: &seed}
	ctx := WithSamplingParameters(context.Background(), p)
	seed = 7
	ctx = WithSamplingDefaults(ctx, SamplingParameters{TopP: 0.5, TopK: 1, RepetitionPenalty: 2})
	for _, c := range []context.Context{ctx, context.Background()} {
		if _, err := client.Stream(c, []Message{{Role: "user", Content: "test"}}, "model", "none", nil, func(string, string) error { return nil }); err != nil {
			t.Fatal(err)
		}
	}
	for key, want := range map[string]float64{"top_p": 0.95, "top_k": 20, "min_p": 0, "repetition_penalty": 1, "presence_penalty": 0, "frequency_penalty": 0, "seed": 31337} {
		if bodies[0][key] != want {
			t.Error("sampler field differs", key, bodies[0][key], want)
		}
		if _, present := bodies[1][key]; present {
			t.Error("document sampler leaked into chat", key)
		}
	}
	for _, bad := range []SamplingParameters{{TopP: math.NaN(), TopK: 20, RepetitionPenalty: 1}, {TopP: 1.1, RepetitionPenalty: 1}, {TopP: 0.95, TopK: -2, RepetitionPenalty: 1}, {TopP: 0.95, MinP: -1, RepetitionPenalty: 1}} {
		if _, err := client.Stream(WithSamplingParameters(context.Background(), bad), nil, "model", "none", nil, func(string, string) error { return nil }); err == nil {
			t.Error("invalid sampler accepted", bad)
		}
	}
	if len(bodies) != 2 {
		t.Fatal("invalid sampling request reached backend")
	}
}
