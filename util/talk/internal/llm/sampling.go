package llm

import (
	"context"
	"fmt"
	"math"
)

type temperatureKey struct{}

// WithTemperature changes only one request, never the user's chat defaults.
func WithTemperature(ctx context.Context, value float64) context.Context {
	return context.WithValue(ctx, temperatureKey{}, value)
}

func applyRequestTemperature(ctx context.Context, payload map[string]any) error {
	if value, ok := ctx.Value(temperatureKey{}).(float64); ok {
		if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 || value > 2 {
			return fmt.Errorf("temperature must be finite and between 0 and 2")
		}
		payload["temperature"] = value
	}
	return nil
}

type samplingParametersKey struct{}

// SamplingParameters records the request-local sampler, including neutral
// penalties, so comparisons do not silently inherit different engine defaults.
type SamplingParameters struct {
	TopP              float64 `json:"top_p"`
	TopK              int     `json:"top_k"`
	MinP              float64 `json:"min_p"`
	RepetitionPenalty float64 `json:"repetition_penalty"`
	PresencePenalty   float64 `json:"presence_penalty"`
	FrequencyPenalty  float64 `json:"frequency_penalty"`
	Seed              *int64  `json:"seed,omitempty"`
}

func (p SamplingParameters) Validate() error {
	for _, v := range []float64{p.TopP, p.MinP, p.RepetitionPenalty, p.PresencePenalty, p.FrequencyPenalty} {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("sampling values must be finite")
		}
	}
	if p.TopP <= 0 || p.TopP > 1 || p.TopK < 0 || p.MinP < 0 || p.MinP > 1 || p.RepetitionPenalty <= 0 || p.PresencePenalty < -2 || p.PresencePenalty > 2 || p.FrequencyPenalty < -2 || p.FrequencyPenalty > 2 {
		return fmt.Errorf("invalid request sampling parameters")
	}
	return nil
}

func WithSamplingParameters(ctx context.Context, p SamplingParameters) context.Context {
	if p.Seed != nil {
		seed := *p.Seed
		p.Seed = &seed
	}
	return context.WithValue(ctx, samplingParametersKey{}, p)
}
func WithSamplingDefaults(ctx context.Context, p SamplingParameters) context.Context {
	if _, set := ctx.Value(samplingParametersKey{}).(SamplingParameters); set {
		return ctx
	}
	return WithSamplingParameters(ctx, p)
}
func applyRequestSamplingParameters(ctx context.Context, payload map[string]any) error {
	p, ok := ctx.Value(samplingParametersKey{}).(SamplingParameters)
	if !ok {
		return nil
	}
	if err := p.Validate(); err != nil {
		return err
	}
	payload["top_p"], payload["top_k"], payload["min_p"] = p.TopP, p.TopK, p.MinP
	payload["repetition_penalty"], payload["presence_penalty"], payload["frequency_penalty"] = p.RepetitionPenalty, p.PresencePenalty, p.FrequencyPenalty
	if p.Seed != nil {
		payload["seed"] = *p.Seed
	}
	return nil
}
