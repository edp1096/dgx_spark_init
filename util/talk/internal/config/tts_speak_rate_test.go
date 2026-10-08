package config

import (
	"math"
	"path/filepath"
	"testing"
)

func TestTTSSpeakRateDefaultsPersistsAndValidates(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sparktalk.yaml")
	cfg, _, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.TTS.SpeakRate != 1 {
		t.Fatalf("default speak rate = %v", cfg.TTS.SpeakRate)
	}
	cfg.TTS.SpeakRate = 0
	cfg.Normalize()
	if cfg.TTS.SpeakRate != 1 {
		t.Fatal("old configurations must default to 1.0")
	}
	for _, rate := range []float64{0.5, 1, 1.2, 1.3, 2} {
		cfg.TTS.SpeakRate = rate
		if err := Save(path, cfg); err != nil {
			t.Fatal(err)
		}
		reloaded, _, err := Load(path)
		if err != nil || reloaded.TTS.SpeakRate != rate || reloaded.Public().TTS.SpeakRate != rate {
			t.Fatalf("rate %v not preserved: %v / %v", rate, reloaded.TTS.SpeakRate, err)
		}
	}
	for _, rate := range []float64{-1, 0.4, 2.1, math.NaN(), math.Inf(1)} {
		cfg.TTS.SpeakRate = rate
		if cfg.Validate() == nil {
			t.Fatalf("accepted invalid speak rate %v", rate)
		}
	}
}
