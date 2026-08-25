package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestExistingConfigInheritsNewDefaultEngine(t *testing.T) {
	path := filepath.Join(t.TempDir(), "media.yaml")
	legacy := []byte(`listen: 127.0.0.1:9999
data_dir: data
engines:
  image:
    endpoint: http://example.invalid:8691
  speech:
    endpoint: http://example.invalid:8692
  recognition:
    endpoint: http://example.invalid:8694
recognition:
  model: Qwen/Qwen3-ASR-1.7B-hf
`)
	if err := os.WriteFile(path, legacy, 0o644); err != nil {
		t.Fatal(err)
	}
	cfg, created, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if created {
		t.Fatal("existing config was reported as newly created")
	}
	if got := cfg.Engines["video"].Endpoint; got != "http://127.0.0.1:8695" {
		t.Fatalf("video endpoint=%q", got)
	}
	if got := cfg.Engines["image"].Endpoint; got != "http://example.invalid:8691" {
		t.Fatalf("custom image endpoint=%q", got)
	}
	if got := cfg.Engines["prompt"].Endpoint; got != "http://127.0.0.1:8696" {
		t.Fatalf("prompt endpoint=%q", got)
	}
	if cfg.PromptEnhancement.Model != "huihui-gemma4-12b" || !cfg.PromptEnhancement.VisionEnabled || cfg.PromptEnhancement.MaxTokens != 600 {
		t.Fatalf("prompt enhancement defaults=%#v", cfg.PromptEnhancement)
	}
	if cfg.Recognition.Model != "Qwen/Qwen3-ASR-1.7B" {
		t.Fatalf("legacy ASR model was not migrated: %q", cfg.Recognition.Model)
	}
}
