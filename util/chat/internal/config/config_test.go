package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadCreatesEmbeddedDefaultAndSaveReloads(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sparktalk.yaml")
	cfg, generated, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if !generated {
		t.Fatal("expected default config to be generated")
	}
	if cfg.Server.ListenAddr == "" || cfg.Model.Endpoint == "" {
		t.Fatalf("generated config is incomplete: %+v", cfg)
	}
	cfg.Model.ReasoningEffort = "xhigh"
	cfg.Model.SystemPrompt = "항상 존댓말로 답변합니다."
	cfg.Model.SystemPromptPreset = "존댓말"
	cfg.Model.SystemPromptPresets = []PromptPreset{{Name: "존댓말", Prompt: "항상 존댓말로 답변합니다."}}
	if err := Save(path, cfg); err != nil {
		t.Fatal(err)
	}
	reloaded, generated, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if generated || reloaded.Model.ReasoningEffort != "xhigh" || reloaded.Model.SystemPrompt != "항상 존댓말로 답변합니다." ||
		reloaded.Model.SystemPromptPreset != "존댓말" || len(reloaded.Model.SystemPromptPresets) != 1 {
		t.Fatalf("saved config was not reloaded: %+v", reloaded)
	}
}

func TestLoadOldConfigDefaultsToolsToEnabled(t *testing.T) {
	path := filepath.Join(t.TempDir(), "old.yaml")
	data := []byte("server:\n  listen_addr: 127.0.0.1:8585\n  database: test.db\nmodel:\n  endpoint: http://example.com:8000\n")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	cfg, _, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.Tools.Enabled || cfg.Tools.MaxRounds != 3 || cfg.Tools.SearchResults != 5 {
		t.Fatalf("old config did not receive tool defaults: %+v", cfg.Tools)
	}
}
