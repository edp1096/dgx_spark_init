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
	if cfg.Appearance.AssistantAvatar != "preset:spark" || cfg.Appearance.UserAvatar != "preset:person-blue" {
		t.Fatalf("generated avatar defaults are incomplete: %+v", cfg.Appearance)
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
	if cfg.Appearance.AssistantAvatar != "preset:spark" || cfg.Appearance.UserAvatar != "preset:person-blue" {
		t.Fatalf("old config did not receive avatar defaults: %+v", cfg.Appearance)
	}
}

func TestNormalizeAvatarAcceptsPreset(t *testing.T) {
	cfg := Config{Appearance: AppearanceConfig{AssistantAvatar: "preset:saturn", UserAvatar: "invalid"}}
	cfg.Normalize()
	if cfg.Appearance.AssistantAvatar != "preset:saturn" || cfg.Appearance.UserAvatar != "preset:person-blue" {
		t.Fatalf("unexpected normalized avatars: %+v", cfg.Appearance)
	}
}

func TestNormalizeAvatarMigratesComputerPreset(t *testing.T) {
	cfg := Config{Appearance: AppearanceConfig{AssistantAvatar: "preset:computer"}}
	cfg.Normalize()
	if cfg.Appearance.AssistantAvatar != "preset:quantum-computer" {
		t.Fatalf("legacy computer preset was not migrated: %+v", cfg.Appearance)
	}
}
