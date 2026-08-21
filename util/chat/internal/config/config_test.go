package config

import (
	"bytes"
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
	if !cfg.Context.Enabled || cfg.Context.CompactAtPercent != 80 || cfg.Context.OutputReserve != 8192 {
		t.Fatalf("generated context defaults are incomplete: %+v", cfg.Context)
	}
	if !cfg.ASR.Enabled || cfg.ASR.FFmpegEndpoint != "http://127.0.0.1:8698" || cfg.ASR.Endpoint != "http://127.0.0.1:8694" {
		t.Fatalf("generated ASR defaults are incomplete: %+v", cfg.ASR)
	}
	if !cfg.TTS.Enabled || cfg.TTS.Endpoint != "http://127.0.0.1:8692" || cfg.TTS.Voice != "Sohee" || cfg.TTS.Seed != -1 {
		t.Fatalf("generated TTS defaults are incomplete: %+v", cfg.TTS)
	}
	wantPresetNames := []string{"없음", "보좌관", "언니여동생", "오빠여동생"}
	if len(cfg.Model.SystemPromptPresets) != len(wantPresetNames) {
		t.Fatalf("generated prompt presets are incomplete: %+v", cfg.Model.SystemPromptPresets)
	}
	for i, name := range wantPresetNames {
		if cfg.Model.SystemPromptPresets[i].Name != name {
			t.Fatalf("generated prompt preset %d = %q, want %q", i, cfg.Model.SystemPromptPresets[i].Name, name)
		}
	}
	if cfg.Model.SystemPromptPreset != "" || cfg.Model.SystemPrompt != "" {
		t.Fatalf("a generated prompt preset must not be selected automatically: %+v", cfg.Model)
	}
	generatedYAML, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(generatedYAML, []byte("# SparkTalk 설정")) ||
		!bytes.Contains(generatedYAML, []byte("name: 보좌관")) {
		t.Fatalf("generated config did not preserve comments and inject presets:\n%s", generatedYAML)
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

func TestLoadExistingConfigDoesNotInjectDefaultPromptPresets(t *testing.T) {
	tests := []struct {
		name string
		yaml string
		want []PromptPreset
	}{
		{
			name: "explicit empty list",
			yaml: "system_prompt_presets: []\n",
		},
		{
			name: "custom list",
			yaml: "system_prompt_presets:\n    - name: 사용자 프리셋\n      prompt: 사용자 전용 지침\n",
			want: []PromptPreset{{Name: "사용자 프리셋", Prompt: "사용자 전용 지침"}},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "sparktalk.yaml")
			data := []byte("server:\n  listen_addr: 127.0.0.1:8585\n  database: test.db\nmodel:\n  endpoint: http://example.com:8000\n  " + test.yaml)
			if err := os.WriteFile(path, data, 0600); err != nil {
				t.Fatal(err)
			}
			cfg, generated, err := Load(path)
			if err != nil {
				t.Fatal(err)
			}
			if generated {
				t.Fatal("existing config was reported as generated")
			}
			if len(cfg.Model.SystemPromptPresets) != len(test.want) {
				t.Fatalf("existing presets were changed: got %+v, want %+v", cfg.Model.SystemPromptPresets, test.want)
			}
			for i := range test.want {
				if cfg.Model.SystemPromptPresets[i] != test.want[i] {
					t.Fatalf("existing preset %d was changed: got %+v, want %+v", i, cfg.Model.SystemPromptPresets[i], test.want[i])
				}
			}
		})
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
	if !cfg.Tools.MediaImportEnabled {
		t.Fatalf("old config did not enable URL media import: %+v", cfg.Tools)
	}
	if !cfg.Context.Enabled || cfg.Context.RecentTokens != 32768 {
		t.Fatalf("old config did not receive context defaults: %+v", cfg.Context)
	}
	if !cfg.ASR.Enabled || !cfg.ASR.FilterFillers || cfg.ASR.Model != "qwen3-asr" || cfg.ASR.Timeout != "30m" {
		t.Fatalf("old config did not receive ASR defaults: %+v", cfg.ASR)
	}
	if !cfg.TTS.Enabled || cfg.TTS.Voice != "Sohee" || cfg.TTS.Seed != -1 {
		t.Fatalf("old config did not receive TTS defaults: %+v", cfg.TTS)
	}
	if cfg.Appearance.AssistantAvatar != "preset:spark" || cfg.Appearance.UserAvatar != "preset:person-blue" {
		t.Fatalf("old config did not receive avatar defaults: %+v", cfg.Appearance)
	}
	if len(cfg.Model.SystemPromptPresets) != 0 {
		t.Fatalf("old config unexpectedly received embedded prompt presets: %+v", cfg.Model.SystemPromptPresets)
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
