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
	if cfg.Model.ModelType != "qwen3.8" {
		t.Fatalf("generated model type = %q, want qwen3.8", cfg.Model.ModelType)
	}
	if cfg.Model.ThinkingBudget != 512 {
		t.Fatalf("generated thinking budget = %d, want 512", cfg.Model.ThinkingBudget)
	}
	if cfg.Appearance.AssistantAvatar != "preset:spark" || cfg.Appearance.UserAvatar != "preset:person-blue" {
		t.Fatalf("generated avatar defaults are incomplete: %+v", cfg.Appearance)
	}
	if cfg.Appearance.Theme != "system" {
		t.Fatalf("generated theme default is incomplete: %+v", cfg.Appearance)
	}
	if !cfg.Context.Enabled || cfg.Context.CompactAtPercent != 80 || cfg.Context.OutputReserve != 8192 {
		t.Fatalf("generated context defaults are incomplete: %+v", cfg.Context)
	}
	if !cfg.ASR.Enabled || cfg.ASR.FFmpegEndpoint != "http://127.0.0.1:8690" ||
		cfg.ASR.Endpoint != "http://127.0.0.1:8693" || cfg.ASR.Model != "nemotron-3.5-asr-streaming-0.6b" ||
		cfg.ASR.VoiceLanguage != "ko-KR" || cfg.ASR.MediaLanguage != "auto" {
		t.Fatalf("generated ASR defaults are incomplete: %+v", cfg.ASR)
	}
	if !cfg.TTS.Enabled || cfg.TTS.Endpoint != "http://127.0.0.1:8692" ||
		cfg.TTS.Model != "magpietts" || cfg.TTS.Language != "auto" || cfg.TTS.HanjaReading != "korean" ||
		cfg.TTS.Voice != "Sofia" || cfg.TTS.SampleRate != 22050 || !cfg.TTS.OmitParentheticals {
		t.Fatalf("generated TTS defaults are incomplete: %+v", cfg.TTS)
	}
	if cfg.Image.Enabled || cfg.Image.Endpoint != "http://127.0.0.1:8691" || cfg.Image.Model != "flux2-klein-4b-nvfp4" || cfg.Image.Mode != "basic" || cfg.Image.DefaultSize != "1024x1024" {
		t.Fatalf("generated image defaults are incomplete: %+v", cfg.Image)
	}
	if !cfg.Extra.CollectorEnabled || cfg.Extra.CollectorEndpoint != "http://127.0.0.1:8695" {
		t.Fatalf("generated collector endpoint is incomplete: %+v", cfg.Extra)
	}
	if cfg.Version != 2 || cfg.Runtime.Mode != "managed" || cfg.Runtime.Bundle != "flash-next" || cfg.Runtime.MemoryReserveGiB != 8 {
		t.Fatalf("generated runtime defaults are incomplete: %+v", cfg.Runtime)
	}
	if cfg.Memory.AlwaysMaxResults != 6 || cfg.Memory.AlwaysTokenBudget != 1024 || cfg.Memory.MaxResults != 5 || cfg.Memory.TokenBudget != 2048 {
		t.Fatalf("generated memory defaults are incomplete: %+v", cfg.Memory)
	}
	wantPresetNames := []string{"없음", "한글 전용", "보좌관", "언니여동생", "오빠여동생"}
	if len(cfg.Model.SystemPromptPresets) != len(wantPresetNames) {
		t.Fatalf("generated prompt presets are incomplete: %+v", cfg.Model.SystemPromptPresets)
	}
	for i, name := range wantPresetNames {
		if cfg.Model.SystemPromptPresets[i].Name != name {
			t.Fatalf("generated prompt preset %d = %q, want %q", i, cfg.Model.SystemPromptPresets[i].Name, name)
		}
	}
	wantHangulOnlyPrompt := `답변 생성 시 한자(漢字)를 절대 사용하지 마세요. 모든 한자 표현은 한글 표기(음독)로 번역하거나 한글 단어로 대체하여 출력하세요. 예: "漢字" 대신 "한자".`
	if cfg.Model.SystemPromptPresets[1].Prompt != wantHangulOnlyPrompt {
		t.Fatalf("한글 전용 prompt = %q, want %q", cfg.Model.SystemPromptPresets[1].Prompt, wantHangulOnlyPrompt)
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

func TestManagedDefaultBundleAndActiveModelCanDiffer(t *testing.T) {
	cfg := Config{Runtime: RuntimeConfig{Mode: "managed", Bundle: "gemma", ActiveBundle: "flash-next"}}
	cfg.Normalize()

	if cfg.Runtime.Bundle != "gemma" {
		t.Fatalf("active model changed startup default: %q", cfg.Runtime.Bundle)
	}
	if cfg.Runtime.ActiveBundle != "flash-next" {
		t.Fatalf("startup default changed active model: %q", cfg.Runtime.ActiveBundle)
	}
	if cfg.Model.DefaultModel != "qwen3.8-flash-next" || cfg.Model.ModelType != "qwen3.8" || cfg.Context.WindowTokens != 65536 {
		t.Fatalf("active model profile was not applied: %+v", cfg.Model)
	}
}

func TestManagedEXL3BundleAppliesItsModelProfile(t *testing.T) {
	cfg := Config{Runtime: RuntimeConfig{Mode: "managed", Bundle: "qwen27-exl3", ActiveBundle: "qwen27-exl3"}}
	cfg.Normalize()

	if cfg.Runtime.Bundle != "qwen27-exl3" || cfg.Runtime.ActiveBundle != "qwen27-exl3" {
		t.Fatalf("EXL3 bundle was rejected: %+v", cfg.Runtime)
	}
	if cfg.Model.DefaultModel != "Qwen3.8-27B-Uncensored-EXL3-4bpw" || cfg.Model.ModelType != "qwen3.8-exl3" || cfg.Context.WindowTokens != 262144 {
		t.Fatalf("EXL3 model profile was not applied: model=%+v context=%+v", cfg.Model, cfg.Context)
	}
}

func TestNormalizeConstrainsModelSpecificReasoning(t *testing.T) {
	for _, test := range []struct {
		modelType string
		effort    string
		want      string
	}{
		{"qwen3.8", "xhigh", "xhigh"},
		{"qwen3.8", "high", "medium"},
		{"qwen3.8", "on", "medium"},
		{"qwen3.8-exl3", "none", "none"},
		{"qwen3.8-exl3", "low", "on"},
		{"gemma4", "xhigh", "on"},
		{"gemma4", "none", "none"},
		{"glm5.3", "none", "off"},
		{"glm5.3", "low", "low"},
		{"glm5.3", "high", "high"},
		{"glm5.3", "max", "max"},
		{"glm5.3", "xhigh", "max"},
		{"generic", "0.75", "0.75"},
	} {
		if got := normalizeReasoningEffort(test.modelType, test.effort); got != test.want {
			t.Fatalf("normalizeReasoningEffort(%q, %q) = %q, want %q", test.modelType, test.effort, got, test.want)
		}
	}
}

func TestNormalizePreservesJapaneseHanjaReading(t *testing.T) {
	cfg := Config{TTS: TTSConfig{HanjaReading: " Japanese "}}
	cfg.Normalize()
	if cfg.TTS.HanjaReading != "japanese" {
		t.Fatalf("TTS Japanese Hanja reading = %q, want japanese", cfg.TTS.HanjaReading)
	}
}

func TestLoadPreservesDisabledParentheticalOmission(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sparktalk.yaml")
	data := []byte("tts:\n  enabled: true\n  omit_parentheticals: false\n")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	cfg, _, err := Load(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.TTS.OmitParentheticals {
		t.Fatal("explicitly disabled parenthetical omission was reset")
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
	if !cfg.ASR.Enabled || !cfg.ASR.FilterFillers || cfg.ASR.Model != "nemotron-3.5-asr-streaming-0.6b" ||
		cfg.ASR.VoiceLanguage != "ko-KR" || cfg.ASR.MediaLanguage != "auto" || cfg.ASR.Timeout != "30m" {
		t.Fatalf("old config did not receive ASR defaults: %+v", cfg.ASR)
	}
	if !cfg.TTS.Enabled || cfg.TTS.Voice != "Sofia" || cfg.TTS.HanjaReading != "korean" || cfg.TTS.SampleRate != 22050 || !cfg.TTS.OmitParentheticals {
		t.Fatalf("old config did not receive TTS defaults: %+v", cfg.TTS)
	}
	if cfg.Appearance.AssistantAvatar != "preset:spark" || cfg.Appearance.UserAvatar != "preset:person-blue" {
		t.Fatalf("old config did not receive avatar defaults: %+v", cfg.Appearance)
	}
	if cfg.Appearance.Theme != "system" {
		t.Fatalf("old config did not receive system theme default: %+v", cfg.Appearance)
	}
	if len(cfg.Model.SystemPromptPresets) != 0 {
		t.Fatalf("old config unexpectedly received embedded prompt presets: %+v", cfg.Model.SystemPromptPresets)
	}
}

func TestNormalizeMigratesSplitASRToSharedNemotron(t *testing.T) {
	cfg := Config{ASR: ASRConfig{
		Enabled:        true,
		VoiceEndpoint:  "http://127.0.0.1:8693",
		VoiceModel:     "nemotron-3.5-asr-streaming-0.6b",
		VoiceLanguage:  "ko-KR",
		MediaEndpoint:  "http://127.0.0.1:8694",
		MediaModel:     "qwen3-asr",
		MediaLanguage:  "auto",
		FFmpegEndpoint: "http://127.0.0.1:8690",
	}}
	cfg.Normalize()
	if cfg.ASR.Endpoint != "http://127.0.0.1:8693" || cfg.ASR.Model != "nemotron-3.5-asr-streaming-0.6b" {
		t.Fatalf("split ASR did not migrate to the voice engine: %+v", cfg.ASR)
	}
	if cfg.ASR.VoiceLanguage != "ko-KR" || cfg.ASR.MediaLanguage != "auto" {
		t.Fatalf("per-request ASR languages were not preserved: %+v", cfg.ASR)
	}
	if cfg.ASR.VoiceEndpoint != "" || cfg.ASR.MediaEndpoint != "" || cfg.ASR.VoiceModel != "" || cfg.ASR.MediaModel != "" {
		t.Fatalf("legacy split ASR fields were not cleared: %+v", cfg.ASR)
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

func TestNormalizeAppearanceTheme(t *testing.T) {
	for _, theme := range []string{"dark", "light", "system"} {
		cfg := Config{Appearance: AppearanceConfig{Theme: theme}}
		cfg.Normalize()
		if cfg.Appearance.Theme != theme {
			t.Fatalf("theme %q normalized to %q", theme, cfg.Appearance.Theme)
		}
	}
	cfg := Config{Appearance: AppearanceConfig{Theme: "invalid"}}
	cfg.Normalize()
	if cfg.Appearance.Theme != "system" {
		t.Fatalf("invalid theme normalized to %q", cfg.Appearance.Theme)
	}
}
