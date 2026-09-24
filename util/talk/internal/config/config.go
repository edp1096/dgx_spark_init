package config

import (
	"embed"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
	"sparktalk/internal/orchestrator"
)

const DefaultPath = "sparktalk.yaml"

//go:embed assets/sparktalk.default.yaml assets/system_prompt_presets.default.yaml assets/prompt_composer.defaults.json
var assets embed.FS

type Config struct {
	Version    int              `yaml:"version" json:"version"`
	Server     ServerConfig     `yaml:"server" json:"server"`
	Runtime    RuntimeConfig    `yaml:"runtime" json:"runtime"`
	Model      ModelConfig      `yaml:"model" json:"model"`
	ASR        ASRConfig        `yaml:"asr" json:"asr"`
	TTS        TTSConfig        `yaml:"tts" json:"tts"`
	Context    ContextConfig    `yaml:"context" json:"context"`
	Memory     MemoryConfig     `yaml:"memory" json:"memory"`
	Tools      ToolsConfig      `yaml:"tools" json:"tools"`
	Image      ImageConfig      `yaml:"image" json:"image"`
	Extra      ExtraConfig      `yaml:"extra" json:"extra"`
	Appearance AppearanceConfig `yaml:"appearance" json:"appearance"`
}

// RuntimeConfig stores editable runtime sets and their service connections.
type RuntimeConfig struct {
	BuiltinRevision  int                          `yaml:"builtin_revision,omitempty" json:"builtin_revision,omitempty"`
	KeyStorePeers    map[string]orchestrator.Host `yaml:"key_store_peers,omitempty" json:"key_store_peers,omitempty"`
	KeyStoreHosts    []string                     `yaml:"key_store_hosts,omitempty" json:"key_store_hosts,omitempty"`
	Catalog          *orchestrator.Catalog        `yaml:"catalog,omitempty" json:"catalog,omitempty"`
	Mode             string                       `yaml:"mode" json:"mode"`
	Bundle           string                       `yaml:"bundle" json:"bundle"`
	ActiveBundle     string                       `yaml:"active_bundle,omitempty" json:"-"`
	AutoStart        bool                         `yaml:"auto_start" json:"auto_start"`
	DataDir          string                       `yaml:"data_dir" json:"data_dir"`
	ModelCache       string                       `yaml:"model_cache" json:"model_cache"`
	MemoryReserveGiB float64                      `yaml:"memory_reserve_gib" json:"memory_reserve_gib"`
}

// ASRConfig connects local media preparation and speech recognition services.
// Audio attachments become text; video attachments keep their visual stream and
// gain a transcript of their audio track.
type ASRConfig struct {
	Diarization    bool   `yaml:"diarization" json:"diarization"`
	Enabled        bool   `yaml:"enabled" json:"enabled"`
	FFmpegEndpoint string `yaml:"ffmpeg_endpoint" json:"ffmpeg_endpoint"`
	Endpoint       string `yaml:"endpoint" json:"endpoint"`
	Model          string `yaml:"model" json:"model"`
	VoiceLanguage  string `yaml:"voice_language" json:"voice_language"`
	MediaLanguage  string `yaml:"media_language" json:"media_language"`
	// These fields read former ASR layouts for one-time migration.
	VoiceEndpoint string `yaml:"voice_endpoint,omitempty" json:"-"`
	VoiceModel    string `yaml:"voice_model,omitempty" json:"-"`
	MediaEndpoint string `yaml:"media_endpoint,omitempty" json:"-"`
	MediaModel    string `yaml:"media_model,omitempty" json:"-"`
	Language      string `yaml:"language,omitempty" json:"-"`
	Prompt        string `yaml:"prompt" json:"prompt"`
	FilterFillers bool   `yaml:"filter_fillers" json:"filter_fillers"`
	Timeout       string `yaml:"timeout" json:"timeout"`
}

// TTSConfig connects the assistant reply reader to the Magpie TTS service.
type TTSConfig struct {
	Enabled            bool   `yaml:"enabled" json:"enabled"`
	Endpoint           string `yaml:"endpoint" json:"endpoint"`
	Model              string `yaml:"model" json:"model"`
	Language           string `yaml:"language" json:"language"`
	HanjaReading       string `yaml:"hanja_reading" json:"hanja_reading"`
	Voice              string `yaml:"voice" json:"voice"`
	SampleRate         int    `yaml:"sample_rate" json:"sample_rate"`
	AutoPlay           bool   `yaml:"auto_play" json:"auto_play"`
	OmitParentheticals bool   `yaml:"omit_parentheticals" json:"omit_parentheticals"`
	Timeout            string `yaml:"timeout" json:"timeout"`
}

type ServerConfig struct {
	ListenAddr string `yaml:"listen_addr" json:"listen_addr"`
	Database   string `yaml:"database" json:"database"`
}

type ModelConfig struct {
	VideoInputs         map[string]string `yaml:"video_inputs,omitempty" json:"video_inputs,omitempty"`
	Endpoint            string            `yaml:"endpoint" json:"endpoint"`
	DefaultModel        string            `yaml:"default_model" json:"default_model"`
	ModelType           string            `yaml:"model_type" json:"model_type"`
	APIKey              string            `yaml:"api_key" json:"-"`
	ReasoningEffort     string            `yaml:"reasoning_effort" json:"reasoning_effort"`
	ThinkingBudget      int               `yaml:"thinking_budget" json:"thinking_budget"`
	SystemPrompt        string            `yaml:"system_prompt" json:"system_prompt"`
	SystemPromptPreset  string            `yaml:"system_prompt_preset,omitempty" json:"system_prompt_preset"`
	SystemPromptPresets []PromptPreset    `yaml:"system_prompt_presets,omitempty" json:"system_prompt_presets"`
	PromptComposer      *PromptComposer   `yaml:"prompt_composer,omitempty" json:"prompt_composer"`
}

type PromptPreset struct {
	Name   string `yaml:"name" json:"name"`
	Prompt string `yaml:"prompt" json:"prompt"`
}

// ContextConfig controls the model-facing working set. The complete transcript
// always remains in SQLite and in the browser; only the payload sent to the
// model is compacted.
type ContextConfig struct {
	Enabled          bool `yaml:"enabled" json:"enabled"`
	WindowTokens     int  `yaml:"window_tokens" json:"window_tokens"`
	CompactAtPercent int  `yaml:"compact_at_percent" json:"compact_at_percent"`
	OutputReserve    int  `yaml:"output_reserve" json:"output_reserve"`
	OutputAuto       bool `yaml:"output_auto" json:"output_auto"`
	SafetyMargin     int  `yaml:"safety_margin" json:"safety_margin"`
	RecentTokens     int  `yaml:"recent_tokens" json:"recent_tokens"`
	ImageTokens      int  `yaml:"image_tokens" json:"image_tokens"`
}

// EffectiveOutputReserve chooses an output budget from the current context size.
// An unknown context keeps the user's saved manual budget.
func (c ContextConfig) EffectiveOutputReserve(window int) int {
	if !c.OutputAuto || window <= 0 {
		return c.OutputReserve
	}
	for _, size := range []int{262144, 131072, 65536, 32768} {
		if window >= size {
			return min(size/4, max(1, (window-c.SafetyMargin)/2))
		}
	}
	return max(1, (window-c.SafetyMargin)/4)
}

// MemoryConfig controls bounded cross-session recall. The source transcript
// stays in SQLite; only a small relevant excerpt is added to model context.
type MemoryConfig struct {
	Enabled           bool `yaml:"enabled" json:"enabled"`
	RecallSessions    bool `yaml:"recall_sessions" json:"recall_sessions"`
	AllowProposals    bool `yaml:"allow_proposals" json:"allow_proposals"`
	AlwaysMaxResults  int  `yaml:"always_max_results" json:"always_max_results"`
	AlwaysTokenBudget int  `yaml:"always_token_budget" json:"always_token_budget"`
	MaxResults        int  `yaml:"max_results" json:"max_results"`
	TokenBudget       int  `yaml:"token_budget" json:"token_budget"`
}

type ToolsConfig struct {
	Enabled            bool   `yaml:"enabled" json:"enabled"`
	MediaImportEnabled bool   `yaml:"media_import_enabled" json:"media_import_enabled"`
	SkillsEnabled      bool   `yaml:"skills_enabled" json:"skills_enabled"`
	MaxRounds          int    `yaml:"max_rounds" json:"max_rounds"`
	SearchResults      int    `yaml:"search_results" json:"search_results"`
	Timeout            string `yaml:"timeout" json:"timeout"`
}

// ImageConfig connects SparkTalk's model tools to an OpenAI-compatible local
// image API. Basic mode exposes text-to-image arguments, reference mode adds
// single-image editing, and extended mode enables controls, LoRAs, and helpers.
type ImageConfig struct {
	Enabled     bool   `yaml:"enabled" json:"enabled"`
	Endpoint    string `yaml:"endpoint" json:"endpoint"`
	Model       string `yaml:"model" json:"model"`
	Mode        string `yaml:"mode" json:"mode"`
	DefaultSize string `yaml:"default_size" json:"default_size"`
	Timeout     string `yaml:"timeout" json:"timeout"`
}

type ExtraConfig struct {
	MediaEndpoint     string `yaml:"media_endpoint" json:"media_endpoint"`
	DocumentsEnabled  bool   `yaml:"documents_enabled" json:"documents_enabled"`
	DocumentsEndpoint string `yaml:"documents_endpoint" json:"documents_endpoint"`
	SSHEnabled        bool   `yaml:"ssh_enabled" json:"ssh_enabled"`
	SSHEndpoint       string `yaml:"ssh_endpoint" json:"ssh_endpoint"`
	CollectorEnabled  bool   `yaml:"collector_enabled" json:"collector_enabled"`
	CollectorEndpoint string `yaml:"collector_endpoint" json:"collector_endpoint"`
}

type AppearanceConfig struct {
	UserName        string `yaml:"user_name" json:"user_name"`
	AssistantAvatar string `yaml:"assistant_avatar" json:"assistant_avatar"`
	UserAvatar      string `yaml:"user_avatar" json:"user_avatar"`
	Theme           string `yaml:"theme" json:"theme"`
}

type PublicConfig struct {
	Version    int              `json:"version"`
	Server     ServerConfig     `json:"server"`
	Runtime    RuntimeConfig    `json:"runtime"`
	Model      ModelConfig      `json:"model"`
	ASR        ASRConfig        `json:"asr"`
	TTS        TTSConfig        `json:"tts"`
	Context    ContextConfig    `json:"context"`
	Memory     MemoryConfig     `json:"memory"`
	Tools      ToolsConfig      `json:"tools"`
	Image      ImageConfig      `json:"image"`
	Extra      ExtraConfig      `json:"extra"`
	Appearance AppearanceConfig `json:"appearance"`
	APIKeySet  bool             `json:"api_key_set"`
}

func Load(path string) (Config, bool, error) {
	if path == "" {
		path = DefaultPath
	}
	generated := false
	if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
		data, readErr := generatedDefaultConfig()
		if readErr != nil {
			return Config{}, false, readErr
		}
		if writeErr := os.WriteFile(path, data, 0600); writeErr != nil {
			return Config{}, false, fmt.Errorf("write default config: %w", writeErr)
		}
		generated = true
	} else if err != nil {
		return Config{}, false, err
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return Config{}, generated, err
	}
	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return Config{}, generated, fmt.Errorf("parse %s: %w", path, err)
	}
	var presence struct {
		ASR *struct {
			Enabled       *bool `yaml:"enabled"`
			FilterFillers *bool `yaml:"filter_fillers"`
		} `yaml:"asr"`
		TTS *struct {
			Enabled            *bool `yaml:"enabled"`
			OmitParentheticals *bool `yaml:"omit_parentheticals"`
		} `yaml:"tts"`
		Context *struct {
			Enabled *bool `yaml:"enabled"`
		} `yaml:"context"`
		Memory *struct {
			Enabled        *bool `yaml:"enabled"`
			RecallSessions *bool `yaml:"recall_sessions"`
			AllowProposals *bool `yaml:"allow_proposals"`
		} `yaml:"memory"`
		Tools *struct {
			Enabled            *bool `yaml:"enabled"`
			MediaImportEnabled *bool `yaml:"media_import_enabled"`
			SkillsEnabled      *bool `yaml:"skills_enabled"`
		} `yaml:"tools"`
		Image *struct {
			Enabled *bool `yaml:"enabled"`
		} `yaml:"image"`
		Extra *struct {
			CollectorEnabled *bool `yaml:"collector_enabled"`
		} `yaml:"extra"`
	}
	_ = yaml.Unmarshal(data, &presence)
	if presence.ASR == nil || presence.ASR.Enabled == nil {
		cfg.ASR.Enabled = true
	}
	if presence.ASR == nil || presence.ASR.FilterFillers == nil {
		cfg.ASR.FilterFillers = true
	}
	if presence.TTS == nil || presence.TTS.Enabled == nil {
		cfg.TTS.Enabled = true
	}
	if presence.TTS == nil || presence.TTS.OmitParentheticals == nil {
		cfg.TTS.OmitParentheticals = true
	}
	if presence.Context == nil || presence.Context.Enabled == nil {
		cfg.Context.Enabled = true
	}
	if presence.Memory == nil || presence.Memory.Enabled == nil {
		cfg.Memory.Enabled = true
	}
	if presence.Memory == nil || presence.Memory.RecallSessions == nil {
		cfg.Memory.RecallSessions = true
	}
	if presence.Memory == nil || presence.Memory.AllowProposals == nil {
		cfg.Memory.AllowProposals = true
	}
	if presence.Tools == nil || presence.Tools.Enabled == nil {
		cfg.Tools.Enabled = true
	}
	if presence.Tools == nil || presence.Tools.MediaImportEnabled == nil {
		cfg.Tools.MediaImportEnabled = true
	}
	if presence.Tools == nil || presence.Tools.SkillsEnabled == nil {
		cfg.Tools.SkillsEnabled = true
	}
	if presence.Image == nil || presence.Image.Enabled == nil {
		// Older configurations predate selectable image engines. Keep the tool
		// disabled until an endpoint/model pair is chosen explicitly.
		cfg.Image.Enabled = false
	}
	if presence.Extra == nil || presence.Extra.CollectorEnabled == nil {
		cfg.Extra.CollectorEnabled = true
	}
	cfg.Normalize()
	if err := cfg.Validate(); err != nil {
		return Config{}, generated, err
	}
	return cfg, generated, nil
}

func generatedDefaultConfig() ([]byte, error) {
	data, err := assets.ReadFile("assets/sparktalk.default.yaml")
	if err != nil {
		return nil, fmt.Errorf("read embedded default config: %w", err)
	}
	presetData, err := assets.ReadFile("assets/system_prompt_presets.default.yaml")
	if err != nil {
		return nil, fmt.Errorf("read embedded system prompt presets: %w", err)
	}

	var document yaml.Node
	if err := yaml.Unmarshal(data, &document); err != nil {
		return nil, fmt.Errorf("parse embedded default config: %w", err)
	}
	var presetDocument yaml.Node
	if err := yaml.Unmarshal(presetData, &presetDocument); err != nil {
		return nil, fmt.Errorf("parse embedded system prompt presets: %w", err)
	}
	if len(document.Content) != 1 || document.Content[0].Kind != yaml.MappingNode {
		return nil, errors.New("embedded default config must be a YAML mapping")
	}
	if len(presetDocument.Content) != 1 || presetDocument.Content[0].Kind != yaml.SequenceNode {
		return nil, errors.New("embedded system prompt presets must be a YAML sequence")
	}
	modelNode := mappingNodeValue(document.Content[0], "model")
	if modelNode == nil || modelNode.Kind != yaml.MappingNode {
		return nil, errors.New("embedded default config is missing model mapping")
	}
	presetNode := mappingNodeValue(modelNode, "system_prompt_presets")
	if presetNode == nil {
		return nil, errors.New("embedded default config is missing model.system_prompt_presets")
	}
	*presetNode = *presetDocument.Content[0]

	generated, err := yaml.Marshal(&document)
	if err != nil {
		return nil, fmt.Errorf("generate default config: %w", err)
	}
	return generated, nil
}

func mappingNodeValue(mapping *yaml.Node, key string) *yaml.Node {
	for i := 0; i+1 < len(mapping.Content); i += 2 {
		if mapping.Content[i].Value == key {
			return mapping.Content[i+1]
		}
	}
	return nil
}

func Save(path string, cfg Config) error {
	cfg.Normalize()
	return saveNormalized(path, cfg)
}

func saveNormalized(path string, cfg Config) error {
	if path == "" {
		path = DefaultPath
	}
	if err := cfg.Validate(); err != nil {
		return err
	}
	data, err := yaml.Marshal(cfg)
	if err != nil {
		return err
	}
	file, err := os.CreateTemp(filepath.Dir(path), ".sparktalk-*.tmp")
	if err != nil {
		return err
	}
	temp := file.Name()
	defer os.Remove(temp)
	if _, err := file.Write(data); err != nil {
		file.Close()
		return err
	}
	if err := file.Sync(); err != nil {
		file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	return os.Rename(temp, path)
}

func (c Config) Validate() error {
	if len(c.Appearance.UserName) > 240 {
		return errors.New("user name is too long")
	}
	if c.Runtime.Catalog != nil {
		catalog, err := orchestrator.ValidateCatalog(*c.Runtime.Catalog)
		if err != nil {
			return fmt.Errorf("runtime.catalog: %w", err)
		}
		if c.Runtime.Mode == "managed" {
			if _, ok := catalog.Bundle(c.Runtime.Bundle); !ok {
				return fmt.Errorf("unknown default bundle %q", c.Runtime.Bundle)
			}
			if _, ok := catalog.Bundle(c.Runtime.ActiveBundle); !ok {
				return fmt.Errorf("unknown active bundle %q", c.Runtime.ActiveBundle)
			}
		}
	}

	if c.Runtime.Mode != "managed" && c.Runtime.Mode != "external" {
		return errors.New("runtime.mode must be managed or external")
	}
	if len(c.Runtime.KeyStoreHosts) > 0 {
		catalog, err := orchestrator.LoadCatalog()
		if err != nil {
			return err
		}
		if c.Runtime.Catalog != nil {
			catalog = *c.Runtime.Catalog
		}
		if err := orchestrator.ValidateKeyStoreHosts(catalog, c.Runtime.KeyStoreHosts); err != nil {
			return err
		}
	}
	if c.Runtime.MemoryReserveGiB < 1 || c.Runtime.MemoryReserveGiB > 64 {
		return errors.New("runtime.memory_reserve_gib must be between 1 and 64")
	}
	if c.Model.Endpoint == "" {
		return errors.New("model.endpoint is required")
	}
	if !strings.HasPrefix(c.Model.Endpoint, "http://") && !strings.HasPrefix(c.Model.Endpoint, "https://") {
		return errors.New("model.endpoint must start with http:// or https://")
	}
	if c.ASR.Enabled {
		for name, endpoint := range map[string]string{
			"asr.ffmpeg_endpoint": c.ASR.FFmpegEndpoint,
			"asr.endpoint":        c.ASR.Endpoint,
		} {
			if !strings.HasPrefix(endpoint, "http://") && !strings.HasPrefix(endpoint, "https://") {
				return fmt.Errorf("%s must start with http:// or https://", name)
			}
		}
		if c.ASR.Model == "" {
			return errors.New("asr.model is required")
		}
	}
	if timeout, err := time.ParseDuration(c.ASR.Timeout); err != nil || timeout <= 0 {
		if err == nil {
			err = errors.New("must be greater than zero")
		}
		return fmt.Errorf("asr.timeout: %w", err)
	}
	if c.TTS.Enabled && !strings.HasPrefix(c.TTS.Endpoint, "http://") && !strings.HasPrefix(c.TTS.Endpoint, "https://") {
		return errors.New("tts.endpoint must start with http:// or https://")
	}
	if c.TTS.SampleRate < 8000 || c.TTS.SampleRate > 192000 {
		return errors.New("tts.sample_rate must be between 8000 and 192000")
	}
	if timeout, err := time.ParseDuration(c.TTS.Timeout); err != nil || timeout <= 0 {
		if err == nil {
			err = errors.New("must be greater than zero")
		}
		return fmt.Errorf("tts.timeout: %w", err)
	}
	if filepath.Clean(c.Server.Database) == "." {
		return errors.New("server.database is required")
	}
	if len(c.Model.SystemPromptPresets) > 200 {
		return errors.New("model.system_prompt_presets supports at most 200 presets")
	}
	if c.Model.PromptComposer != nil {
		if err := c.Model.PromptComposer.Validate(); err != nil {
			return err
		}
	}
	presetNames := make(map[string]struct{}, len(c.Model.SystemPromptPresets))
	for _, preset := range c.Model.SystemPromptPresets {
		if preset.Name == "" {
			return errors.New("system prompt preset name is required")
		}
		if _, exists := presetNames[preset.Name]; exists {
			return fmt.Errorf("duplicate system prompt preset: %s", preset.Name)
		}
		presetNames[preset.Name] = struct{}{}
	}
	if c.Model.SystemPromptPreset != "" {
		if _, exists := presetNames[c.Model.SystemPromptPreset]; !exists {
			return fmt.Errorf("system prompt preset not found: %s", c.Model.SystemPromptPreset)
		}
	}
	if timeout, err := time.ParseDuration(c.Tools.Timeout); err != nil || timeout <= 0 {
		if err == nil {
			err = errors.New("must be greater than zero")
		}
		return fmt.Errorf("tools.timeout: %w", err)
	}
	if c.Image.Enabled && !strings.HasPrefix(c.Image.Endpoint, "http://") && !strings.HasPrefix(c.Image.Endpoint, "https://") {
		return errors.New("image.endpoint must start with http:// or https://")
	}
	if c.Image.Enabled && c.Image.Model == "" {
		return errors.New("image.model is required")
	}
	if c.Image.Mode != "basic" && c.Image.Mode != "extended" && c.Image.Mode != "reference" && c.Image.Mode != "paint" {
		return errors.New("image.mode must be basic, reference, paint, or extended")
	}
	if !validImageSize(c.Image.DefaultSize) {
		return errors.New("image.default_size must be WIDTHxHEIGHT using 512..2048 multiples of 16")
	}
	if timeout, err := time.ParseDuration(c.Image.Timeout); err != nil || timeout <= 0 {
		if err == nil {
			err = errors.New("must be greater than zero")
		}
		return fmt.Errorf("image.timeout: %w", err)
	}
	if c.Tools.MediaImportEnabled && !strings.HasPrefix(c.Extra.MediaEndpoint, "http://") && !strings.HasPrefix(c.Extra.MediaEndpoint, "https://") {
		return errors.New("extra.media_endpoint must start with http:// or https://")
	}
	if c.Extra.SSHEnabled && !strings.HasPrefix(c.Extra.SSHEndpoint, "http://") && !strings.HasPrefix(c.Extra.SSHEndpoint, "https://") {
		return errors.New("extra.ssh_endpoint must start with http:// or https://")
	}
	if c.Extra.DocumentsEnabled && !strings.HasPrefix(c.Extra.DocumentsEndpoint, "http://") && !strings.HasPrefix(c.Extra.DocumentsEndpoint, "https://") {
		return fmt.Errorf("문서 서비스 API 주소는 http:// 또는 https://로 시작해야 합니다")
	}
	if c.Extra.CollectorEnabled && !strings.HasPrefix(c.Extra.CollectorEndpoint, "http://") && !strings.HasPrefix(c.Extra.CollectorEndpoint, "https://") {
		return errors.New("extra.collector_endpoint must start with http:// or https://")
	}
	if c.Context.WindowTokens < 0 {
		return errors.New("context.window_tokens must be zero (auto) or greater")
	}
	if c.Context.CompactAtPercent < 50 || c.Context.CompactAtPercent > 95 {
		return errors.New("context.compact_at_percent must be between 50 and 95")
	}
	if c.Context.OutputReserve < 256 || c.Context.SafetyMargin < 256 || c.Context.RecentTokens < 256 || c.Context.ImageTokens < 1 {
		return errors.New("context token budgets are too small")
	}
	if c.Memory.MaxResults < 1 || c.Memory.MaxResults > 12 {
		return errors.New("memory.max_results must be between 1 and 12")
	}
	if c.Memory.TokenBudget < 256 || c.Memory.TokenBudget > 8192 {
		return errors.New("memory.token_budget must be between 256 and 8192")
	}
	if c.Memory.AlwaysMaxResults < 1 || c.Memory.AlwaysMaxResults > 12 {
		return errors.New("memory.always_max_results must be between 1 and 12")
	}
	if c.Memory.AlwaysTokenBudget < 256 || c.Memory.AlwaysTokenBudget > 8192 {
		return errors.New("memory.always_token_budget must be between 256 and 8192")
	}
	return nil
}

func (c Config) Public() PublicConfig {
	public := PublicConfig{Version: c.Version, Server: c.Server, Runtime: c.Runtime, Model: c.Model, ASR: c.ASR, TTS: c.TTS, Context: c.Context, Memory: c.Memory, Tools: c.Tools, Image: c.Image, Extra: c.Extra, Appearance: c.Appearance, APIKeySet: c.Model.APIKey != ""}
	public.Model.APIKey = ""
	return public
}

func normalizeReasoningEffort(modelType, value string) string {
	raw := strings.TrimSpace(value)
	value = strings.ToLower(raw)
	switch modelType {
	case "glm5.3", "deepseek-v4":
		switch value {
		case "", "none", "off", "false", "disabled", "0", "0.0":
			return "off"
		case "low", "high", "max":
			return value
		default:
			return "max"
		}
	case "qwen3.8":
		switch value {
		case "none", "low", "medium", "xhigh":
			return value
		default:
			return "medium"
		}
	case "qwen3.8-gguf", "qwen3.8-exl3":
		switch value {
		case "", "0", "0.0", "none", "off", "false", "no_think", "disabled":
			return "none"
		case "low", "medium", "xhigh":
			return value
		default:
			return "xhigh" // Legacy on used the template's xhigh default.
		}
	case "gemma4", "gemma4-vllm", "qwen3.5":
		switch value {
		case "", "0", "0.0", "none", "off", "false", "no_think", "disabled":
			return "none"
		default:
			return "on"
		}
	default:
		return raw
	}
}

func validImageSize(value string) bool {
	var width, height int
	if _, err := fmt.Sscanf(value, "%dx%d", &width, &height); err != nil {
		return false
	}
	return width >= 512 && width <= 2048 && height >= 512 && height <= 2048 && width%16 == 0 && height%16 == 0 && value == fmt.Sprintf("%dx%d", width, height)
}

func normalizeASRLocale(value string) string {
	value = strings.TrimSpace(value)
	switch strings.ToLower(value) {
	case "korean", "ko", "ko-kr":
		return "ko-KR"
	case "japanese", "ja", "ja-jp":
		return "ja-JP"
	case "english", "en", "en-us":
		return "en-US"
	case "chinese", "mandarin", "zh", "zh-cn":
		return "zh-CN"
	case "auto":
		return "auto"
	default:
		return value
	}
}

func normalizeAvatar(value, fallback string) string {
	value = strings.TrimSpace(value)
	if value == "preset:computer" {
		return "preset:quantum-computer"
	}
	presets := map[string]struct{}{
		"preset:spark": {}, "preset:orbit": {}, "preset:earth": {}, "preset:saturn": {},
		"preset:robot":            {},
		"preset:quantum-computer": {}, "preset:person-blue": {}, "preset:person-warm": {},
		"preset:cat": {}, "preset:dog": {}, "preset:bear": {}, "preset:rabbit": {},
	}
	if _, ok := presets[value]; ok {
		return value
	}
	if id := strings.TrimPrefix(value, "/api/images/"); id != value && len(id) == 32 && isLowerHex(id) {
		return value
	}
	return fallback
}

func isLowerHex(value string) bool {
	for _, character := range value {
		if (character < '0' || character > '9') && (character < 'a' || character > 'f') {
			return false
		}
	}
	return true
}

// Video transport is deployment-specific: same model name on another server
// must not inherit an override from this deployment.
func (m ModelConfig) VideoInputMode() string {
	key := strings.TrimRight(strings.TrimSpace(m.Endpoint), "/") + "\n" + m.DefaultModel
	if mode := m.VideoInputs[key]; mode == "frames" || mode == "native" {
		return mode
	}
	if m.ModelType == "deepseek-v4" {
		return "frames"
	}
	return "native"
}
