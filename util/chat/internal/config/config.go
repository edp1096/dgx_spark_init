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
)

const DefaultPath = "sparktalk.yaml"

//go:embed assets/sparktalk.default.yaml assets/system_prompt_presets.default.yaml
var assets embed.FS

type Config struct {
	Server     ServerConfig     `yaml:"server" json:"server"`
	Model      ModelConfig      `yaml:"model" json:"model"`
	ASR        ASRConfig        `yaml:"asr" json:"asr"`
	TTS        TTSConfig        `yaml:"tts" json:"tts"`
	Context    ContextConfig    `yaml:"context" json:"context"`
	Tools      ToolsConfig      `yaml:"tools" json:"tools"`
	Extra      ExtraConfig      `yaml:"extra" json:"extra"`
	Appearance AppearanceConfig `yaml:"appearance" json:"appearance"`
}

// ASRConfig connects local media preparation and speech recognition services.
// Audio attachments become text; video attachments keep their visual stream and
// gain a transcript of their audio track.
type ASRConfig struct {
	Enabled        bool   `yaml:"enabled" json:"enabled"`
	FFmpegEndpoint string `yaml:"ffmpeg_endpoint" json:"ffmpeg_endpoint"`
	Endpoint       string `yaml:"endpoint" json:"endpoint"`
	Model          string `yaml:"model" json:"model"`
	Language       string `yaml:"language" json:"language"`
	Prompt         string `yaml:"prompt" json:"prompt"`
	FilterFillers  bool   `yaml:"filter_fillers" json:"filter_fillers"`
	Timeout        string `yaml:"timeout" json:"timeout"`
}

// TTSConfig connects the assistant reply reader to an OpenAI-compatible
// Qwen3-TTS CustomVoice service.
type TTSConfig struct {
	Enabled      bool   `yaml:"enabled" json:"enabled"`
	Endpoint     string `yaml:"endpoint" json:"endpoint"`
	Model        string `yaml:"model" json:"model"`
	Language     string `yaml:"language" json:"language"`
	Voice        string `yaml:"voice" json:"voice"`
	Instructions string `yaml:"instructions" json:"instructions"`
	Seed         int64  `yaml:"seed" json:"seed"`
	AutoPlay     bool   `yaml:"auto_play" json:"auto_play"`
	Timeout      string `yaml:"timeout" json:"timeout"`
}

type ServerConfig struct {
	ListenAddr string `yaml:"listen_addr" json:"listen_addr"`
	Database   string `yaml:"database" json:"database"`
}

type ModelConfig struct {
	Endpoint            string         `yaml:"endpoint" json:"endpoint"`
	DefaultModel        string         `yaml:"default_model" json:"default_model"`
	APIKey              string         `yaml:"api_key" json:"-"`
	ReasoningEffort     string         `yaml:"reasoning_effort" json:"reasoning_effort"`
	SystemPrompt        string         `yaml:"system_prompt" json:"system_prompt"`
	SystemPromptPreset  string         `yaml:"system_prompt_preset,omitempty" json:"system_prompt_preset"`
	SystemPromptPresets []PromptPreset `yaml:"system_prompt_presets,omitempty" json:"system_prompt_presets"`
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
	SafetyMargin     int  `yaml:"safety_margin" json:"safety_margin"`
	RecentTokens     int  `yaml:"recent_tokens" json:"recent_tokens"`
	ImageTokens      int  `yaml:"image_tokens" json:"image_tokens"`
}

type ToolsConfig struct {
	Enabled       bool   `yaml:"enabled" json:"enabled"`
	MaxRounds     int    `yaml:"max_rounds" json:"max_rounds"`
	SearchResults int    `yaml:"search_results" json:"search_results"`
	Timeout       string `yaml:"timeout" json:"timeout"`
}

type ExtraConfig struct {
	SSHEnabled  bool   `yaml:"ssh_enabled" json:"ssh_enabled"`
	SSHEndpoint string `yaml:"ssh_endpoint" json:"ssh_endpoint"`
}

type AppearanceConfig struct {
	AssistantAvatar string `yaml:"assistant_avatar" json:"assistant_avatar"`
	UserAvatar      string `yaml:"user_avatar" json:"user_avatar"`
}

type PublicConfig struct {
	Server     ServerConfig     `json:"server"`
	Model      ModelConfig      `json:"model"`
	ASR        ASRConfig        `json:"asr"`
	TTS        TTSConfig        `json:"tts"`
	Context    ContextConfig    `json:"context"`
	Tools      ToolsConfig      `json:"tools"`
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
			Enabled *bool  `yaml:"enabled"`
			Seed    *int64 `yaml:"seed"`
		} `yaml:"tts"`
		Context *struct {
			Enabled *bool `yaml:"enabled"`
		} `yaml:"context"`
		Tools *struct {
			Enabled *bool `yaml:"enabled"`
		} `yaml:"tools"`
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
	if presence.TTS == nil || presence.TTS.Seed == nil {
		cfg.TTS.Seed = -1
	}
	if presence.Context == nil || presence.Context.Enabled == nil {
		cfg.Context.Enabled = true
	}
	if presence.Tools == nil || presence.Tools.Enabled == nil {
		cfg.Tools.Enabled = true
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
	if path == "" {
		path = DefaultPath
	}
	cfg.Normalize()
	if err := cfg.Validate(); err != nil {
		return err
	}
	data, err := yaml.Marshal(cfg)
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0600)
}

func (c *Config) Normalize() {
	c.Server.ListenAddr = strings.TrimSpace(c.Server.ListenAddr)
	c.Server.Database = strings.TrimSpace(c.Server.Database)
	c.Model.Endpoint = strings.TrimRight(strings.TrimSpace(c.Model.Endpoint), "/")
	c.Model.DefaultModel = strings.TrimSpace(c.Model.DefaultModel)
	c.Model.ReasoningEffort = strings.TrimSpace(c.Model.ReasoningEffort)
	c.Model.SystemPrompt = strings.TrimSpace(c.Model.SystemPrompt)
	c.Model.SystemPromptPreset = strings.TrimSpace(c.Model.SystemPromptPreset)
	c.ASR.FFmpegEndpoint = strings.TrimRight(strings.TrimSpace(c.ASR.FFmpegEndpoint), "/")
	c.ASR.Endpoint = strings.TrimRight(strings.TrimSpace(c.ASR.Endpoint), "/")
	c.ASR.Model = strings.TrimSpace(c.ASR.Model)
	c.ASR.Language = strings.TrimSpace(c.ASR.Language)
	c.ASR.Prompt = strings.TrimSpace(c.ASR.Prompt)
	c.ASR.Timeout = strings.TrimSpace(c.ASR.Timeout)
	c.TTS.Endpoint = strings.TrimRight(strings.TrimSpace(c.TTS.Endpoint), "/")
	c.TTS.Model = strings.TrimSpace(c.TTS.Model)
	c.TTS.Language = strings.TrimSpace(c.TTS.Language)
	c.TTS.Voice = strings.TrimSpace(c.TTS.Voice)
	c.TTS.Instructions = strings.TrimSpace(c.TTS.Instructions)
	c.TTS.Timeout = strings.TrimSpace(c.TTS.Timeout)
	c.Extra.SSHEndpoint = strings.TrimRight(strings.TrimSpace(c.Extra.SSHEndpoint), "/")
	for i := range c.Model.SystemPromptPresets {
		c.Model.SystemPromptPresets[i].Name = strings.TrimSpace(c.Model.SystemPromptPresets[i].Name)
		c.Model.SystemPromptPresets[i].Prompt = strings.TrimSpace(c.Model.SystemPromptPresets[i].Prompt)
	}
	if c.Server.ListenAddr == "" {
		c.Server.ListenAddr = "127.0.0.1:8585"
	}
	if c.Server.Database == "" {
		c.Server.Database = "sparktalk.db"
	}
	if c.ASR.FFmpegEndpoint == "" {
		c.ASR.FFmpegEndpoint = "http://127.0.0.1:8698"
	}
	if c.ASR.Endpoint == "" {
		c.ASR.Endpoint = "http://127.0.0.1:8694"
	}
	if c.ASR.Model == "" {
		c.ASR.Model = "qwen3-asr"
	}
	if c.ASR.Language == "" {
		c.ASR.Language = "auto"
	}
	if c.ASR.Timeout == "" {
		c.ASR.Timeout = "30m"
	}
	if c.TTS.Endpoint == "" {
		c.TTS.Endpoint = "http://127.0.0.1:8692"
	}
	if c.TTS.Model == "" {
		c.TTS.Model = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
	}
	if c.TTS.Language == "" {
		c.TTS.Language = "Korean"
	}
	if c.TTS.Voice == "" {
		c.TTS.Voice = "Sohee"
	}
	if c.TTS.Timeout == "" {
		c.TTS.Timeout = "10m"
	}
	if c.Tools.MaxRounds <= 0 {
		c.Tools.MaxRounds = 3
	}
	if c.Tools.MaxRounds > 8 {
		c.Tools.MaxRounds = 8
	}
	if c.Tools.SearchResults <= 0 {
		c.Tools.SearchResults = 5
	}
	if c.Tools.SearchResults > 10 {
		c.Tools.SearchResults = 10
	}
	if c.Tools.Timeout == "" {
		c.Tools.Timeout = "15s"
	}
	if c.Extra.SSHEndpoint == "" {
		c.Extra.SSHEndpoint = "http://127.0.0.1:8699"
	}
	if c.Context.CompactAtPercent <= 0 {
		c.Context.CompactAtPercent = 80
	}
	if c.Context.OutputReserve <= 0 {
		c.Context.OutputReserve = 8192
	}
	if c.Context.SafetyMargin <= 0 {
		c.Context.SafetyMargin = 4096
	}
	if c.Context.RecentTokens <= 0 {
		c.Context.RecentTokens = 32768
	}
	if c.Context.ImageTokens <= 0 {
		c.Context.ImageTokens = 2048
	}
	c.Appearance.AssistantAvatar = normalizeAvatar(c.Appearance.AssistantAvatar, "preset:spark")
	c.Appearance.UserAvatar = normalizeAvatar(c.Appearance.UserAvatar, "preset:person-blue")
}

func (c Config) Validate() error {
	if c.Model.Endpoint == "" {
		return errors.New("model.endpoint is required")
	}
	if !strings.HasPrefix(c.Model.Endpoint, "http://") && !strings.HasPrefix(c.Model.Endpoint, "https://") {
		return errors.New("model.endpoint must start with http:// or https://")
	}
	if c.ASR.Enabled {
		for name, endpoint := range map[string]string{"asr.ffmpeg_endpoint": c.ASR.FFmpegEndpoint, "asr.endpoint": c.ASR.Endpoint} {
			if !strings.HasPrefix(endpoint, "http://") && !strings.HasPrefix(endpoint, "https://") {
				return fmt.Errorf("%s must start with http:// or https://", name)
			}
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
	if c.Extra.SSHEnabled && !strings.HasPrefix(c.Extra.SSHEndpoint, "http://") && !strings.HasPrefix(c.Extra.SSHEndpoint, "https://") {
		return errors.New("extra.ssh_endpoint must start with http:// or https://")
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
	return nil
}

func (c Config) Public() PublicConfig {
	public := PublicConfig{Server: c.Server, Model: c.Model, ASR: c.ASR, TTS: c.TTS, Context: c.Context, Tools: c.Tools, Extra: c.Extra, Appearance: c.Appearance, APIKeySet: c.Model.APIKey != ""}
	public.Model.APIKey = ""
	return public
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
