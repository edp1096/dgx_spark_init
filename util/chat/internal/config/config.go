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

//go:embed assets/sparktalk.default.yaml
var assets embed.FS

type Config struct {
	Server     ServerConfig     `yaml:"server" json:"server"`
	Model      ModelConfig      `yaml:"model" json:"model"`
	Tools      ToolsConfig      `yaml:"tools" json:"tools"`
	Appearance AppearanceConfig `yaml:"appearance" json:"appearance"`
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

type ToolsConfig struct {
	Enabled       bool   `yaml:"enabled" json:"enabled"`
	MaxRounds     int    `yaml:"max_rounds" json:"max_rounds"`
	SearchResults int    `yaml:"search_results" json:"search_results"`
	Timeout       string `yaml:"timeout" json:"timeout"`
}

type AppearanceConfig struct {
	AssistantAvatar string `yaml:"assistant_avatar" json:"assistant_avatar"`
	UserAvatar      string `yaml:"user_avatar" json:"user_avatar"`
}

type PublicConfig struct {
	Server     ServerConfig     `json:"server"`
	Model      ModelConfig      `json:"model"`
	Tools      ToolsConfig      `json:"tools"`
	Appearance AppearanceConfig `json:"appearance"`
	APIKeySet  bool             `json:"api_key_set"`
}

func Load(path string) (Config, bool, error) {
	if path == "" {
		path = DefaultPath
	}
	generated := false
	if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
		data, readErr := assets.ReadFile("assets/sparktalk.default.yaml")
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
		Tools *struct {
			Enabled *bool `yaml:"enabled"`
		} `yaml:"tools"`
	}
	_ = yaml.Unmarshal(data, &presence)
	if presence.Tools == nil || presence.Tools.Enabled == nil {
		cfg.Tools.Enabled = true
	}
	cfg.Normalize()
	if err := cfg.Validate(); err != nil {
		return Config{}, generated, err
	}
	return cfg, generated, nil
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
	return nil
}

func (c Config) Public() PublicConfig {
	public := PublicConfig{Server: c.Server, Model: c.Model, Tools: c.Tools, Appearance: c.Appearance, APIKeySet: c.Model.APIKey != ""}
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
