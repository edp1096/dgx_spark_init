package config

import (
	_ "embed"
	"fmt"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

//go:embed default.yaml
var defaultYAML []byte

type Engine struct {
	Endpoint string `yaml:"endpoint" json:"endpoint"`
}

type ImageBackend struct {
	Endpoint string `yaml:"endpoint" json:"endpoint"`
	Model    string `yaml:"model" json:"model"`
}

type Image struct {
	Model                 string                  `yaml:"model" json:"model"`
	DefaultCheckpoint     string                  `yaml:"default_checkpoint" json:"default_checkpoint"`
	VisibleCheckpoints    []string                `yaml:"visible_checkpoints" json:"visible_checkpoints"`
	DefaultMode           string                  `yaml:"default_mode" json:"default_mode"`
	DefaultPromptEnhancer bool                    `yaml:"default_prompt_enhancer" json:"default_prompt_enhancer"`
	Backends              map[string]ImageBackend `yaml:"backends" json:"backends"`
	DefaultWidth          int                     `yaml:"default_width" json:"default_width"`
	DefaultHeight         int                     `yaml:"default_height" json:"default_height"`
	MaxReferenceImages    int                     `yaml:"max_reference_images" json:"max_reference_images"`
}

type Speech struct {
	CustomVoiceModel string `yaml:"custom_voice_model" json:"custom_voice_model"`
	DefaultLanguage  string `yaml:"default_language" json:"default_language"`
	DefaultSpeaker   string `yaml:"default_speaker" json:"default_speaker"`
}

type Recognition struct {
	Model                      string   `yaml:"model" json:"model"`
	DefaultLanguage            string   `yaml:"default_language" json:"default_language"`
	MaxUploadMB                int64    `yaml:"max_upload_mb" json:"max_upload_mb"`
	SegmentSeconds             int      `yaml:"segment_seconds" json:"segment_seconds"`
	DefaultOutputFormats       []string `yaml:"default_output_formats" json:"default_output_formats"`
	DefaultTranslationMode     string   `yaml:"default_translation_mode" json:"default_translation_mode"`
	DefaultTranslationLanguage string   `yaml:"default_translation_language" json:"default_translation_language"`
}

type Video struct {
	Model                     string  `yaml:"model" json:"model"`
	DefaultWidth              int     `yaml:"default_width" json:"default_width"`
	DefaultHeight             int     `yaml:"default_height" json:"default_height"`
	DefaultFrames             int     `yaml:"default_frames" json:"default_frames"`
	DefaultFPS                float64 `yaml:"default_fps" json:"default_fps"`
	DefaultMotionLoRAEnabled  bool    `yaml:"default_motion_lora_enabled" json:"default_motion_lora_enabled"`
	DefaultMotionLoRAStrength float64 `yaml:"default_motion_lora_strength" json:"default_motion_lora_strength"`
	Acceleration              string  `yaml:"acceleration" json:"acceleration"`
}

type PromptEnhancement struct {
	Model          string `yaml:"model" json:"model"`
	DefaultEnabled bool   `yaml:"default_enabled" json:"default_enabled"`
	VisionEnabled  bool   `yaml:"vision_enabled" json:"vision_enabled"`
	MaxTokens      int    `yaml:"max_tokens" json:"max_tokens"`
}

type ImageMetadata struct {
	Creator   string `yaml:"creator" json:"creator"`
	Copyright string `yaml:"copyright" json:"copyright"`
	Website   string `yaml:"website" json:"website"`
	Note      string `yaml:"note" json:"note"`
}

type Storage struct {
	CleanupOnStartup   bool `yaml:"cleanup_on_startup" json:"cleanup_on_startup"`
	TempRetentionHours int  `yaml:"temp_retention_hours" json:"temp_retention_hours"`
}

type Config struct {
	Listen            string            `yaml:"listen" json:"listen"`
	DataDir           string            `yaml:"data_dir" json:"data_dir"`
	Engines           map[string]Engine `yaml:"engines" json:"engines"`
	Image             Image             `yaml:"image" json:"image"`
	Speech            Speech            `yaml:"speech" json:"speech"`
	Recognition       Recognition       `yaml:"recognition" json:"recognition"`
	Video             Video             `yaml:"video" json:"video"`
	PromptEnhancement PromptEnhancement `yaml:"prompt_enhancement" json:"prompt_enhancement"`
	ImageMetadata     ImageMetadata     `yaml:"image_metadata" json:"image_metadata"`
	Storage           Storage           `yaml:"storage" json:"storage"`
}

func Load(path string) (Config, bool, error) {
	created := false
	if _, err := os.Stat(path); os.IsNotExist(err) {
		if err := os.WriteFile(path, defaultYAML, 0o644); err != nil {
			return Config{}, false, fmt.Errorf("create config: %w", err)
		}
		created = true
	} else if err != nil {
		return Config{}, false, err
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return Config{}, created, err
	}
	var cfg Config
	if err := yaml.Unmarshal(defaultYAML, &cfg); err != nil {
		return Config{}, created, fmt.Errorf("parse embedded defaults: %w", err)
	}
	if err := yaml.Unmarshal(b, &cfg); err != nil {
		return Config{}, created, fmt.Errorf("parse config: %w", err)
	}
	cfg = Normalize(cfg)
	if err := Validate(cfg); err != nil {
		return Config{}, created, err
	}
	base := filepath.Dir(path)
	if !filepath.IsAbs(cfg.DataDir) {
		cfg.DataDir = filepath.Join(base, cfg.DataDir)
	}
	return cfg, created, nil
}

func Validate(cfg Config) error {
	if strings.TrimSpace(cfg.Listen) == "" || strings.TrimSpace(cfg.DataDir) == "" {
		return fmt.Errorf("listen and data_dir are required")
	}
	if _, _, err := net.SplitHostPort(cfg.Listen); err != nil {
		return fmt.Errorf("invalid listen address: %w", err)
	}
	for _, kind := range []string{"image", "speech", "recognition", "video", "prompt", "media", "trainer", "upscale"} {
		endpoint := cfg.Engines[kind].Endpoint
		parsed, err := url.Parse(endpoint)
		if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
			return fmt.Errorf("engines.%s.endpoint must be an http(s) URL", kind)
		}
	}
	if cfg.Image.DefaultWidth < 256 || cfg.Image.DefaultHeight < 256 || cfg.Image.MaxReferenceImages < 1 {
		return fmt.Errorf("invalid image defaults")
	}
	if _, ok := cfg.Image.Backends[cfg.Image.DefaultMode]; !ok {
		return fmt.Errorf("image.default_mode must name a configured backend")
	}
	validCheckpoints := map[string]bool{
		"official": true, "ray-v1": true, "ray-v2": true, "ray-v2-nvfp4": true,
		"ray-v3": true, "ray-v4": true, "ray-v4-nvfp4": true,
		"moody-v7": true, "moody-cutie-v4": true, "moody-amateur-v1": true,
		"chriscole-edit-v1.1": true,
	}
	if !validCheckpoints[cfg.Image.DefaultCheckpoint] {
		return fmt.Errorf("image.default_checkpoint is unsupported")
	}
	for _, checkpoint := range cfg.Image.VisibleCheckpoints {
		if !validCheckpoints[checkpoint] {
			return fmt.Errorf("image.visible_checkpoints contains an unsupported checkpoint")
		}
	}
	for mode, backend := range cfg.Image.Backends {
		if mode != "create" && mode != "edit" && mode != "control" {
			return fmt.Errorf("unsupported image backend mode: %s", mode)
		}
		parsed, err := url.Parse(backend.Endpoint)
		if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
			return fmt.Errorf("image.backends.%s.endpoint must be an http(s) URL", mode)
		}
		if strings.TrimSpace(backend.Model) == "" {
			return fmt.Errorf("image.backends.%s.model is required", mode)
		}
	}
	if cfg.Video.DefaultWidth < 256 || cfg.Video.DefaultHeight < 256 || cfg.Video.DefaultWidth > 1920 || cfg.Video.DefaultHeight > 1920 || cfg.Video.DefaultWidth%64 != 0 || cfg.Video.DefaultHeight%64 != 0 {
		return fmt.Errorf("video width and height must be between 256 and 1920 and divisible by 64")
	}
	if cfg.Video.DefaultFrames < 9 || (cfg.Video.DefaultFrames-1)%8 != 0 || cfg.Video.DefaultFPS <= 0 || cfg.Video.DefaultFPS > 60 {
		return fmt.Errorf("invalid video frame or fps defaults")
	}
	if cfg.Video.DefaultMotionLoRAStrength < 0 || cfg.Video.DefaultMotionLoRAStrength > 1 {
		return fmt.Errorf("video.default_motion_lora_strength must be between 0 and 1")
	}
	if cfg.Video.Acceleration != "auto" && cfg.Video.Acceleration != "dense" {
		return fmt.Errorf("video.acceleration must be auto or dense")
	}
	if cfg.Recognition.MaxUploadMB < 1 {
		return fmt.Errorf("recognition.max_upload_mb must be positive")
	}
	if cfg.Recognition.SegmentSeconds < 5 || cfg.Recognition.SegmentSeconds > 180 {
		return fmt.Errorf("recognition.segment_seconds must be between 5 and 180")
	}
	if len(cfg.Recognition.DefaultOutputFormats) == 0 {
		return fmt.Errorf("recognition.default_output_formats must not be empty")
	}
	for _, format := range cfg.Recognition.DefaultOutputFormats {
		if !isOutputFormat(format) {
			return fmt.Errorf("unsupported recognition output format: %s", format)
		}
	}
	if mode := cfg.Recognition.DefaultTranslationMode; mode != "none" && mode != "translated" && mode != "bilingual" {
		return fmt.Errorf("invalid recognition.default_translation_mode")
	}
	if strings.TrimSpace(cfg.PromptEnhancement.Model) == "" {
		return fmt.Errorf("prompt_enhancement.model is required")
	}
	if cfg.PromptEnhancement.MaxTokens < 64 || cfg.PromptEnhancement.MaxTokens > 2048 {
		return fmt.Errorf("prompt_enhancement.max_tokens must be between 64 and 2048")
	}
	if cfg.Storage.TempRetentionHours < 1 || cfg.Storage.TempRetentionHours > 8760 {
		return fmt.Errorf("storage.temp_retention_hours must be between 1 and 8760")
	}
	if len(cfg.ImageMetadata.Creator) > 256 || len(cfg.ImageMetadata.Copyright) > 512 || len(cfg.ImageMetadata.Website) > 2048 || len(cfg.ImageMetadata.Note) > 2000 {
		return fmt.Errorf("image_metadata fields are too long")
	}
	return nil
}

func Normalize(cfg Config) Config {
	if cfg.Engines == nil {
		cfg.Engines = map[string]Engine{}
	}
	if strings.TrimSpace(cfg.Engines["trainer"].Endpoint) == "" {
		cfg.Engines["trainer"] = Engine{Endpoint: "http://127.0.0.1:8704"}
	}
	if strings.TrimSpace(cfg.Engines["upscale"].Endpoint) == "" {
		cfg.Engines["upscale"] = Engine{Endpoint: "http://127.0.0.1:8698"}
	}
	if strings.TrimSpace(cfg.Engines["garment"].Endpoint) == "" {
		cfg.Engines["garment"] = Engine{Endpoint: "http://127.0.0.1:8705"}
	}
	for kind, engine := range cfg.Engines {
		engine.Endpoint = strings.TrimRight(strings.TrimSpace(engine.Endpoint), "/")
		cfg.Engines[kind] = engine
	}
	cfg.Listen = strings.TrimSpace(cfg.Listen)
	cfg.DataDir = strings.TrimSpace(cfg.DataDir)
	cfg.ImageMetadata.Creator = strings.TrimSpace(cfg.ImageMetadata.Creator)
	cfg.ImageMetadata.Copyright = strings.TrimSpace(cfg.ImageMetadata.Copyright)
	cfg.ImageMetadata.Website = strings.TrimSpace(cfg.ImageMetadata.Website)
	cfg.ImageMetadata.Note = strings.TrimSpace(cfg.ImageMetadata.Note)
	if cfg.Image.DefaultMode == "" {
		cfg.Image.DefaultMode = "create"
	}
	if cfg.Image.DefaultCheckpoint == "" {
		cfg.Image.DefaultCheckpoint = "official"
	}
	if len(cfg.Image.VisibleCheckpoints) == 0 {
		cfg.Image.VisibleCheckpoints = []string{
			"official", "chriscole-edit-v1.1", "moody-v7", "moody-cutie-v4", "moody-amateur-v1",
			"ray-v1", "ray-v2", "ray-v2-nvfp4", "ray-v3", "ray-v4", "ray-v4-nvfp4",
		}
	}
	visible := make([]string, 0, len(cfg.Image.VisibleCheckpoints)+1)
	seenVisible := map[string]bool{}
	for _, checkpoint := range append([]string{"official"}, cfg.Image.VisibleCheckpoints...) {
		checkpoint = strings.TrimSpace(checkpoint)
		if checkpoint != "" && !seenVisible[checkpoint] {
			visible = append(visible, checkpoint)
			seenVisible[checkpoint] = true
		}
	}
	cfg.Image.VisibleCheckpoints = visible
	if cfg.Image.Backends == nil {
		cfg.Image.Backends = map[string]ImageBackend{}
	}
	if len(cfg.Image.Backends) == 0 {
		legacy := cfg.Engines["image"]
		cfg.Image.Backends["create"] = ImageBackend{Endpoint: legacy.Endpoint, Model: cfg.Image.Model}
	}
	for mode, backend := range cfg.Image.Backends {
		backend.Endpoint = strings.TrimRight(strings.TrimSpace(backend.Endpoint), "/")
		backend.Model = strings.TrimSpace(backend.Model)
		cfg.Image.Backends[mode] = backend
	}
	if cfg.Recognition.Model == "Qwen/Qwen3-ASR-1.7B-hf" {
		cfg.Recognition.Model = "Qwen/Qwen3-ASR-1.7B"
	}
	if cfg.Recognition.SegmentSeconds == 0 {
		cfg.Recognition.SegmentSeconds = 180
	}
	if len(cfg.Recognition.DefaultOutputFormats) == 0 {
		cfg.Recognition.DefaultOutputFormats = []string{"srt", "txt"}
	}
	if cfg.Recognition.DefaultTranslationMode == "" {
		cfg.Recognition.DefaultTranslationMode = "none"
	}
	if cfg.Recognition.DefaultTranslationLanguage == "" {
		cfg.Recognition.DefaultTranslationLanguage = "Korean"
	}
	if cfg.Storage.TempRetentionHours == 0 {
		cfg.Storage.TempRetentionHours = 24
	}
	if cfg.Video.DefaultMotionLoRAStrength == 0 {
		cfg.Video.DefaultMotionLoRAStrength = 0.5
	}
	if cfg.Video.Acceleration == "" {
		cfg.Video.Acceleration = "auto"
	}
	return cfg
}

func isOutputFormat(value string) bool {
	switch value {
	case "srt", "vtt", "timestamped_txt", "txt":
		return true
	default:
		return false
	}
}

func Save(path string, cfg Config) error {
	cfg = Normalize(cfg)
	if err := Validate(cfg); err != nil {
		return err
	}
	b, err := yaml.Marshal(cfg)
	if err != nil {
		return fmt.Errorf("encode config: %w", err)
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return fmt.Errorf("write config: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("replace config: %w", err)
	}
	return nil
}
