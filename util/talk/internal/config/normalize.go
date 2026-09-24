package config

import (
	"os"
	"path/filepath"
	"sparktalk/internal/orchestrator"
	"strings"
)

func (c *Config) Normalize() {

	if c.Version < 2 {
		c.Version = 2
	}
	c.Server.ListenAddr = strings.TrimSpace(c.Server.ListenAddr)
	c.Server.Database = strings.TrimSpace(c.Server.Database)
	c.Runtime.Mode = strings.ToLower(strings.TrimSpace(c.Runtime.Mode))
	c.Runtime.Bundle = strings.ToLower(strings.TrimSpace(c.Runtime.Bundle))
	c.Runtime.ActiveBundle = strings.ToLower(strings.TrimSpace(c.Runtime.ActiveBundle))
	c.Runtime.DataDir = strings.TrimSpace(c.Runtime.DataDir)
	c.Runtime.ModelCache = strings.TrimSpace(c.Runtime.ModelCache)
	if c.Runtime.Mode != "external" {
		c.Runtime.Mode = "managed"
	}
	if c.Runtime.Catalog == nil {
		catalog, _ := orchestrator.LoadCatalog()
		c.Runtime.Catalog = &catalog
	}
	c.removeRetiredRuntimes()
	if c.Runtime.BuiltinRevision < 1 {
		// Offer the new optional service without changing any existing set membership.
		hasDocuments := false
		for _, component := range c.Runtime.Catalog.Components {
			if component.ID == "extra-documents" {
				hasDocuments = true
			}
		}
		if !hasDocuments {
			if defaults, err := orchestrator.LoadCatalog(); err == nil {
				if component, ok := defaults.Component("extra-documents"); ok {
					c.Runtime.Catalog.Components = append(c.Runtime.Catalog.Components, component)
				}
			}
		}
		c.Runtime.BuiltinRevision = 1
	}
	if c.Runtime.BuiltinRevision < 2 {
		if defaults, err := orchestrator.LoadCatalog(); err == nil {
			hasComponent, hasBundle := false, false
			for _, item := range c.Runtime.Catalog.Components {
				if item.ID == "ds41" {
					hasComponent = true
				}
			}
			for _, item := range c.Runtime.Catalog.Bundles {
				if item.ID == "ds41" {
					hasBundle = true
				}
			}

			// Do not invent hosts in a custom catalog or override an existing DS41 definition.
			_, head := c.Runtime.Catalog.Hosts["local"]
			_, worker := c.Runtime.Catalog.Hosts["worker"]
			if !hasComponent && head && worker {
				item, _ := defaults.Component("ds41")
				c.Runtime.Catalog.Components = append(c.Runtime.Catalog.Components, item)
			}
			if !hasBundle && head && worker {
				item, _ := defaults.Bundle("ds41")
				available := make(map[string]bool)
				for _, component := range c.Runtime.Catalog.Components {
					available[component.ID] = true
				}
				complete := true
				for _, id := range item.Components {
					complete = complete && available[id]
				}
				if complete {
					c.Runtime.Catalog.Bundles = append(c.Runtime.Catalog.Bundles, item)
				}
			}
		}
		c.Runtime.BuiltinRevision = 2
	}
	if c.Runtime.BuiltinRevision < 3 {
		if defaults, err := orchestrator.LoadCatalog(); err == nil {
			hasComponent, hasBundle := false, false
			for _, item := range c.Runtime.Catalog.Components {
				if item.ID == "flash-next-tp2" {
					hasComponent = true
				}
			}
			for _, item := range c.Runtime.Catalog.Bundles {
				if item.ID == "flash-next-tp2" {
					hasBundle = true
				}
			}

			// Do not invent hosts in a custom catalog or override an existing Qwen TP2 definition.
			_, head := c.Runtime.Catalog.Hosts["local"]
			_, worker := c.Runtime.Catalog.Hosts["worker"]
			if !hasComponent && head && worker {
				item, _ := defaults.Component("flash-next-tp2")
				c.Runtime.Catalog.Components = append(c.Runtime.Catalog.Components, item)
			}
			if !hasBundle && head && worker {
				item, _ := defaults.Bundle("flash-next-tp2")
				available := make(map[string]bool)
				for _, component := range c.Runtime.Catalog.Components {
					available[component.ID] = true
				}
				complete := true
				for _, id := range item.Components {
					complete = complete && available[id]
				}
				if complete {
					c.Runtime.Catalog.Bundles = append(c.Runtime.Catalog.Bundles, item)
				}
			}
		}
		c.Runtime.BuiltinRevision = 3
	}
	if c.Runtime.BuiltinRevision < 4 {
		c.migrateQwenHuihuiModel()
		c.Runtime.BuiltinRevision = 4
	}
	if c.Runtime.BuiltinRevision < 5 {
		c.addSingleSparkMoEModels()
		c.Runtime.BuiltinRevision = 5
	}
	if c.Runtime.BuiltinRevision < 9 {
		c.migrateGemmaCheckpoint()
		c.Runtime.BuiltinRevision = 9
	}
	if c.Runtime.BuiltinRevision < 10 {
		c.migrateGemmaSGLang()
		c.Runtime.BuiltinRevision = 10
	}
	if c.Runtime.BuiltinRevision < 11 {
		c.migrateGemmaOwnNVFP4()
		c.Runtime.BuiltinRevision = 11
	}
	if c.Runtime.BuiltinRevision < 12 {
		c.migrateQwenQADTP1()
		c.migrateNemotronDiarizationBudget()
		c.Runtime.BuiltinRevision = 12
	}
	if c.Runtime.BuiltinRevision < 13 {
		c.migrateQwenTP1Context1M()
		c.Runtime.BuiltinRevision = 13
	}
	if c.Runtime.BuiltinRevision < 14 {
		c.migrateQwenQADFlux()
		c.Runtime.BuiltinRevision = 14
	}
	c.normalizeRuntimeDisplayNames()
	if catalog, err := orchestrator.ValidateCatalog(*c.Runtime.Catalog); err == nil {
		c.Runtime.Catalog = &catalog
	}
	if c.Runtime.Bundle == "" {
		c.Runtime.Bundle = "flash-next"
	}
	if c.Runtime.ActiveBundle == "" {
		c.Runtime.ActiveBundle = c.Runtime.Bundle
	}
	if c.Runtime.MemoryReserveGiB <= 0 {
		c.Runtime.MemoryReserveGiB = 4
	}
	if home, err := os.UserHomeDir(); err == nil {
		if c.Runtime.DataDir == "" {
			c.Runtime.DataDir = filepath.Join(home, ".local", "share", "sparktalk")
		}
		if c.Runtime.ModelCache == "" {
			c.Runtime.ModelCache = filepath.Join(home, ".cache", "huggingface")
		}
	}
	c.Model.Endpoint = strings.TrimRight(strings.TrimSpace(c.Model.Endpoint), "/")
	c.Model.DefaultModel = strings.TrimSpace(c.Model.DefaultModel)
	c.Model.ModelType = strings.ToLower(strings.TrimSpace(c.Model.ModelType))
	switch c.Model.ModelType {
	case "qwen3.5", "qwen3.8", "qwen3.8-gguf", "qwen3.8-exl3", "gemma4", "gemma4-vllm", "glm5.3", "deepseek-v4", "generic":
	default:
		c.Model.ModelType = "generic"
	}
	c.Model.ReasoningEffort = strings.TrimSpace(c.Model.ReasoningEffort)
	if c.Model.ThinkingBudget < 0 {
		c.Model.ThinkingBudget = 0
	}
	c.Model.SystemPrompt = strings.TrimSpace(c.Model.SystemPrompt)
	c.Model.SystemPromptPreset = strings.TrimSpace(c.Model.SystemPromptPreset)
	c.Model.normalizePromptComposer()
	c.ASR.FFmpegEndpoint = strings.TrimRight(strings.TrimSpace(c.ASR.FFmpegEndpoint), "/")
	c.ASR.Endpoint = strings.TrimRight(strings.TrimSpace(c.ASR.Endpoint), "/")
	c.ASR.Model = strings.TrimSpace(c.ASR.Model)
	c.ASR.VoiceLanguage = normalizeASRLocale(c.ASR.VoiceLanguage)
	c.ASR.MediaLanguage = normalizeASRLocale(c.ASR.MediaLanguage)
	c.ASR.VoiceEndpoint = strings.TrimRight(strings.TrimSpace(c.ASR.VoiceEndpoint), "/")
	c.ASR.VoiceModel = strings.TrimSpace(c.ASR.VoiceModel)
	c.ASR.MediaEndpoint = strings.TrimRight(strings.TrimSpace(c.ASR.MediaEndpoint), "/")
	c.ASR.MediaModel = strings.TrimSpace(c.ASR.MediaModel)
	c.ASR.Language = normalizeASRLocale(c.ASR.Language)
	c.ASR.Prompt = strings.TrimSpace(c.ASR.Prompt)
	c.ASR.Timeout = strings.TrimSpace(c.ASR.Timeout)
	c.TTS.Endpoint = strings.TrimRight(strings.TrimSpace(c.TTS.Endpoint), "/")
	c.TTS.Model = strings.TrimSpace(c.TTS.Model)
	c.TTS.Language = strings.TrimSpace(c.TTS.Language)
	c.TTS.HanjaReading = strings.ToLower(strings.TrimSpace(c.TTS.HanjaReading))
	c.TTS.Voice = strings.TrimSpace(c.TTS.Voice)
	c.TTS.Timeout = strings.TrimSpace(c.TTS.Timeout)
	c.Image.Endpoint = strings.TrimRight(strings.TrimSpace(c.Image.Endpoint), "/")
	c.Image.Model = strings.TrimSpace(c.Image.Model)
	c.Image.Mode = strings.ToLower(strings.TrimSpace(c.Image.Mode))
	c.Image.DefaultSize = strings.ToLower(strings.TrimSpace(c.Image.DefaultSize))
	c.Image.Timeout = strings.TrimSpace(c.Image.Timeout)
	if strings.TrimSpace(c.Extra.MediaEndpoint) == "" {
		c.Extra.MediaEndpoint = c.ASR.FFmpegEndpoint
	}
	if strings.TrimSpace(c.Extra.MediaEndpoint) == "" {
		c.Extra.MediaEndpoint = "http://127.0.0.1:8690"
	}
	c.Extra.MediaEndpoint = strings.TrimRight(strings.TrimSpace(c.Extra.MediaEndpoint), "/")
	c.ASR.FFmpegEndpoint = c.Extra.MediaEndpoint
	c.Extra.SSHEndpoint = strings.TrimRight(strings.TrimSpace(c.Extra.SSHEndpoint), "/")
	c.Extra.DocumentsEndpoint = strings.TrimRight(strings.TrimSpace(c.Extra.DocumentsEndpoint), "/")
	if c.Extra.DocumentsEndpoint == "" {
		c.Extra.DocumentsEndpoint = "http://127.0.0.1:8696"
	}
	c.Extra.CollectorEndpoint = strings.TrimRight(strings.TrimSpace(c.Extra.CollectorEndpoint), "/")
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
		c.ASR.FFmpegEndpoint = "http://127.0.0.1:8690"
	}
	if c.ASR.Endpoint == "" {
		c.ASR.Endpoint = c.ASR.VoiceEndpoint
		if c.ASR.Endpoint == "" {
			c.ASR.Endpoint = c.ASR.MediaEndpoint
		}
	}
	if c.ASR.Endpoint == "" {
		c.ASR.Endpoint = "http://127.0.0.1:8693"
	}
	if c.ASR.Model == "" {
		c.ASR.Model = c.ASR.VoiceModel
		if c.ASR.Model == "" {
			c.ASR.Model = c.ASR.MediaModel
		}
	}
	if c.ASR.Model == "" {
		c.ASR.Model = "nemotron-3.5-asr-streaming-0.6b"
	}
	if c.ASR.VoiceLanguage == "" || c.ASR.VoiceLanguage == "auto" {
		legacy := normalizeASRLocale(c.ASR.Language)
		if legacy != "" && legacy != "auto" {
			c.ASR.VoiceLanguage = legacy
		} else {
			c.ASR.VoiceLanguage = "ko-KR"
		}
	}
	if c.ASR.MediaLanguage == "" {
		c.ASR.MediaLanguage = "auto"
	}
	c.ASR.VoiceEndpoint, c.ASR.VoiceModel = "", ""
	c.ASR.MediaEndpoint, c.ASR.MediaModel, c.ASR.Language = "", "", ""
	if c.ASR.Timeout == "" {
		c.ASR.Timeout = "30m"
	}
	if c.TTS.Endpoint == "" {
		c.TTS.Endpoint = "http://127.0.0.1:8692"
	}
	if c.TTS.Model == "" {
		c.TTS.Model = "magpietts"
	}
	if c.TTS.Language == "" {
		c.TTS.Language = "auto"
	}
	if c.TTS.HanjaReading != "chinese" && c.TTS.HanjaReading != "japanese" {
		c.TTS.HanjaReading = "korean"
	}
	if c.TTS.Voice == "" {
		c.TTS.Voice = "Sofia"
	}
	if c.TTS.SampleRate <= 0 {
		c.TTS.SampleRate = 22050
	}
	if c.TTS.Timeout == "" {
		c.TTS.Timeout = "10m"
	}
	if c.Tools.MaxRounds <= 0 {
		c.Tools.MaxRounds = 256
	}
	if c.Tools.MaxRounds > 1024 {
		c.Tools.MaxRounds = 1024
	}
	if c.Tools.SearchResults <= 0 {
		c.Tools.SearchResults = 15
	}
	if c.Tools.SearchResults > 30 {
		c.Tools.SearchResults = 30
	}
	if c.Tools.Timeout == "" {
		c.Tools.Timeout = "15s"
	}
	if c.Image.Endpoint == "" {
		c.Image.Endpoint = "http://127.0.0.1:8691"
	}
	if c.Image.Mode != "extended" && c.Image.Mode != "reference" && c.Image.Mode != "paint" {
		c.Image.Mode = "basic"
	}
	if c.Image.DefaultSize == "" {
		c.Image.DefaultSize = "1024x1024"
	}
	if c.Image.Timeout == "" {
		c.Image.Timeout = "30m"
	}
	if c.Extra.SSHEndpoint == "" {
		c.Extra.SSHEndpoint = "http://127.0.0.1:8699"
	}
	if c.Extra.CollectorEndpoint == "" {
		c.Extra.CollectorEndpoint = "http://127.0.0.1:8695"
	}
	if c.Runtime.Mode == "managed" {
		c.applyManagedRuntime()
	}
	c.Model.ReasoningEffort = normalizeReasoningEffort(c.Model.ModelType, c.Model.ReasoningEffort)
	if c.Context.CompactAtPercent <= 0 {
		c.Context.CompactAtPercent = 80
	}
	if c.Context.OutputReserve <= 0 {
		c.Context.OutputReserve = 16384
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
	if c.Memory.MaxResults <= 0 {
		c.Memory.MaxResults = 5
	}
	if c.Memory.MaxResults > 12 {
		c.Memory.MaxResults = 12
	}
	if c.Memory.TokenBudget <= 0 {
		c.Memory.TokenBudget = 2048
	}
	if c.Memory.TokenBudget > 8192 {
		c.Memory.TokenBudget = 8192
	}
	if c.Memory.AlwaysMaxResults <= 0 {
		c.Memory.AlwaysMaxResults = 6
	}
	if c.Memory.AlwaysMaxResults > 12 {
		c.Memory.AlwaysMaxResults = 12
	}
	if c.Memory.AlwaysTokenBudget <= 0 {
		c.Memory.AlwaysTokenBudget = 1024
	}
	if c.Memory.AlwaysTokenBudget > 8192 {
		c.Memory.AlwaysTokenBudget = 8192
	}
	c.Appearance.UserName = strings.TrimSpace(c.Appearance.UserName)
	c.Appearance.AssistantAvatar = normalizeAvatar(c.Appearance.AssistantAvatar, "preset:spark")
	c.Appearance.UserAvatar = normalizeAvatar(c.Appearance.UserAvatar, "preset:person-blue")
	c.Appearance.Theme = strings.ToLower(strings.TrimSpace(c.Appearance.Theme))
	switch c.Appearance.Theme {
	case "dark", "light", "system":
	default:
		c.Appearance.Theme = "system"
	}
}
