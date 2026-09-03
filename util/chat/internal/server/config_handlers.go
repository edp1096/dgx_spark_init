package server

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"time"

	"sparktalk/internal/asr"
	"sparktalk/internal/config"
	"sparktalk/internal/extra"
	"sparktalk/internal/imagegen"
	"sparktalk/internal/knowledge"
	"sparktalk/internal/llm"
	"sparktalk/internal/tts"
)

func healthEndpoint(ctx context.Context, endpoint string) map[string]any {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return map[string]any{"status": "offline", "error": err.Error()}
	}
	client := &http.Client{Timeout: 800 * time.Millisecond}
	response, err := client.Do(request)
	if err != nil {
		return map[string]any{"status": "offline", "error": err.Error()}
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return map[string]any{"status": "offline", "error": response.Status}
	}
	result := map[string]any{"status": "ok"}
	_ = json.NewDecoder(io.LimitReader(response.Body, 64<<10)).Decode(&result)
	return result
}

func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	cfg, client := s.snapshot()
	model, err := client.Health(r.Context())
	status := "ok"
	if err != nil {
		status = "degraded"
	}
	imageHealth := map[string]any{"status": "disabled"}
	if cfg.Image.Enabled {
		timeout, _ := time.ParseDuration(cfg.Image.Timeout)
		if timeout > 800*time.Millisecond || timeout <= 0 {
			timeout = 800 * time.Millisecond
		}
		imageHealth = imagegen.New(cfg.Image.Endpoint, cfg.Image.Model, timeout).Health(r.Context())
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"status": status, "endpoint": cfg.Model.Endpoint, "model": model, "error": errorText(err),
		"asr":   s.asrSnapshot().Health(r.Context()),
		"tts":   s.ttsSnapshot().Health(r.Context()),
		"image": imageHealth,
		"extra": map[string]any{
			"ssh":       s.extraSnapshot().Health(r.Context()),
			"collector": healthEndpoint(r.Context(), cfg.Extra.CollectorEndpoint+"/health"),
		},
	})
}

func (s *Server) configuration(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		cfg, _ := s.snapshot()
		writeJSON(w, http.StatusOK, cfg.Public())
	case http.MethodPut:
		var req struct {
			Version     int                     `json:"version"`
			Server      config.ServerConfig     `json:"server"`
			Runtime     config.RuntimeConfig    `json:"runtime"`
			Model       config.ModelConfig      `json:"model"`
			ASR         config.ASRConfig        `json:"asr"`
			TTS         config.TTSConfig        `json:"tts"`
			Context     config.ContextConfig    `json:"context"`
			Memory      config.MemoryConfig     `json:"memory"`
			Tools       config.ToolsConfig      `json:"tools"`
			Image       config.ImageConfig      `json:"image"`
			Extra       config.ExtraConfig      `json:"extra"`
			Appearance  config.AppearanceConfig `json:"appearance"`
			APIKey      string                  `json:"api_key"`
			ClearAPIKey bool                    `json:"clear_api_key"`
		}
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&req); err != nil {
			http.Error(w, "invalid config", http.StatusBadRequest)
			return
		}
		old, _ := s.snapshot()
		next := config.Config{Version: req.Version, Server: req.Server, Runtime: req.Runtime, Model: req.Model, ASR: req.ASR, TTS: req.TTS, Context: req.Context, Memory: req.Memory, Tools: req.Tools, Image: req.Image, Extra: req.Extra, Appearance: req.Appearance}
		if req.ClearAPIKey {
			next.Model.APIKey = ""
		} else if req.APIKey != "" {
			next.Model.APIKey = req.APIKey
		} else {
			next.Model.APIKey = old.Model.APIKey
		}
		activeBundle := s.runtime.ActiveBundle(r.Context())
		if activeBundle == "" {
			activeBundle = old.Runtime.ActiveBundle
		}
		if activeBundle == "" {
			activeBundle = next.Runtime.Bundle
		}
		next.Runtime.ActiveBundle = activeBundle
		next.Normalize()
		if err := config.Save(s.configPath, next); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		restartRequired := s.startup != next.Server
		s.replaceConfig(next)
		s.contextMu.Lock()
		s.contextWindows = make(map[string]int)
		s.contextMu.Unlock()
		writeJSON(w, http.StatusOK, map[string]any{
			"config": next.Public(), "restart_required": restartRequired,
		})
	default:
		methodNotAllowed(w)
	}
}

func (s *Server) replaceConfig(next config.Config) {
	s.runtime.ConfigurePaths(next.Runtime.DataDir, next.Runtime.ModelCache)
	s.mu.Lock()
	s.cfg = next
	s.llm = llm.New(next.Model.Endpoint, next.Model.DefaultModel, next.Model.APIKey, next.Model.ModelType).WithThinkingBudget(next.Model.ThinkingBudget)
	s.asr = asr.New(next.ASR)
	s.tts = tts.New(next.TTS)
	s.extra = extra.New(next.Extra.SSHEndpoint)
	s.collector = knowledge.NewCollectorClient(next.Extra.CollectorEndpoint)
	s.mu.Unlock()
}

func (s *Server) models(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	_, client := s.snapshot()
	models, err := client.Models(r.Context())
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	writeJSON(w, http.StatusOK, models)
}
