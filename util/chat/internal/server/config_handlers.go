package server

import (
	"encoding/json"
	"net/http"

	"sparktalk/internal/asr"
	"sparktalk/internal/config"
	"sparktalk/internal/extra"
	"sparktalk/internal/llm"
)

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
	writeJSON(w, http.StatusOK, map[string]any{
		"status": status, "endpoint": cfg.Model.Endpoint, "model": model, "error": errorText(err),
		"asr":   s.asrSnapshot().Health(r.Context()),
		"extra": map[string]any{"ssh": s.extraSnapshot().Health(r.Context())},
	})
}

func (s *Server) configuration(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		cfg, _ := s.snapshot()
		writeJSON(w, http.StatusOK, cfg.Public())
	case http.MethodPut:
		var req struct {
			Server      config.ServerConfig     `json:"server"`
			Model       config.ModelConfig      `json:"model"`
			ASR         config.ASRConfig        `json:"asr"`
			Context     config.ContextConfig    `json:"context"`
			Tools       config.ToolsConfig      `json:"tools"`
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
		next := config.Config{Server: req.Server, Model: req.Model, ASR: req.ASR, Context: req.Context, Tools: req.Tools, Extra: req.Extra, Appearance: req.Appearance}
		if req.ClearAPIKey {
			next.Model.APIKey = ""
		} else if req.APIKey != "" {
			next.Model.APIKey = req.APIKey
		} else {
			next.Model.APIKey = old.Model.APIKey
		}
		next.Normalize()
		if err := config.Save(s.configPath, next); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		restartRequired := s.startup != next.Server
		s.mu.Lock()
		s.cfg = next
		s.llm = llm.New(next.Model.Endpoint, next.Model.DefaultModel, next.Model.APIKey)
		s.asr = asr.New(next.ASR)
		s.extra = extra.New(next.Extra.SSHEndpoint)
		s.mu.Unlock()
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
