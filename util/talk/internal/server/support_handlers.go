package server

import (
	"context"
	"net/http"
	"sparktalk/internal/config"
	"sparktalk/internal/orchestrator"
	"strings"
	"sync"
)

// Availability and feature permission are independent, including when ASR is off.
func supportHealth(ctx context.Context, cfg config.Config) map[string]any {
	out := map[string]any{}
	var mu sync.Mutex
	var wg sync.WaitGroup
	for _, spec := range orchestrator.SupportSpecs() {
		wg.Add(1)
		go func(key string) {
			defer wg.Done()
			state := healthEndpoint(ctx, strings.TrimRight(cfg.SupportEndpoint(key), "/")+"/health")
			state["enabled"] = cfg.SupportEnabled(key)
			mu.Lock()
			out[key] = state
			mu.Unlock()
		}(spec.Key)
	}
	wg.Wait()
	return out
}
func (s *Server) supportServices(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	cfg, _ := s.snapshot()
	type row struct {
		orchestrator.SupportStatus
		Enabled bool `json:"enabled"`
	}
	rows := []row{}
	if cfg.Runtime.Mode == "managed" {
		for _, status := range s.runtime.SupportSnapshot(r.Context(), cfg.Runtime.ActiveBundle) {
			rows = append(rows, row{status, cfg.SupportEnabled(status.Key)})
		}
	} else {
		health := supportHealth(r.Context(), cfg)
		for _, spec := range orchestrator.SupportSpecs() {
			state := health[spec.Key].(map[string]any)
			h := "offline"
			if state["status"] == "ok" {
				h = "online"
			}
			component := orchestrator.Component{ID: spec.ID, Name: spec.Name, Controller: "external", Host: "external", Endpoint: cfg.SupportEndpoint(spec.Key)}
			rows = append(rows, row{orchestrator.SupportStatus{ComponentStatus: orchestrator.ComponentStatus{Component: component, Status: "external", Health: h}, Key: spec.Key, Description: spec.Description, Image: spec.Image, Version: spec.Version, Installed: "external"}, cfg.SupportEnabled(spec.Key)})
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"services": rows, "bundle_id": cfg.Runtime.ActiveBundle, "managed": cfg.Runtime.Mode == "managed", "operation": s.runtime.Operation()})
}
