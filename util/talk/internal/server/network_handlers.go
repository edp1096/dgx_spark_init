package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sparktalk/internal/config"
	"sparktalk/internal/orchestrator"
)

func resolveAutoNetwork(ctx context.Context, cfg *config.Config) error {
	if cfg.Runtime.Catalog == nil || cfg.Runtime.Catalog.Network == nil || !cfg.Runtime.Catalog.Network.Enabled {
		return nil
	}
	result := orchestrator.DiscoverNetwork(ctx, *cfg.Runtime.Catalog, "")
	if result.Catalog == nil {
		cloned, _ := orchestrator.ValidateCatalog(*cfg.Runtime.Catalog)
		cloned.Network.LastError = result.Error
		cfg.Runtime.Catalog = &cloned
		return fmt.Errorf("%s", result.Error)
	}
	cfg.Runtime.Catalog = result.Catalog
	cfg.Normalize()
	return nil
}

func (s *Server) networkDiscover(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	var req struct {
		Catalog  *orchestrator.Catalog `json:"catalog"`
		WorkerID string                `json:"worker_id"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&req); err != nil {
		http.Error(w, "invalid discovery request", 400)
		return
	}
	if req.Catalog == nil {
		cfg, _ := s.snapshot()
		req.Catalog = cfg.Runtime.Catalog
	}
	if req.Catalog == nil {
		http.Error(w, "runtime catalog is required", 400)
		return
	}
	writeJSON(w, 200, orchestrator.DiscoverNetwork(r.Context(), *req.Catalog, req.WorkerID))
}

func (s *Server) refreshAutoNetwork(ctx context.Context, cfg *config.Config) error {
	if cfg.Runtime.Catalog == nil || cfg.Runtime.Catalog.Network == nil || !cfg.Runtime.Catalog.Network.Enabled {
		return nil
	}
	if err := resolveAutoNetwork(ctx, cfg); err != nil {
		return err
	}
	if err := s.runtime.UpdateCatalog(*cfg.Runtime.Catalog, func() error { return config.Save(s.configPath, *cfg) }); err != nil {
		return err
	}
	s.replaceConfig(*cfg)
	return nil
}
