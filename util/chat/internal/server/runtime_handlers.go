package server

import (
	"net/http"
	"strings"

	"sparktalk/internal/config"
)

func (s *Server) runtimeStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	cfg, _ := s.snapshot()
	writeJSON(w, http.StatusOK, s.runtime.Snapshot(r.Context(), cfg.Runtime.ActiveBundle))
}

func (s *Server) runtimeAction(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	parts := strings.Split(strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/runtime/"), "/"), "/")
	if len(parts) != 3 {
		http.Error(w, "runtime action path must be bundles/ID/ACTION or components/ID/ACTION", http.StatusNotFound)
		return
	}
	cfg, _ := s.snapshot()
	if cfg.Runtime.Mode != "managed" {
		http.Error(w, "runtime operations require managed mode", http.StatusConflict)
		return
	}
	var err error
	switch parts[0] {
	case "bundles":
		bundle, exists := s.runtime.Catalog().Bundle(parts[1])
		if !exists {
			http.Error(w, "unknown runtime bundle", http.StatusNotFound)
			return
		}
		switch parts[2] {
		case "start":
			err = s.runtime.StartBundle(r.Context(), bundle.ID, cfg.Runtime.MemoryReserveGiB)
			if err == nil {
				next := cfg
				next.Runtime.ActiveBundle = bundle.ID
				next.Normalize()
				if saveErr := config.Save(s.configPath, next); saveErr != nil {
					err = saveErr
				} else {
					s.replaceConfig(next)
				}
			}
		case "stop":
			err = s.runtime.StopBundle(bundle.ID)
		default:
			http.Error(w, "bundle action must be start or stop", http.StatusBadRequest)
			return
		}
	case "components":
		err = s.runtime.ComponentAction(parts[1], parts[2])
	default:
		http.Error(w, "unknown runtime resource", http.StatusNotFound)
		return
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}
	selectedBundle := cfg.Runtime.ActiveBundle
	if parts[0] == "bundles" && parts[2] == "start" {
		selectedBundle = parts[1]
	}
	writeJSON(w, http.StatusAccepted, s.runtime.Snapshot(r.Context(), selectedBundle))
}
