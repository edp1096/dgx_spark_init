package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"gopkg.in/yaml.v3"
	"io"
	"net/http"
	"sparktalk/internal/orchestrator"
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
	s.runtimeMu.Lock()
	defer s.runtimeMu.Unlock()
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
		err = s.runtime.ComponentAction(parts[1], parts[2], cfg.Runtime.ActiveBundle)
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

// Parse imports without changing live configuration or starting any service.
func (s *Server) runtimeCatalogParse(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	data, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 1<<20))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	var catalog orchestrator.Catalog
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err = decoder.Decode(&catalog); err == nil {
		var trailing any
		if tailErr := decoder.Decode(&trailing); tailErr != io.EOF {
			err = fmt.Errorf("세트 파일에는 문서 하나만 넣으세요")
		}
	}
	if err == nil {
		catalog, err = orchestrator.ValidateCatalog(catalog)
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	writeJSON(w, http.StatusOK, catalog)
}

func (s *Server) runtimeProbe(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	var req struct {
		URL string `json:"url"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 8192)).Decode(&req); err != nil {
		http.Error(w, "invalid probe", http.StatusBadRequest)
		return
	}
	if !strings.HasPrefix(req.URL, "http://") && !strings.HasPrefix(req.URL, "https://") {
		http.Error(w, "HTTP URL required", http.StatusBadRequest)
		return
	}
	writeJSON(w, http.StatusOK, healthEndpoint(r.Context(), req.URL))
}
