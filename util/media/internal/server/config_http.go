package server

import (
	"encoding/json"
	"io"
	"mediaapp/internal/config"
	"net/http"
)

func (s *Server) updateConfig(w http.ResponseWriter, r *http.Request) {
	if s.configPath == "" {
		http.Error(w, "configuration is read-only", http.StatusNotImplemented)
		return
	}
	var next config.Config
	decoder := json.NewDecoder(io.LimitReader(r.Body, 1<<20))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&next); err != nil {
		http.Error(w, "invalid configuration: "+err.Error(), http.StatusBadRequest)
		return
	}
	next = config.Normalize(next)
	if err := config.Validate(next); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	previous := s.config()
	if err := config.Save(s.configPath, next); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	s.cfgMu.Lock()
	s.cfg = next
	s.cfgMu.Unlock()
	restartRequired := next.Listen != previous.Listen || next.DataDir != previous.DataDir
	writeJSON(w, http.StatusOK, map[string]any{"config": next, "restart_required": restartRequired})
}
