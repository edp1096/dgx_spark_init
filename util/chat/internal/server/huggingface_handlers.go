package server

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

var modelPreparationMu sync.Mutex
var modelPreparationState = struct {
	sync.RWMutex
	State, Component, Detail string
	Logs                     []string
	StartedAt                time.Time
}{}

func (s *Server) hfTokenPath() string {
	cfg, _ := s.snapshot()
	return filepath.Join(cfg.Runtime.DataDir, "credentials", "huggingface.token")
}
func (s *Server) huggingFaceToken(w http.ResponseWriter, r *http.Request) {
	path := s.hfTokenPath()
	switch r.Method {
	case http.MethodGet:
		data, err := os.ReadFile(path)
		writeJSON(w, 200, map[string]bool{"configured": err == nil && len(data) > 0})
	case http.MethodPut:
		var req struct {
			Token string `json:"token"`
		}
		if json.NewDecoder(http.MaxBytesReader(w, r.Body, 8192)).Decode(&req) != nil {
			http.Error(w, "invalid token request", 400)
			return
		}
		token := strings.TrimSpace(req.Token)
		if len(token) < 8 || len(token) > 4096 || strings.ContainsAny(token, " \t\r\n\x00") {
			http.Error(w, "invalid token", 400)
			return
		}
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			http.Error(w, "credential storage unavailable", 500)
			return
		}
		file, err := os.CreateTemp(filepath.Dir(path), ".hf-token-")
		if err != nil {
			http.Error(w, "credential storage unavailable", 500)
			return
		}
		name := file.Name()
		defer os.Remove(name)
		_, err = io.WriteString(file, token)
		closeErr := file.Close()
		if err != nil || closeErr != nil {
			http.Error(w, "credential save failed", 500)
			return
		}
		if err = os.Rename(name, path); err != nil {
			http.Error(w, "credential save failed", 500)
			return
		}
		writeJSON(w, 200, map[string]bool{"configured": true})
	case http.MethodDelete:
		if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
			http.Error(w, "credential delete failed", 500)
			return
		}
		writeJSON(w, 200, map[string]bool{"configured": false})
	default:
		methodNotAllowed(w)
	}
}
func (s *Server) modelPreparation(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		modelPreparationState.RLock()
		defer modelPreparationState.RUnlock()
		writeJSON(w, 200, map[string]any{"state": modelPreparationState.State, "component": modelPreparationState.Component, "detail": modelPreparationState.Detail, "logs": modelPreparationState.Logs, "started_at": modelPreparationState.StartedAt})
		return
	}
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	var req struct {
		Component string `json:"component"`
		Variant   string `json:"variant"`
		Action    string `json:"action"`
	}
	if json.NewDecoder(http.MaxBytesReader(w, r.Body, 4096)).Decode(&req) != nil {
		http.Error(w, "invalid preparation request", 400)
		return
	}
	if req.Action == "" {
		req.Action = "model"
	}
	if (req.Action != "model" && req.Action != "setup") || (req.Variant != "official" && req.Variant != "abliterated") {
		http.Error(w, "invalid preparation options", 400)
		return
	}
	component, ok := s.runtime.Catalog().Component(req.Component)
	if !ok {
		http.Error(w, "unknown service", 404)
		return
	}
	tokenBytes, err := os.ReadFile(s.hfTokenPath())
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		http.Error(w, "credential storage unavailable", 500)
		return
	}
	if !modelPreparationMu.TryLock() {
		http.Error(w, "model preparation is already running", 409)
		return
	}
	modelPreparationState.Lock()
	modelPreparationState.State = "running"
	modelPreparationState.Component = component.ID
	modelPreparationState.Detail = "모델 준비 중"
	modelPreparationState.Logs = nil
	modelPreparationState.StartedAt = time.Now()
	modelPreparationState.Unlock()
	go func() {
		defer modelPreparationMu.Unlock()
		ctx, cancel := context.WithTimeout(context.Background(), 6*time.Hour)
		defer cancel()
		err := s.runtime.PrepareModel(ctx, component, req.Variant, req.Action, string(tokenBytes), func(line string) {
			modelPreparationState.Lock()
			defer modelPreparationState.Unlock()
			modelPreparationState.Detail = line
			modelPreparationState.Logs = append(modelPreparationState.Logs, line)
			if len(modelPreparationState.Logs) > 100 {
				modelPreparationState.Logs = append([]string(nil), modelPreparationState.Logs[len(modelPreparationState.Logs)-100:]...)
			}
		})
		modelPreparationState.Lock()
		defer modelPreparationState.Unlock()
		modelPreparationState.State = "complete"
		modelPreparationState.Detail = "모델 준비 완료. 실행 중인 모델은 자동으로 교체하지 않습니다."
		if err != nil {
			modelPreparationState.State = "failed"
			modelPreparationState.Detail = err.Error()
		}
	}()
	writeJSON(w, http.StatusAccepted, map[string]string{"state": "running"})
}
