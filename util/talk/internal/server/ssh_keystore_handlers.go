package server

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"sparktalk/internal/config"
	"sparktalk/internal/orchestrator"
)

func (s *Server) sshKeyStore(w http.ResponseWriter, r *http.Request) {
	cfg, _ := s.snapshot()
	if r.Method == http.MethodGet {
		report := orchestrator.KeyStoreReport{}
		if len(cfg.Runtime.KeyStoreHosts) > 0 {
			_, report, _ = s.runtime.KeyStore(r.Context(), cfg.Runtime.KeyStoreHosts, cfg.Runtime.KeyStorePeers, "status", "", nil)
		}
		writeJSON(w, 200, map[string]any{"hosts": cfg.Runtime.KeyStoreHosts, "peers": cfg.Runtime.KeyStorePeers, "available_hosts": orchestrator.SortedKeyHosts(s.runtime.Catalog()), "report": report})
		return
	}
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	var req struct {
		Action string   `json:"action"`
		Target string   `json:"target"`
		Hosts  []string `json:"hosts"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4096)).Decode(&req); err != nil {
		http.Error(w, "invalid key store request", 400)
		return
	}
	if req.Action == "configure" {
		s.runtimeMu.Lock()
		defer s.runtimeMu.Unlock()
		cfg, _ = s.snapshot()
		if len(req.Hosts) == 0 {
			http.Error(w, "키 저장소 호스트를 하나 이상 선택하세요", 400)
			return
		}
		if err := orchestrator.ValidateKeyStoreHosts(s.runtime.Catalog(), req.Hosts); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		// Do not drop the current owner or an offline replica silently.
		if len(cfg.Runtime.KeyStoreHosts) > 0 {
			_, old, _ := s.runtime.KeyStore(r.Context(), cfg.Runtime.KeyStoreHosts, cfg.Runtime.KeyStorePeers, "status", "", nil)
			for _, h := range cfg.Runtime.KeyStoreHosts {
				found := false
				for _, id := range req.Hosts {
					if id == h {
						found = true
					}
				}
				if !found {
					http.Error(w, "기존 복제 호스트 제거는 지원하지 않습니다. 먼저 관리 권한과 저장소를 정리하세요.", 409)
					return
				}
			}
			if old.Error != "" {
				http.Error(w, old.Error, 409)
				return
			}
		}
		peers, err := s.runtime.PrepareKeyStorePeers(req.Hosts, cfg.Runtime.KeyStorePeers)
		if err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		_, report, err := s.runtime.KeyStore(r.Context(), req.Hosts, peers, "sync", "", nil)
		if err != nil {
			http.Error(w, err.Error(), 409)
			return
		}
		for _, replica := range report.Replicas {
			if replica.Error != "" {
				http.Error(w, replica.Error, 409)
				return
			}
		}
		cfg.Runtime.KeyStoreHosts = req.Hosts
		cfg.Runtime.KeyStorePeers = peers
		if err := config.Save(s.configPath, cfg); err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		s.replaceConfig(cfg)
		writeJSON(w, 200, report)
		return
	}
	if req.Action != "sync" && req.Action != "handoff" {
		http.Error(w, "invalid action", 400)
		return
	}
	_, report, err := s.runtime.KeyStore(r.Context(), cfg.Runtime.KeyStoreHosts, cfg.Runtime.KeyStorePeers, req.Action, req.Target, nil)
	if err != nil {
		http.Error(w, err.Error(), 409)
		return
	}
	writeJSON(w, 200, report)
}

func (s *Server) managedKeyAction(w http.ResponseWriter, r *http.Request, action, id string, data []byte) bool {
	cfg, _ := s.snapshot()
	if len(cfg.Runtime.KeyStoreHosts) == 0 {
		return false
	}
	input, _ := json.Marshal(map[string]any{"id": id, "data": data})
	out, report, err := s.runtime.KeyStore(r.Context(), cfg.Runtime.KeyStoreHosts, cfg.Runtime.KeyStorePeers, action, "", input)
	if err != nil {
		http.Error(w, err.Error(), http.StatusConflict)
		return true
	}
	pending := report.Error != ""
	for _, replica := range report.Replicas {
		if replica.Error != "" {
			pending = true
		}
	}
	if pending {
		w.Header().Set("X-SparkTalk-Key-Sync", "pending")
	}
	if action == "delete" {
		w.WriteHeader(http.StatusNoContent)
		return true
	}
	status := 200
	if action == "generate" || action == "import" {
		status = 201
	}
	writeJSON(w, status, json.RawMessage(out))
	return true
}

func (s *Server) runKeySync(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			cfg, _ := s.snapshot()
			if len(cfg.Runtime.KeyStoreHosts) == 0 {
				continue
			}
			syncCtx, cancel := context.WithTimeout(ctx, 90*time.Second)
			_, _, _ = s.runtime.KeyStore(syncCtx, cfg.Runtime.KeyStoreHosts, cfg.Runtime.KeyStorePeers, "sync", "", nil)
			cancel()
		}
	}
}
