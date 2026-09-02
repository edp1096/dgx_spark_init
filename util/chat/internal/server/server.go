package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"net/http"
	"path"
	"strings"
	"sync"
	"time"

	"sparktalk/internal/asr"
	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/extra"
	"sparktalk/internal/llm"
	"sparktalk/internal/media"
	"sparktalk/internal/orchestrator"
	"sparktalk/internal/tts"
)

type Server struct {
	mu             sync.RWMutex
	cfg            config.Config
	startup        config.ServerConfig
	configPath     string
	db             *db.DB
	llm            *llm.Client
	asr            *asr.Client
	tts            *tts.Client
	extra          *extra.Client
	media          *media.Store
	runtime        *orchestrator.Controller
	server         *http.Server
	contextMu      sync.Mutex
	contextWindows map[string]int
	compactionMu   sync.Mutex
	asrMu          sync.Mutex
	ttsMu          sync.Mutex
	approvalsMu    sync.Mutex
	approvals      map[string]*toolApproval
}

func New(cfg config.Config, configPath string, store *db.DB, client *llm.Client, embedded fs.FS) (*Server, error) {
	web, err := fs.Sub(embedded, "web/dist")
	if err != nil {
		return nil, err
	}
	mediaStore, err := media.New(cfg.Server.Database)
	if err != nil {
		return nil, fmt.Errorf("media storage: %w", err)
	}
	runtimeController, err := orchestrator.NewController()
	if err != nil {
		return nil, fmt.Errorf("runtime controller: %w", err)
	}
	runtimeController.ConfigurePaths(cfg.Runtime.DataDir, cfg.Runtime.ModelCache)
	if cfg.Runtime.Mode == "managed" {
		activeBundle := runtimeController.ActiveBundle(context.Background())
		if cfg.Runtime.AutoStart {
			activeBundle = cfg.Runtime.Bundle
		} else if activeBundle == "" {
			activeBundle = cfg.Runtime.ActiveBundle
		}
		cfg.Runtime.ActiveBundle = activeBundle
		cfg.Normalize()
		client = llm.New(cfg.Model.Endpoint, cfg.Model.DefaultModel, cfg.Model.APIKey, cfg.Model.ModelType).WithThinkingBudget(cfg.Model.ThinkingBudget)
	}
	s := &Server{cfg: cfg, startup: cfg.Server, configPath: configPath, db: store, llm: client, asr: asr.New(cfg.ASR), tts: tts.New(cfg.TTS), extra: extra.New(cfg.Extra.SSHEndpoint), media: mediaStore, runtime: runtimeController, contextWindows: make(map[string]int), approvals: make(map[string]*toolApproval)}
	mux := http.NewServeMux()
	mux.HandleFunc("/api/health", s.health)
	mux.HandleFunc("/api/config", s.configuration)
	mux.HandleFunc("/api/runtime", s.runtimeStatus)
	mux.HandleFunc("/api/runtime/", s.runtimeAction)
	mux.HandleFunc("/api/models", s.models)
	mux.HandleFunc("/api/images", s.uploadImage)
	mux.HandleFunc("/api/images/", s.image)
	mux.HandleFunc("/api/files", s.uploadFile)
	mux.HandleFunc("/api/files/", s.file)
	mux.HandleFunc("/api/media", s.mediaUsage)
	mux.HandleFunc("/api/media/source", s.uploadSource)
	mux.HandleFunc("/api/memories", s.memories)
	mux.HandleFunc("/api/memories/", s.memory)
	mux.HandleFunc("/api/search/page", s.searchConversationPage)
	mux.HandleFunc("/api/search", s.searchConversations)
	mux.HandleFunc("/api/skills", s.skillCatalog)
	mux.HandleFunc("/api/tool-audit", s.toolAudits)
	mux.HandleFunc("/api/asr/transcribe", s.transcribeVoice)
	mux.HandleFunc("/api/tts/speech", s.synthesizeSpeech)
	mux.HandleFunc("/api/ssh/hosts", s.sshHosts)
	mux.HandleFunc("/api/ssh/hosts/", s.sshHost)
	mux.HandleFunc("/api/ssh/keys", s.sshKeys)
	mux.HandleFunc("/api/ssh/keys/", s.sshKey)
	mux.HandleFunc("/api/tool-approvals/", s.toolApproval)
	mux.HandleFunc("/api/messages/", s.messageAction)
	mux.HandleFunc("/api/groups", s.groups)
	mux.HandleFunc("/api/groups/", s.group)
	mux.HandleFunc("/api/sessions", s.sessions)
	mux.HandleFunc("/api/sessions/", s.session)
	mux.HandleFunc("/api/chat", s.chat)
	mux.Handle("/", spaHandler(web))
	s.server = &http.Server{Addr: cfg.Server.ListenAddr, Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	if cfg.Runtime.Mode == "managed" && cfg.Runtime.AutoStart {
		_ = s.runtime.StartBundle(context.Background(), cfg.Runtime.Bundle, cfg.Runtime.MemoryReserveGiB)
	}
	return s, nil
}

func (s *Server) ListenAndServe() error { return s.server.ListenAndServe() }

func (s *Server) snapshot() (config.Config, *llm.Client) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.cfg, s.llm
}

func (s *Server) asrSnapshot() *asr.Client {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.asr
}

func (s *Server) ttsSnapshot() *tts.Client {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.tts
}

func (s *Server) extraSnapshot() *extra.Client {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.extra
}

func spaHandler(web fs.FS) http.Handler {
	files := http.FileServer(http.FS(web))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requested := strings.TrimPrefix(path.Clean(r.URL.Path), "/")
		if requested != "." {
			if _, err := fs.Stat(web, requested); err == nil {
				files.ServeHTTP(w, r)
				return
			}
		}
		index, err := fs.ReadFile(web, "index.html")
		if err != nil {
			http.Error(w, "web UI is not built; run make dist", 503)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write(index)
	})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func methodNotAllowed(w http.ResponseWriter) {
	http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
}
func errorText(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}
func newID() string { return fmt.Sprintf("chat-%d", time.Now().UnixNano()) }
