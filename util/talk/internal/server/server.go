package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net"
	"net/http"
	"path"
	"strings"
	"sync"
	"time"

	"sparktalk/internal/asr"
	"sparktalk/internal/browserbridge"
	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/knowledge"
	"sparktalk/internal/llm"
	"sparktalk/internal/media"
	"sparktalk/internal/orchestrator"
	"sparktalk/internal/plugins"
	supportssh "sparktalk/internal/support/ssh"
	"sparktalk/internal/tasklife"
	"sparktalk/internal/tts"
)

type Server struct {
	tasks              tasklife.Group
	plugins            *plugins.Manager
	turnMu             sync.Mutex
	turns              map[string]*activeTurn
	browserSubmissions browserSubmissionState
	browser            *browserbridge.Bridge
	generationMu       sync.Mutex
	generations        map[uint64]context.CancelFunc
	generationID       uint64
	queueClearing      bool
	mu                 sync.RWMutex
	runtimeMu          sync.Mutex
	cfg                config.Config
	startup            config.ServerConfig
	configPath         string
	db                 *db.DB
	llm                *llm.Client
	asr                *asr.Client
	tts                *tts.Client
	sshClient          *supportssh.Client
	media              *media.Store
	knowledge          *knowledge.Store
	knowledgeIndex     *knowledge.Extractor
	collector          *knowledge.CollectorClient
	runtime            *orchestrator.Controller
	server             *http.Server
	contextMu          sync.Mutex
	contextWindows     map[string]int
	compactionMu       sync.Mutex
	asrMu              sync.Mutex
	ttsMu              sync.Mutex
	documentMu         sync.Mutex
	knowledgeJobMu     sync.Mutex
	knowledgeJobs      map[string]*knowledgeJobRun
	knowledgeJobSem    chan struct{}
	knowledgeOCRMu     sync.Mutex
	knowledgeOCRJobs   map[string]*knowledgeOCRRun
	knowledgeOCRSem    chan struct{}
	approvalsMu        sync.Mutex
	approvals          map[string]*toolApproval
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
	knowledgeStore, err := knowledge.New(cfg.Server.Database)
	if err != nil {
		return nil, fmt.Errorf("knowledge storage: %w", err)
	}
	var runtimeController *orchestrator.Controller
	// Resolve physical hosts before probing services or initiating AutoStart.
	if cfg.Runtime.Mode == "managed" && cfg.Runtime.Catalog != nil && cfg.Runtime.Catalog.Network != nil && cfg.Runtime.Catalog.Network.Enabled {
		_ = resolveAutoNetwork(context.Background(), &cfg)
		if err := config.Save(configPath, cfg); err != nil {
			return nil, err
		}
	}
	if cfg.Runtime.Catalog != nil {
		runtimeController, err = orchestrator.NewControllerWithCatalog(*cfg.Runtime.Catalog)
	} else {
		runtimeController, err = orchestrator.NewController()
	}
	if err != nil {
		return nil, fmt.Errorf("runtime controller: %w", err)
	}
	runtimeController.ConfigurePaths(cfg.Runtime.DataDir, cfg.Runtime.ModelCache)
	if cfg.Runtime.Mode == "managed" {
		activeBundle := runtimeController.ActiveBundlePreferred(context.Background(), cfg.Runtime.ActiveBundle)
		if cfg.Runtime.AutoStart {
			activeBundle = cfg.Runtime.Bundle
		} else if activeBundle == "" {
			activeBundle = cfg.Runtime.ActiveBundle
		}
		cfg.Runtime.ActiveBundle = activeBundle
		cfg.Normalize()
		client = llm.New(cfg.Model.Endpoint, cfg.Model.DefaultModel, cfg.Model.APIKey, cfg.Model.ModelType).WithThinkingBudget(cfg.Model.ThinkingBudget)
	}
	s := &Server{cfg: cfg, startup: cfg.Server, configPath: configPath, db: store, llm: client, asr: asr.New(cfg.ASR), tts: tts.New(cfg.TTS), sshClient: supportssh.New(cfg.Extra.SSHEndpoint), media: mediaStore, knowledge: knowledgeStore, knowledgeIndex: &knowledge.Extractor{}, collector: knowledge.NewCollectorClient(cfg.Extra.CollectorEndpoint), runtime: runtimeController, contextWindows: make(map[string]int), approvals: make(map[string]*toolApproval), knowledgeJobs: make(map[string]*knowledgeJobRun), knowledgeJobSem: make(chan struct{}, 1), knowledgeOCRJobs: make(map[string]*knowledgeOCRRun), knowledgeOCRSem: make(chan struct{}, 1)}
	s.browser, err = browserbridge.New(cfg.Server.Database + ".browser-key")
	if err != nil {
		return nil, err
	}
	s.plugins, err = plugins.New(context.Background(), store, s.pluginServices(), plugins.Builtins())
	if err != nil {
		return nil, fmt.Errorf("plugin runtime: %w", err)
	}
	initialized := false
	defer func() {
		if !initialized {
			s.tasks.Stop()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			_ = s.plugins.Close(ctx)
			_ = s.tasks.Wait(ctx)
		}
	}()
	if err = s.plugins.OpenPackages(context.Background(), cfg.Server.Database+".plugins"); err != nil {
		return nil, fmt.Errorf("plugin packages: %w", err)
	}
	mux := s.routes(web)
	s.server = &http.Server{Addr: cfg.Server.ListenAddr, Handler: mux, BaseContext: func(net.Listener) context.Context { return s.tasks.Context() }, ReadHeaderTimeout: 10 * time.Second}
	if err := s.db.RecoverKnowledgeOCR(); err != nil {
		return nil, fmt.Errorf("recover knowledge OCR: %w", err)
	}
	if recovered, recoverErr := s.db.RecoverKnowledgeJobs(); recoverErr == nil {
		for _, job := range recovered {
			s.scheduleKnowledgeJob(job.ID)
		}
	}
	if cfg.Runtime.Mode == "managed" && cfg.Runtime.AutoStart {
		_ = s.runtime.StartBundle(context.Background(), cfg.Runtime.Bundle, cfg.Runtime.MemoryReserveGiB)
	}
	initialized = true
	return s, nil
}

func (s *Server) Shutdown(ctx context.Context) error {
	s.tasks.Stop()
	var httpErr, pluginErr error
	if s.server != nil {
		httpErr = s.server.Shutdown(ctx)
	}
	if s.plugins != nil {
		pluginErr = s.plugins.Close(ctx)
	}
	return errors.Join(httpErr, pluginErr, s.tasks.Wait(ctx))
}
func (s *Server) ListenAndServe() error {
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = s.Shutdown(ctx)
	}()
	ctx, finish, err := s.tasks.Track(context.Background())
	if err != nil {
		return err
	}
	go func() { defer finish(); s.runKeySync(ctx) }()
	return s.server.ListenAndServe()
}

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

func (s *Server) sshSnapshot() *supportssh.Client {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.sshClient
}

func (s *Server) collectorSnapshot() *knowledge.CollectorClient {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.collector
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
