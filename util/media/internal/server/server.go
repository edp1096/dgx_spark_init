package server

import (
	"context"
	"io/fs"
	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
	"net/http"
	"sync"
	"time"
)

type Server struct {
	cfgMu                sync.RWMutex
	heavyMu              sync.Mutex
	videoPreviewMu       sync.Mutex
	cfg                  config.Config
	configPath           string
	dataDir              string
	jobs                 *jobs.Store
	client               *http.Client
	health               *http.Client
	web                  fs.FS
	systemMu             sync.Mutex
	systemStats          systemUsage
	systemStatsAt        time.Time
	cpuPrevTotal         uint64
	cpuPrevIdle          uint64
	subtitleQueueOnce    sync.Once
	subtitleQueueWake    chan struct{}
	generationQueueOnce  sync.Once
	generationQueueWake  chan struct{}
	generationStateMu    sync.Mutex
	generationCancelMu   sync.Mutex
	generationCancels    map[string]context.CancelFunc
	engineDrainMu        sync.Mutex
	engineDraining       map[string]bool
	runtimeCapabilityMu  sync.RWMutex
	runtimeCapabilities  map[string]bool
	wildcardMu           sync.Mutex
	wildcardMuse         []string
	wildcardMuseNoCamera []string
	wildcardStyles       []string
	portraitLabMu        sync.Mutex
}

func New(cfg config.Config, store *jobs.Store, web fs.FS, configPath ...string) *Server {
	cfg = config.Normalize(cfg)
	path := ""
	if len(configPath) > 0 {
		path = configPath[0]
	}
	return &Server{
		cfg: cfg, configPath: path, dataDir: cfg.DataDir, jobs: store,
		client: &http.Client{Timeout: 2 * time.Hour},
		health: &http.Client{Timeout: 2 * time.Second},
		web:    web, subtitleQueueWake: make(chan struct{}, 1), generationQueueWake: make(chan struct{}, 1),
		generationCancels:   make(map[string]context.CancelFunc),
		engineDraining:      make(map[string]bool),
		runtimeCapabilities: make(map[string]bool),
	}
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/health", func(w http.ResponseWriter, _ *http.Request) { writeJSON(w, 200, map[string]string{"status": "ok"}) })
	mux.HandleFunc("GET /api/config", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, 200, s.config())
	})
	mux.HandleFunc("PUT /api/config", s.updateConfig)
	mux.HandleFunc("GET /api/engines", s.engineStates)
	mux.HandleFunc("GET /api/system", s.systemUsage)
	mux.HandleFunc("GET /api/video/models", s.videoModelStatus)
	mux.HandleFunc("POST /api/video/models/prepare", s.prepareVideoModels)
	mux.HandleFunc("GET /api/image/checkpoints", s.imageCheckpointStatus)
	mux.HandleFunc("POST /api/image/checkpoints/prepare", s.prepareImageCheckpoints)
	mux.HandleFunc("POST /api/image/checkpoints/convert-nvfp4", s.convertImageCheckpointsNVFP4)
	mux.HandleFunc("GET /api/jobs", func(w http.ResponseWriter, _ *http.Request) { writeJSON(w, 200, s.jobs.List()) })
	mux.HandleFunc("DELETE /api/jobs", s.deleteFinishedJobs)
	mux.HandleFunc("GET /api/jobs/{id}", s.getJob)
	mux.HandleFunc("GET /api/jobs/{id}/exif", s.imageJobEXIF)
	mux.HandleFunc("GET /api/jobs/{id}/inputs", s.imageJobInputs)
	mux.HandleFunc("GET /api/jobs/{id}/inputs/{role}/{index}", s.imageJobInput)
	mux.HandleFunc("GET /api/jobs/{id}/video-preview.jpg", s.videoJobPreview)
	mux.HandleFunc("GET /api/jobs/{id}/frame", s.mediaJobFrame)
	mux.HandleFunc("DELETE /api/jobs/{id}", s.deleteJob)
	mux.HandleFunc("POST /api/jobs/{id}/cancel", s.cancelJob)
	mux.HandleFunc("POST /api/jobs/{id}/retry", s.retryJob)
	mux.HandleFunc("POST /api/jobs/image", s.createImage)
	mux.HandleFunc("POST /api/images/fetch", s.fetchRemoteImage)
	mux.HandleFunc("POST /api/jobs/{id}/upscale", s.createImageUpscale)
	mux.HandleFunc("POST /api/jobs/{id}/video-upscale", s.createVideoUpscale)
	mux.HandleFunc("POST /api/jobs/{id}/detail-enhance", s.createImageDetailEnhance)
	mux.HandleFunc("POST /api/jobs/garment-extract", s.createGarmentExtraction)
	mux.HandleFunc("POST /api/jobs/face-swap", s.createFaceSwap)
	mux.HandleFunc("POST /api/jobs/speech", s.createSpeech)
	mux.HandleFunc("POST /api/jobs/recognition", s.createSubtitle)
	mux.HandleFunc("POST /api/jobs/{id}/subtitle-regenerate", s.regenerateSubtitle)
	mux.HandleFunc("POST /api/media/options", s.mediaOptions)
	mux.HandleFunc("GET /api/storage", s.mediaStorage)
	mux.HandleFunc("DELETE /api/storage/temp", s.cleanupMediaTemp)
	mux.HandleFunc("POST /api/jobs/video", s.createVideo)
	mux.HandleFunc("POST /api/prompts/enhance", s.enhancePrompt)
	mux.HandleFunc("POST /api/prompts/character-description", s.describeImageSequenceCharacter)
	mux.HandleFunc("POST /api/images/character-sheet", s.createImageSequenceCharacterSheet)
	mux.HandleFunc("GET /api/images/character-sheet/status", s.imageSequenceCharacterSheetStatus)
	mux.HandleFunc("POST /api/prompts/sequence-plan", s.planImageSequence)
	mux.HandleFunc("GET /api/prompts/wildcard", s.randomPromptWildcard)
	mux.HandleFunc("GET /tools/portrait-lab", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/tools/portrait-lab/", http.StatusTemporaryRedirect)
	})
	mux.HandleFunc("GET /tools/portrait-lab/", s.servePortraitLab)
	mux.HandleFunc("POST /api/assistant/chat", s.assistantChat)
	mux.HandleFunc("/api/lora", s.proxyLoRA)
	mux.HandleFunc("/api/lora/", s.proxyLoRA)
	mux.HandleFunc("GET /api/media/assets/{id}", s.proxyMediaAsset)
	mux.HandleFunc("HEAD /api/media/assets/{id}", s.proxyMediaAsset)
	mux.Handle("GET /api/outputs/", http.StripPrefix("/api/outputs/", http.FileServer(http.Dir(s.jobs.OutputDir()))))
	if s.web != nil {
		mux.Handle("/", spaHandler(s.web))
	}
	return withLog(mux)
}

func (s *Server) config() config.Config {
	s.cfgMu.RLock()
	defer s.cfgMu.RUnlock()
	return s.cfg
}
