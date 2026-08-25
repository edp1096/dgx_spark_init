package server

import (
	"bytes"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"
	"io"
	"io/fs"
	"log"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"

	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
	mediaprompt "mediaapp/internal/prompt"
)

type Server struct {
	cfgMu               sync.RWMutex
	heavyMu             sync.Mutex
	videoPreviewMu      sync.Mutex
	cfg                 config.Config
	configPath          string
	dataDir             string
	jobs                *jobs.Store
	client              *http.Client
	health              *http.Client
	web                 fs.FS
	systemMu            sync.Mutex
	systemStats         systemUsage
	systemStatsAt       time.Time
	cpuPrevTotal        uint64
	cpuPrevIdle         uint64
	subtitleQueueOnce   sync.Once
	subtitleQueueWake   chan struct{}
	generationQueueOnce sync.Once
	generationQueueWake chan struct{}
	generationStateMu   sync.Mutex
	wildcardMu          sync.Mutex
	wildcardMuse        []string
	wildcardStyles      []string
	portraitLabMu       sync.Mutex
}

type imageGenerationOptions struct {
	checkpoint         string
	identityPath       string
	identityRefPaths   []string
	identityPreset     string
	identityAutoPrompt bool
	identityUserPrompt bool
	identityMaskPath   string
	strictMaskPath     string
	strictMaskGrow     int
	strictMaskFeather  float64
	vaeMode            string
	identityFitMode    string
	identityModel      string
	identityEncoder    string
	depthPath          string
	depthPrompt        string
	preparePoseRef     bool
	identityStrength   float64
	refBoost           float64
	sourceRefBoost     float64
	groundingPX        int
	steps              int
	samplingPreset     string
	sampler            string
	scheduler          string
	style              string
	styleStrength      float64
	styles             []styleSelection
	userLoras          []userLoRASelection
	depthStrength      float64
	visionPaths        []string
	visionMode         string
	visionMegapixels   float64
	styleRefPaths      []string
	styleRefStrength   float64
	nk2ePath           string
	nk2eMode           string
	nk2eStrength       float64
	nk2ePreprocessed   bool
	anypaintPath       string
	anypaintMaskPath   string
	outpaintLeft       int
	outpaintTop        int
	outpaintRight      int
	outpaintBottom     int
	anypaintStrength   float64
	anypaintBoundary   int
	filterMode         string
	filterStrength     float64
	promptEnhancer     bool
	promptEnhStrength  float64
	promptTextScale    float64
}

type videoConditioningInput struct {
	Path     string
	FrameIdx int
	Strength float64
	Role     string
}

type savedVideoCondition struct {
	Role     string  `json:"role"`
	Index    int     `json:"index,omitempty"`
	FrameIdx int     `json:"frame_idx"`
	Strength float64 `json:"strength"`
}

type styleSelection struct {
	Name     string  `json:"name"`
	Strength float64 `json:"strength"`
}

type userLoRASelection struct {
	Filename string  `json:"filename"`
	Strength float64 `json:"strength"`
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
	mux.HandleFunc("DELETE /api/jobs/{id}", s.deleteJob)
	mux.HandleFunc("POST /api/jobs/{id}/cancel", s.cancelJob)
	mux.HandleFunc("POST /api/jobs/{id}/retry", s.retryJob)
	mux.HandleFunc("POST /api/jobs/image", s.createImage)
	mux.HandleFunc("POST /api/images/fetch", s.fetchRemoteImage)
	mux.HandleFunc("POST /api/jobs/{id}/upscale", s.createImageUpscale)
	mux.HandleFunc("POST /api/jobs/{id}/detail-enhance", s.createImageDetailEnhance)
	mux.HandleFunc("POST /api/jobs/garment-extract", s.createGarmentExtraction)
	mux.HandleFunc("POST /api/jobs/speech", s.createSpeech)
	mux.HandleFunc("POST /api/jobs/recognition", s.createSubtitle)
	mux.HandleFunc("POST /api/media/options", s.mediaOptions)
	mux.HandleFunc("GET /api/storage", s.mediaStorage)
	mux.HandleFunc("DELETE /api/storage/temp", s.cleanupMediaTemp)
	mux.HandleFunc("POST /api/jobs/video", s.createVideo)
	mux.HandleFunc("POST /api/prompts/enhance", s.enhancePrompt)
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

func (s *Server) videoModelStatus(w http.ResponseWriter, r *http.Request) {
	endpoint := s.config().Engines["video"].Endpoint
	req, err := http.NewRequestWithContext(r.Context(), http.MethodGet, endpoint+"/v1/models/status", nil)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	resp, err := s.health.Do(req)
	if err != nil {
		http.Error(w, "LTX model service unavailable: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, io.LimitReader(resp.Body, 1<<20))
}

func (s *Server) prepareVideoModels(w http.ResponseWriter, r *http.Request) {
	var request struct {
		HFToken string `json:"hf_token"`
	}
	decoder := json.NewDecoder(io.LimitReader(r.Body, 16<<10))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&request); err != nil {
		http.Error(w, "invalid model preparation request", http.StatusBadRequest)
		return
	}
	payload := map[string]string{"hf_token": strings.TrimSpace(request.HFToken)}
	data, _, err := s.callJSON(s.config().Engines["video"].Endpoint+"/v1/models/prepare", payload)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	_, _ = w.Write(data)
}

func (s *Server) imageCheckpointStatus(w http.ResponseWriter, r *http.Request) {
	endpoint := s.config().Engines["image"].Endpoint
	req, err := http.NewRequestWithContext(r.Context(), http.MethodGet, endpoint+"/v1/checkpoints/status", nil)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	resp, err := s.health.Do(req)
	if err != nil {
		http.Error(w, "Krea model service unavailable: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, io.LimitReader(resp.Body, 1<<20))
}

func (s *Server) prepareImageCheckpoints(w http.ResponseWriter, r *http.Request) {
	var request struct {
		CivitaiToken string   `json:"civitai_token"`
		HFToken      string   `json:"hf_token"`
		Variants     []string `json:"variants"`
	}
	decoder := json.NewDecoder(io.LimitReader(r.Body, 32<<10))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&request); err != nil {
		http.Error(w, "invalid checkpoint preparation request", http.StatusBadRequest)
		return
	}
	payload := map[string]any{
		"civitai_token": strings.TrimSpace(request.CivitaiToken),
		"hf_token":      strings.TrimSpace(request.HFToken),
		"variants":      request.Variants,
	}
	data, _, err := s.callJSON(s.config().Engines["image"].Endpoint+"/v1/checkpoints/prepare", payload)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	_, _ = w.Write(data)
}

func (s *Server) convertImageCheckpointsNVFP4(w http.ResponseWriter, r *http.Request) {
	var request struct {
		CivitaiToken     string   `json:"civitai_token"`
		Variants         []string `json:"variants"`
		RemoveBF16Source bool     `json:"remove_bf16_sources"`
	}
	decoder := json.NewDecoder(io.LimitReader(r.Body, 32<<10))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&request); err != nil {
		http.Error(w, "invalid NVFP4 conversion request", http.StatusBadRequest)
		return
	}
	payload := map[string]any{
		"civitai_token":       strings.TrimSpace(request.CivitaiToken),
		"variants":            request.Variants,
		"remove_bf16_sources": request.RemoveBF16Source,
	}
	data, _, err := s.callJSON(s.config().Engines["image"].Endpoint+"/v1/checkpoints/convert-nvfp4", payload)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	_, _ = w.Write(data)
}

func (s *Server) config() config.Config {
	s.cfgMu.RLock()
	defer s.cfgMu.RUnlock()
	return s.cfg
}

func (s *Server) proxyLoRA(w http.ResponseWriter, r *http.Request) {
	endpoint := s.config().Engines["image"].Endpoint
	target, err := url.Parse(endpoint)
	if err != nil || target.Host == "" {
		http.Error(w, "invalid LoRA manager endpoint", http.StatusBadGateway)
		return
	}
	proxy := httputil.NewSingleHostReverseProxy(target)
	proxy.ErrorHandler = func(w http.ResponseWriter, _ *http.Request, err error) {
		http.Error(w, "LoRA manager unavailable: "+err.Error(), http.StatusBadGateway)
	}
	originalDirector := proxy.Director
	proxy.Director = func(request *http.Request) {
		originalDirector(request)
		path := strings.TrimPrefix(request.URL.Path, "/api/lora")
		request.URL.Path = "/v1/user-loras" + path
		request.URL.RawPath = ""
	}
	proxy.ServeHTTP(w, r)
}

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

func (s *Server) createVideo(w http.ResponseWriter, r *http.Request) {
	cfg := s.config()
	if err := r.ParseMultipartForm(320 << 20); err != nil {
		http.Error(w, "invalid or oversized form", http.StatusBadRequest)
		return
	}
	effectivePrompt := strings.TrimSpace(r.FormValue("prompt"))
	if effectivePrompt == "" {
		http.Error(w, "prompt is required", http.StatusBadRequest)
		return
	}
	originalPrompt := strings.TrimSpace(r.FormValue("original_prompt"))
	if originalPrompt == "" {
		originalPrompt = effectivePrompt
	}
	width := formInt(r, "width", cfg.Video.DefaultWidth)
	height := formInt(r, "height", cfg.Video.DefaultHeight)
	frames := formInt(r, "num_frames", cfg.Video.DefaultFrames)
	fps := formFloat64(r, "fps", cfg.Video.DefaultFPS)
	seed := formInt64(r, "seed", -1)
	strength := formFloat64(r, "image_strength", 1)
	if width < 256 || height < 256 || width%64 != 0 || height%64 != 0 {
		http.Error(w, "width and height must be >= 256 and divisible by 64", http.StatusBadRequest)
		return
	}
	if frames < 9 || (frames-1)%8 != 0 {
		http.Error(w, "num_frames must be 8*k+1 and at least 9", http.StatusBadRequest)
		return
	}
	id := newID()
	inputDir := filepath.Join(s.dataDir, "inputs", id)
	loadImage := func(uploadField, reuseField, dir string) (string, error) {
		paths, err := saveUploads(r, uploadField, dir, 1)
		if err != nil {
			return "", err
		}
		paths, err = s.appendReusedImageInputs(r, reuseField, dir, 1, paths)
		if err != nil {
			return "", err
		}
		if len(paths) == 0 {
			return "", nil
		}
		return paths[0], nil
	}

	startPath, err := loadImage("start_image", "reuse_start_image", filepath.Join(inputDir, "start"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	// Keep the original single-image API compatible with older clients.
	if startPath == "" {
		legacy, legacyErr := saveUploads(r, "image", filepath.Join(inputDir, "start"), 1)
		if legacyErr != nil {
			http.Error(w, legacyErr.Error(), http.StatusBadRequest)
			return
		}
		if len(legacy) > 0 {
			startPath = legacy[0]
		}
	}
	endPath, err := loadImage("end_image", "reuse_end_image", filepath.Join(inputDir, "end"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	conditions := make([]videoConditioningInput, 0, 10)
	usedFrames := map[int]bool{}
	if startPath != "" {
		conditions = append(conditions, videoConditioningInput{Path: startPath, FrameIdx: 0, Strength: strength, Role: "start"})
		usedFrames[0] = true
	}
	keyframeCount := formInt(r, "keyframe_count", 0)
	if keyframeCount < 0 || keyframeCount > 8 {
		http.Error(w, "keyframe_count must be between 0 and 8", http.StatusBadRequest)
		return
	}
	for i := 0; i < keyframeCount; i++ {
		keyframePath, loadErr := loadImage(
			fmt.Sprintf("keyframe_image_%d", i),
			fmt.Sprintf("reuse_keyframe_image_%d", i),
			filepath.Join(inputDir, fmt.Sprintf("keyframe-%d", i)),
		)
		if loadErr != nil {
			http.Error(w, loadErr.Error(), http.StatusBadRequest)
			return
		}
		if keyframePath == "" {
			http.Error(w, fmt.Sprintf("keyframe %d image is required", i+1), http.StatusBadRequest)
			return
		}
		seconds := formFloat64(r, fmt.Sprintf("keyframe_time_%d", i), -1)
		frameIdx := int(seconds*fps + 0.5)
		if seconds < 0 || frameIdx <= 0 || frameIdx >= frames-1 {
			http.Error(w, fmt.Sprintf("keyframe %d time must be between the start and end frames", i+1), http.StatusBadRequest)
			return
		}
		if usedFrames[frameIdx] {
			http.Error(w, fmt.Sprintf("keyframe %d overlaps another conditioning frame", i+1), http.StatusBadRequest)
			return
		}
		keyframeStrength := formFloat64(r, fmt.Sprintf("keyframe_strength_%d", i), 1)
		if keyframeStrength < 0 || keyframeStrength > 1 {
			http.Error(w, fmt.Sprintf("keyframe %d strength must be between 0 and 1", i+1), http.StatusBadRequest)
			return
		}
		conditions = append(conditions, videoConditioningInput{Path: keyframePath, FrameIdx: frameIdx, Strength: keyframeStrength, Role: "keyframe"})
		usedFrames[frameIdx] = true
	}
	if endPath != "" {
		conditions = append(conditions, videoConditioningInput{Path: endPath, FrameIdx: frames - 1, Strength: formFloat64(r, "end_image_strength", 1), Role: "end"})
	}
	for _, condition := range conditions {
		if condition.Strength < 0 || condition.Strength > 1 {
			http.Error(w, condition.Role+" image strength must be between 0 and 1", http.StatusBadRequest)
			return
		}
	}
	savedConditions := make([]savedVideoCondition, 0, len(conditions))
	keyframeIndex := 0
	for _, condition := range conditions {
		saved := savedVideoCondition{Role: condition.Role, FrameIdx: condition.FrameIdx, Strength: condition.Strength}
		if condition.Role == "keyframe" {
			saved.Index = keyframeIndex
			keyframeIndex++
		}
		savedConditions = append(savedConditions, saved)
	}
	j := jobs.Job{
		ID: id, Kind: "video", Status: "queued", Prompt: originalPrompt,
		Params:    map[string]any{"width": width, "height": height, "num_frames": frames, "fps": fps, "seed": seed, "image_strength": strength, "image": len(conditions) > 0, "start_image": startPath != "", "end_image": endPath != "", "keyframes": keyframeCount, "video_conditions": savedConditions, "motion_lora_enabled": cfg.Video.DefaultMotionLoRAEnabled, "motion_lora_strength": cfg.Video.DefaultMotionLoRAStrength, "enhanced_prompt": valueIfDifferent(effectivePrompt, originalPrompt), "stage": "queued", "queued_at": time.Now().Format(time.RFC3339Nano)},
		CreatedAt: time.Now(),
	}
	if err := s.jobs.Save(j); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	s.wakeGenerationQueue()
	writeJSON(w, http.StatusAccepted, j)
}

func (s *Server) createImage(w http.ResponseWriter, r *http.Request) {
	cfg := s.config()
	if err := r.ParseMultipartForm(80 << 20); err != nil {
		http.Error(w, "invalid form", 400)
		return
	}
	sequencePrompts, err := parseImageSequencePrompts(r.FormValue("sequence_prompts"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	sequenceRegions, err := parseImageSequenceRegions(r.FormValue("sequence_regions"), len(sequencePrompts))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	sequenceIdentityStrength := formFloat64(r, "sequence_identity_strength", 0.8)
	if len(sequencePrompts) > 0 && (sequenceIdentityStrength < 0 || sequenceIdentityStrength > 2) {
		http.Error(w, "sequence identity strength must be between 0 and 2", http.StatusBadRequest)
		return
	}
	effectivePrompt := strings.TrimSpace(r.FormValue("prompt"))
	if len(sequencePrompts) > 0 {
		effectivePrompt = sequencePrompts[0]
	}
	if effectivePrompt == "" {
		outpaintPadding := formInt(r, "outpaint_left", 0) + formInt(r, "outpaint_top", 0) + formInt(r, "outpaint_right", 0) + formInt(r, "outpaint_bottom", 0)
		hasAnyPaintSource := len(r.MultipartForm.File["anypaint_image"]) > 0 || len(r.MultipartForm.Value["reuse_anypaint_image"]) > 0
		hasAnyPaintMask := len(r.MultipartForm.File["anypaint_mask"]) > 0 || len(r.MultipartForm.Value["reuse_anypaint_mask"]) > 0
		if outpaintPadding > 0 && hasAnyPaintSource && !hasAnyPaintMask {
			effectivePrompt = "Extend the original image naturally into a complete, coherent composition while preserving its subjects, style, lighting, perspective, and visual continuity."
		} else {
			http.Error(w, "prompt is required", 400)
			return
		}
	}
	originalPrompt := strings.TrimSpace(r.FormValue("original_prompt"))
	if originalPrompt == "" {
		originalPrompt = effectivePrompt
	}
	id := newID()
	inputDir := filepath.Join(s.dataDir, "inputs", id)
	refs, err := saveUploads(r, "references", inputDir, cfg.Image.MaxReferenceImages)
	if err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	refs, err = s.appendReusedImageInputs(r, "reuse_references", inputDir, cfg.Image.MaxReferenceImages, refs)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	width := formInt(r, "width", cfg.Image.DefaultWidth)
	height := formInt(r, "height", cfg.Image.DefaultHeight)
	seed := formInt64(r, "seed", -1)
	mode := strings.ToLower(strings.TrimSpace(r.FormValue("mode")))
	if mode == "" {
		mode = cfg.Image.DefaultMode
	}
	if _, ok := cfg.Image.Backends[mode]; !ok {
		http.Error(w, "unsupported image mode", http.StatusBadRequest)
		return
	}
	controlType := strings.ToLower(strings.TrimSpace(r.FormValue("control_type")))
	controlStrength := formFloat64(r, "control_strength", 0.65)
	krea := imageGenerationOptions{
		checkpoint:         strings.ToLower(strings.TrimSpace(r.FormValue("checkpoint"))),
		identityAutoPrompt: strings.EqualFold(r.FormValue("identity_auto_prompt"), "true"),
		identityUserPrompt: strings.EqualFold(r.FormValue("identity_user_prompt"), "true"),
		identityStrength:   formFloat64(r, "identity_strength", 1),
		refBoost:           formFloat64(r, "ref_boost", 4),
		sourceRefBoost:     formFloat64(r, "source_ref_boost", 1),
		groundingPX:        formInt(r, "grounding_px", 768),
		steps:              formInt(r, "steps", 0),
		samplingPreset:     strings.ToLower(strings.TrimSpace(r.FormValue("sampling_preset"))),
		style:              strings.ToLower(strings.TrimSpace(r.FormValue("style"))),
		styleStrength:      formFloat64(r, "style_strength", 1),
		depthStrength:      formFloat64(r, "depth_strength", 0.8),
		depthPrompt:        strings.TrimSpace(r.FormValue("depth_pose_prompt")),
		preparePoseRef:     strings.EqualFold(r.FormValue("prepare_pose_reference"), "true"),
		visionMode:         strings.ToLower(strings.TrimSpace(r.FormValue("vision_mode"))),
		visionMegapixels:   formFloat64(r, "vision_megapixels", 1),
		styleRefStrength:   formFloat64(r, "style_reference_strength", 1),
		nk2eMode:           strings.ToLower(strings.TrimSpace(r.FormValue("nk2e_mode"))),
		nk2eStrength:       formFloat64(r, "nk2e_strength", 0.7),
		outpaintLeft:       formInt(r, "outpaint_left", 0),
		outpaintTop:        formInt(r, "outpaint_top", 0),
		outpaintRight:      formInt(r, "outpaint_right", 0),
		outpaintBottom:     formInt(r, "outpaint_bottom", 0),
		anypaintStrength:   formFloat64(r, "anypaint_strength", 1),
		anypaintBoundary:   formInt(r, "anypaint_boundary_redraw_px", 32),
		strictMaskGrow:     formInt(r, "strict_mask_grow", 0),
		strictMaskFeather:  formFloat64(r, "strict_mask_feather", 0),
		vaeMode:            strings.ToLower(strings.TrimSpace(r.FormValue("vae_mode"))),
		identityFitMode:    strings.ToLower(strings.TrimSpace(r.FormValue("identity_fit_mode"))),
		identityModel:      strings.ToLower(strings.TrimSpace(r.FormValue("identity_model"))),
		identityEncoder:    strings.ToLower(strings.TrimSpace(r.FormValue("identity_encoder"))),
		nk2ePreprocessed:   strings.EqualFold(r.FormValue("nk2e_preprocessed"), "true"),
		filterMode:         strings.ToLower(strings.TrimSpace(r.FormValue("filter_mode"))),
		filterStrength:     formFloat64(r, "filter_strength", 1),
		promptEnhancer:     strings.EqualFold(r.FormValue("prompt_enhancer"), "true"),
		promptEnhStrength:  formFloat64(r, "prompt_enhancer_strength", 1),
		promptTextScale:    formFloat64(r, "prompt_text_scale", 1.75),
	}
	identityPreset := strings.TrimSpace(r.FormValue("identity_preset"))
	validIdentityPresets := map[string]bool{"": true, "restage": true, "sheet": true, "faceSwap": true, "headSwap": true, "personSwap": true, "tryon": true, "replace": true}
	if !validIdentityPresets[identityPreset] {
		http.Error(w, "unsupported identity preset", http.StatusBadRequest)
		return
	}
	krea.identityPreset = identityPreset
	identityPreserveItems := []string{}
	if raw := strings.TrimSpace(r.FormValue("identity_preserve_items")); raw != "" {
		if err := json.Unmarshal([]byte(raw), &identityPreserveItems); err != nil {
			http.Error(w, "invalid identity preservation selection", http.StatusBadRequest)
			return
		}
		allowed := map[string]bool{"identity": true, "face": true, "hair": true, "body": true, "clothing": true, "pose": true, "background": true, "lighting": true, "composition": true, "untouched": true}
		for _, item := range identityPreserveItems {
			if !allowed[item] {
				http.Error(w, "invalid identity preservation item", http.StatusBadRequest)
				return
			}
		}
	}
	identityPreserveCustom := strings.TrimSpace(r.FormValue("identity_preserve_custom"))
	if len(identityPreserveCustom) > 500 {
		http.Error(w, "identity custom preservation text is too long", http.StatusBadRequest)
		return
	}
	if krea.checkpoint == "" {
		krea.checkpoint = cfg.Image.DefaultCheckpoint
	}
	validCheckpoints := map[string]bool{
		"official": true, "ray-v1": true, "ray-v2": true, "ray-v2-nvfp4": true,
		"ray-v3": true, "ray-v4": true, "ray-v4-nvfp4": true,
		"moody-v7": true, "moody-cutie-v4": true, "moody-amateur-v1": true,
		"chriscole-edit-v1.1": true,
	}
	if !validCheckpoints[krea.checkpoint] {
		http.Error(w, "unsupported Krea checkpoint", http.StatusBadRequest)
		return
	}
	if krea.vaeMode == "" {
		krea.vaeMode = "default"
	}
	if krea.identityFitMode == "" {
		krea.identityFitMode = "fit"
	}
	if krea.identityModel == "" {
		krea.identityModel = "convrot"
	}
	if krea.identityEncoder == "" {
		krea.identityEncoder = "heretic"
	}
	if krea.identityModel != "selected" && krea.identityModel != "convrot" {
		http.Error(w, "identity model must be selected or convrot", http.StatusBadRequest)
		return
	}
	if krea.identityEncoder != "default" && krea.identityEncoder != "heretic" {
		http.Error(w, "identity encoder must be default or heretic", http.StatusBadRequest)
		return
	}
	if krea.filterMode == "" {
		if krea.checkpoint == "official" {
			krea.filterMode = "balanced"
		} else {
			krea.filterMode = "off"
		}
	}
	if krea.checkpoint != "official" && krea.filterMode != "off" {
		http.Error(w, "third-party checkpoints already include tuning; select original filter mode", http.StatusBadRequest)
		return
	}
	if krea.samplingPreset == "" {
		krea.samplingPreset = "default"
	}
	switch krea.samplingPreset {
	case "default":
		krea.sampler, krea.scheduler = "euler", "simple"
	case "detail":
		krea.sampler, krea.scheduler = "er_sde", "simple"
	case "moody":
		krea.sampler, krea.scheduler = "euler_ancestral", "beta"
	default:
		http.Error(w, "sampling preset must be default, detail, or moody", http.StatusBadRequest)
		return
	}
	if rawStyles := strings.TrimSpace(r.FormValue("styles")); rawStyles != "" {
		if err := json.Unmarshal([]byte(rawStyles), &krea.styles); err != nil {
			http.Error(w, "invalid Krea styles", http.StatusBadRequest)
			return
		}
	} else if krea.style != "" {
		krea.styles = []styleSelection{{Name: krea.style, Strength: krea.styleStrength}}
	}
	if rawUserLoras := strings.TrimSpace(r.FormValue("user_loras")); rawUserLoras != "" {
		if err := json.Unmarshal([]byte(rawUserLoras), &krea.userLoras); err != nil {
			http.Error(w, "invalid user LoRA selection", http.StatusBadRequest)
			return
		}
	}
	if krea.visionMode == "" {
		krea.visionMode = "descriptor"
	}
	if krea.nk2eMode == "" {
		krea.nk2eMode = "edit"
	}
	switch mode {
	case "create":
		if len(refs) != 0 {
			http.Error(w, "high quality generation does not accept reference images", http.StatusBadRequest)
			return
		}
		identity, uploadErr := saveUploads(r, "identity_image", filepath.Join(inputDir, "identity"), 1)
		if uploadErr != nil {
			http.Error(w, uploadErr.Error(), http.StatusBadRequest)
			return
		}
		identity, uploadErr = s.appendReusedImageInputs(r, "reuse_identity_image", filepath.Join(inputDir, "identity"), 1, identity)
		if uploadErr != nil {
			http.Error(w, uploadErr.Error(), http.StatusBadRequest)
			return
		}
		identityRef, uploadErr := saveUploads(r, "identity_reference", filepath.Join(inputDir, "identity-reference"), 3)
		if uploadErr != nil {
			http.Error(w, uploadErr.Error(), http.StatusBadRequest)
			return
		}
		identityRef, uploadErr = s.appendReusedImageInputs(r, "reuse_identity_reference", filepath.Join(inputDir, "identity-reference"), 3, identityRef)
		if uploadErr != nil {
			http.Error(w, uploadErr.Error(), http.StatusBadRequest)
			return
		}
		depth, uploadErr := saveUploads(r, "depth_image", filepath.Join(inputDir, "depth"), 1)
		if uploadErr != nil {
			http.Error(w, uploadErr.Error(), http.StatusBadRequest)
			return
		}
		depth, uploadErr = s.appendReusedImageInputs(r, "reuse_depth_image", filepath.Join(inputDir, "depth"), 1, depth)
		if uploadErr != nil {
			http.Error(w, uploadErr.Error(), http.StatusBadRequest)
			return
		}
		if len(identity) > 0 {
			krea.identityPath = identity[0]
		}
		krea.identityRefPaths = identityRef
		identityMask, uploadErr := saveUploads(r, "identity_mask", filepath.Join(inputDir, "identity-mask"), 1)
		if uploadErr != nil {
			http.Error(w, uploadErr.Error(), http.StatusBadRequest)
			return
		}
		identityMask, uploadErr = s.appendReusedImageInputs(r, "reuse_identity_mask", filepath.Join(inputDir, "identity-mask"), 1, identityMask)
		if uploadErr != nil {
			http.Error(w, uploadErr.Error(), http.StatusBadRequest)
			return
		}
		if len(identityMask) > 0 {
			krea.identityMaskPath = identityMask[0]
		}
		strictMask, uploadErr := saveUploads(r, "strict_mask", filepath.Join(inputDir, "strict-mask"), 1)
		if uploadErr != nil {
			http.Error(w, uploadErr.Error(), http.StatusBadRequest)
			return
		}
		strictMask, uploadErr = s.appendReusedImageInputs(r, "reuse_strict_mask", filepath.Join(inputDir, "strict-mask"), 1, strictMask)
		if uploadErr != nil {
			http.Error(w, uploadErr.Error(), http.StatusBadRequest)
			return
		}
		if len(strictMask) > 0 {
			krea.strictMaskPath = strictMask[0]
		}
		if len(depth) > 0 {
			krea.depthPath = depth[0]
		}
		// The pose/structure field is semantically different from a photographic
		// Identity Edit reference. Let the Krea service prepare it regardless of
		// whether the currently open browser supplied newer frontend metadata.
		if krea.identityPath != "" && krea.depthPath != "" {
			krea.preparePoseRef = true
		}
		vision, uploadErr := saveUploads(r, "vision_images", filepath.Join(inputDir, "vision"), 4)
		if uploadErr != nil {
			http.Error(w, uploadErr.Error(), http.StatusBadRequest)
			return
		}
		vision, uploadErr = s.appendReusedImageInputs(r, "reuse_vision_images", filepath.Join(inputDir, "vision"), 4, vision)
		if uploadErr != nil {
			http.Error(w, uploadErr.Error(), http.StatusBadRequest)
			return
		}
		krea.visionPaths = vision
		styleRefs, uploadErr := saveUploads(r, "style_reference_images", filepath.Join(inputDir, "style-reference"), 2)
		if uploadErr != nil {
			http.Error(w, uploadErr.Error(), http.StatusBadRequest)
			return
		}
		styleRefs, uploadErr = s.appendReusedImageInputs(r, "reuse_style_reference_images", filepath.Join(inputDir, "style-reference"), 2, styleRefs)
		if uploadErr != nil {
			http.Error(w, uploadErr.Error(), http.StatusBadRequest)
			return
		}
		krea.styleRefPaths = styleRefs
		nk2e, uploadErr := saveUploads(r, "nk2e_image", filepath.Join(inputDir, "nk2e"), 1)
		if uploadErr != nil {
			http.Error(w, uploadErr.Error(), http.StatusBadRequest)
			return
		}
		nk2e, uploadErr = s.appendReusedImageInputs(r, "reuse_nk2e_image", filepath.Join(inputDir, "nk2e"), 1, nk2e)
		if uploadErr != nil {
			http.Error(w, uploadErr.Error(), http.StatusBadRequest)
			return
		}
		if len(nk2e) > 0 {
			krea.nk2ePath = nk2e[0]
		}
		anypaint, uploadErr := saveUploads(r, "anypaint_image", filepath.Join(inputDir, "anypaint"), 1)
		if uploadErr != nil {
			http.Error(w, uploadErr.Error(), http.StatusBadRequest)
			return
		}
		anypaint, uploadErr = s.appendReusedImageInputs(r, "reuse_anypaint_image", filepath.Join(inputDir, "anypaint"), 1, anypaint)
		if uploadErr != nil {
			http.Error(w, uploadErr.Error(), http.StatusBadRequest)
			return
		}
		anypaintMask, uploadErr := saveUploads(r, "anypaint_mask", filepath.Join(inputDir, "anypaint-mask"), 1)
		if uploadErr != nil {
			http.Error(w, uploadErr.Error(), http.StatusBadRequest)
			return
		}
		anypaintMask, uploadErr = s.appendReusedImageInputs(r, "reuse_anypaint_mask", filepath.Join(inputDir, "anypaint-mask"), 1, anypaintMask)
		if uploadErr != nil {
			http.Error(w, uploadErr.Error(), http.StatusBadRequest)
			return
		}
		if len(anypaint) > 0 {
			krea.anypaintPath = anypaint[0]
		}
		if len(anypaintMask) > 0 {
			krea.anypaintMaskPath = anypaintMask[0]
		}
		if krea.steps == 0 {
			krea.steps = 8
			if krea.identityPath != "" {
				krea.steps = 10
			}
		}
		if len(krea.identityRefPaths) > 0 && krea.identityPath == "" {
			http.Error(w, "a primary identity image is required before an additional reference", http.StatusBadRequest)
			return
		}
		if (krea.identityMaskPath != "" || krea.strictMaskPath != "") && krea.identityPath == "" {
			http.Error(w, "identity masks require a primary identity image", http.StatusBadRequest)
			return
		}
		if krea.strictMaskGrow < 0 || krea.strictMaskGrow > 128 || krea.strictMaskFeather < 0 || krea.strictMaskFeather > 128 {
			http.Error(w, "strict mask grow and feather must be between 0 and 128", http.StatusBadRequest)
			return
		}
		if krea.vaeMode != "default" && krea.vaeMode != "wan" && krea.vaeMode != "real" {
			http.Error(w, "VAE mode must be default, wan, or real", http.StatusBadRequest)
			return
		}
		if krea.identityFitMode != "fit" && krea.identityFitMode != "crop" {
			http.Error(w, "identity fit mode must be fit or crop", http.StatusBadRequest)
			return
		}
		if krea.filterMode != "off" && krea.filterMode != "adherence" && krea.filterMode != "balanced" && krea.filterMode != "strong" {
			http.Error(w, "filter mode must be off, adherence, balanced, or strong", http.StatusBadRequest)
			return
		}
		if krea.filterStrength < 0 || krea.filterStrength > 10 || krea.promptEnhStrength < 0 || krea.promptEnhStrength > 2 || krea.promptTextScale < 0.25 || krea.promptTextScale > 4 {
			http.Error(w, "invalid Krea filter or prompt adherence settings", http.StatusBadRequest)
			return
		}
		validStyles := map[string]bool{"darkbrush": true, "dotmatrix": true, "kidsdrawing": true, "neondrip": true, "rainywindow": true, "retroanime": true, "softwatercolor": true, "sunsetblur": true, "vintagetarot": true}
		seenStyles := make(map[string]bool, len(krea.styles))
		for _, style := range krea.styles {
			if !validStyles[style.Name] || seenStyles[style.Name] || style.Strength < 0 || style.Strength > 2 {
				http.Error(w, "invalid Krea style LoRA selection", http.StatusBadRequest)
				return
			}
			seenStyles[style.Name] = true
		}
		if len(krea.styles) > len(validStyles) {
			http.Error(w, "too many Krea style LoRAs", http.StatusBadRequest)
			return
		}
		seenUserLoras := make(map[string]bool, len(krea.userLoras))
		for _, selection := range krea.userLoras {
			if selection.Filename == "" || filepath.Base(selection.Filename) != selection.Filename || !strings.HasSuffix(strings.ToLower(selection.Filename), ".safetensors") || seenUserLoras[selection.Filename] || selection.Strength < -2 || selection.Strength > 2 {
				http.Error(w, "invalid user LoRA selection", http.StatusBadRequest)
				return
			}
			seenUserLoras[selection.Filename] = true
		}
		if len(krea.userLoras) > 5 {
			http.Error(w, "at most five user LoRAs may be stacked", http.StatusBadRequest)
			return
		}
		if krea.identityStrength < 0 || krea.identityStrength > 2 || krea.depthStrength < 0 || krea.depthStrength > 2 || krea.styleRefStrength < 0 || krea.styleRefStrength > 2 || krea.nk2eStrength < 0 || krea.nk2eStrength > 2 || krea.anypaintStrength < 0 || krea.anypaintStrength > 2 {
			http.Error(w, "Krea module strength must be between 0 and 2", http.StatusBadRequest)
			return
		}
		if krea.nk2eMode != "edit" && krea.nk2eMode != "canny" {
			http.Error(w, "NK2E mode must be edit or canny", http.StatusBadRequest)
			return
		}
		if krea.refBoost < 0 || krea.refBoost > 20 || krea.groundingPX < 384 || krea.groundingPX > 1024 {
			http.Error(w, "invalid Krea identity fidelity settings", http.StatusBadRequest)
			return
		}
		if krea.steps != 0 && (krea.steps < 1 || krea.steps > 20) {
			http.Error(w, "Krea steps must be between 1 and 20", http.StatusBadRequest)
			return
		}
		if krea.visionMode != "descriptor" && krea.visionMode != "instruct" {
			http.Error(w, "vision mode must be descriptor or instruct", http.StatusBadRequest)
			return
		}
		if krea.visionMegapixels < 0.1 || krea.visionMegapixels > 4 {
			http.Error(w, "vision megapixels must be between 0.1 and 4", http.StatusBadRequest)
			return
		}
		if len(krea.styleRefPaths) > 0 && (len(krea.visionPaths) > 0 || krea.identityPath != "" || krea.depthPath != "" || len(krea.styles) > 0 || len(krea.userLoras) > 0) {
			http.Error(w, "style reference cannot be combined with other Krea modules yet", http.StatusBadRequest)
			return
		}
		if len(krea.visionPaths) > 0 && krea.identityPath != "" {
			http.Error(w, "vision reference cannot be combined with identity yet", http.StatusBadRequest)
			return
		}
		if krea.nk2ePath != "" && (krea.identityPath != "" || krea.depthPath != "" || len(krea.styles) > 0 || len(krea.userLoras) > 0 || len(krea.visionPaths) > 0 || len(krea.styleRefPaths) > 0) {
			http.Error(w, "NK2E cannot be combined with other Krea modules yet", http.StatusBadRequest)
			return
		}
		if krea.anypaintMaskPath != "" && krea.anypaintPath == "" {
			http.Error(w, "AnyPaint mask requires a source image", http.StatusBadRequest)
			return
		}
		if krea.anypaintPath != "" {
			if krea.identityPath != "" || krea.depthPath != "" || len(krea.styles) > 0 || len(krea.userLoras) > 0 || len(krea.visionPaths) > 0 || len(krea.styleRefPaths) > 0 || krea.nk2ePath != "" {
				http.Error(w, "AnyPaint cannot be combined with other Krea modules yet", http.StatusBadRequest)
				return
			}
			pads := []int{krea.outpaintLeft, krea.outpaintTop, krea.outpaintRight, krea.outpaintBottom}
			for _, padding := range pads {
				if padding < 0 || padding > 1536 || padding%16 != 0 {
					http.Error(w, "outpaint padding must be 0..1536 in multiples of 16", http.StatusBadRequest)
					return
				}
			}
			if krea.anypaintMaskPath == "" && krea.outpaintLeft+krea.outpaintTop+krea.outpaintRight+krea.outpaintBottom == 0 {
				http.Error(w, "AnyPaint requires a mask or at least one expansion direction", http.StatusBadRequest)
				return
			}
			if krea.anypaintBoundary < 0 || krea.anypaintBoundary > 256 {
				http.Error(w, "AnyPaint boundary redraw must be between 0 and 256", http.StatusBadRequest)
				return
			}
		}
		if krea.identityPath != "" && width*height > 2*1024*1024 {
			http.Error(w, "Krea Identity Edit output must not exceed 2 megapixels", http.StatusBadRequest)
			return
		}
		if len(sequencePrompts) > 0 {
			if krea.identityPath != "" || krea.depthPath != "" || len(krea.visionPaths) > 0 || len(krea.styleRefPaths) > 0 || krea.nk2ePath != "" || krea.anypaintPath != "" {
				http.Error(w, "sequence generation cannot be combined with reference, depth, vision, structure, or partial-edit modules", http.StatusBadRequest)
				return
			}
			if width*height > 2*1024*1024 {
				http.Error(w, "sequence generation must not exceed 2 megapixels", http.StatusBadRequest)
				return
			}
		}
	case "edit":
		if len(refs) == 0 {
			http.Error(w, "reference editing requires at least one image", http.StatusBadRequest)
			return
		}
	case "control":
		if len(refs) != 1 {
			http.Error(w, "structure control requires exactly one image", http.StatusBadRequest)
			return
		}
		if controlType == "" {
			controlType = "canny"
		}
		if controlType != "canny" {
			http.Error(w, "only canny control is currently available", http.StatusBadRequest)
			return
		}
		if controlStrength < 0 || controlStrength > 2 {
			http.Error(w, "control strength must be between 0 and 2", http.StatusBadRequest)
			return
		}
	}
	params := map[string]any{"width": width, "height": height, "seed": seed, "references": len(refs), "mode": mode, "model": cfg.Image.Backends[mode].Model}
	if parent := strings.TrimSpace(r.FormValue("parent_job_id")); parent != "" {
		parentJob, ok := s.jobs.Get(parent)
		if !ok || parentJob.Kind != "image" || parentJob.Status != "completed" {
			http.Error(w, "parent image no longer exists", http.StatusBadRequest)
			return
		}
		params["parent_job_id"] = parent
	}
	if mode == "control" {
		params["control_type"] = controlType
		params["control_strength"] = controlStrength
	} else if mode == "create" {
		params["checkpoint"] = krea.checkpoint
		params["identity"] = krea.identityPath != ""
		params["identity_reference"] = len(krea.identityRefPaths) > 0
		params["identity_reference_count"] = len(krea.identityRefPaths)
		params["identity_preset"] = identityPreset
		params["identity_auto_prompt"] = krea.identityAutoPrompt
		params["identity_user_prompt"] = krea.identityUserPrompt
		params["identity_preserve_items"] = identityPreserveItems
		params["identity_preserve_custom"] = identityPreserveCustom
		params["depth"] = krea.depthPath != ""
		if krea.depthPrompt != "" {
			params["depth_pose_prompt"] = krea.depthPrompt
		}
		params["prepare_pose_reference"] = krea.preparePoseRef
		params["style"] = krea.style
		params["styles"] = krea.styles
		params["user_loras"] = krea.userLoras
		params["vision"] = len(krea.visionPaths) > 0
		params["vision_count"] = len(krea.visionPaths)
		params["style_reference"] = len(krea.styleRefPaths) > 0
		params["style_reference_count"] = len(krea.styleRefPaths)
		params["nk2e"] = krea.nk2ePath != ""
		params["anypaint"] = krea.anypaintPath != ""
		params["identity_mask"] = krea.identityMaskPath != ""
		params["strict_mask"] = krea.strictMaskPath != ""
		params["vae_mode"] = krea.vaeMode
		params["identity_fit_mode"] = krea.identityFitMode
		params["identity_model"] = krea.identityModel
		params["identity_encoder"] = krea.identityEncoder
		params["strict_mask_grow"] = krea.strictMaskGrow
		params["strict_mask_feather"] = krea.strictMaskFeather
		params["filter_mode"] = krea.filterMode
		params["filter_strength"] = krea.filterStrength
		params["prompt_enhancer"] = krea.promptEnhancer
		params["prompt_enhancer_strength"] = krea.promptEnhStrength
		params["prompt_text_scale"] = krea.promptTextScale
		params["sampling_preset"] = krea.samplingPreset
		params["sampler"] = krea.sampler
		params["scheduler"] = krea.scheduler
		params["steps"] = krea.steps
		if len(krea.visionPaths) > 0 {
			params["vision_mode"] = krea.visionMode
			params["vision_megapixels"] = krea.visionMegapixels
		}
		if len(krea.styleRefPaths) > 0 {
			params["style_reference_strength"] = krea.styleRefStrength
		}
		if krea.nk2ePath != "" {
			params["nk2e_mode"] = krea.nk2eMode
			params["nk2e_strength"] = krea.nk2eStrength
			params["nk2e_preprocessed"] = krea.nk2ePreprocessed
		}
		if krea.anypaintPath != "" {
			params["anypaint_mask"] = krea.anypaintMaskPath != ""
			params["outpaint_left"] = krea.outpaintLeft
			params["outpaint_top"] = krea.outpaintTop
			params["outpaint_right"] = krea.outpaintRight
			params["outpaint_bottom"] = krea.outpaintBottom
			params["anypaint_strength"] = krea.anypaintStrength
			params["anypaint_boundary_redraw_px"] = krea.anypaintBoundary
		}
		if krea.identityPath != "" {
			params["identity_strength"] = krea.identityStrength
			params["ref_boost"] = krea.refBoost
			params["source_ref_boost"] = krea.sourceRefBoost
			params["grounding_px"] = krea.groundingPX
		}
		if krea.depthPath != "" {
			params["depth_strength"] = krea.depthStrength
		}
		if len(krea.styles) > 0 {
			params["style"] = krea.styles[0].Name
			params["style_strength"] = krea.styles[0].Strength
		}
		if len(krea.userLoras) > 0 {
			params["user_loras"] = krea.userLoras
		}
	}
	params["enhanced_prompt"] = valueIfDifferent(effectivePrompt, originalPrompt)
	params["stage"] = "queued"
	now := time.Now()
	params["queued_at"] = now.Format(time.RFC3339Nano)
	j := jobs.Job{ID: id, Kind: "image", Status: "queued", Prompt: originalPrompt, Params: params, CreatedAt: now}
	sequenceBaseID := strings.TrimSpace(r.FormValue("sequence_base_job_id"))
	var sequenceBase jobs.Job
	if sequenceBaseID != "" {
		if len(sequencePrompts) == 0 {
			http.Error(w, "sequence base requires sequence prompts", http.StatusBadRequest)
			return
		}
		var ok bool
		sequenceBase, ok = s.jobs.Get(sequenceBaseID)
		if !ok || sequenceBase.Kind != "image" || sequenceBase.Status != "completed" || sequenceBase.OutputURL == "" {
			http.Error(w, "selected sequence base image is not available", http.StatusBadRequest)
			return
		}
	}
	for index := 1; index < len(sequenceRegions); index++ {
		if sequenceRegions[index] == "custom" && len(r.MultipartForm.File[fmt.Sprintf("sequence_mask_%d", index)]) != 1 {
			http.Error(w, fmt.Sprintf("scene %d requires a painted mask", index+1), http.StatusBadRequest)
			return
		}
	}
	if len(sequencePrompts) > 0 {
		j.Prompt = sequencePrompts[0]
		j.Params["enhanced_prompt"] = ""
		j.Params["sequence_id"] = id
		j.Params["sequence_index"] = 1
		j.Params["sequence_total"] = len(sequencePrompts)
		j.Params["sequence_identity_strength"] = sequenceIdentityStrength
	}
	created := make([]jobs.Job, 0, max(1, len(sequencePrompts)))
	previousID := id
	startIndex := 1
	if sequenceBaseID == "" {
		if err := s.jobs.Save(j); err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		created = append(created, j)
	} else {
		previousID = sequenceBase.ID
	}
	for index := startIndex; index < len(sequencePrompts); index++ {
		childID := newID()
		childParams := cloneJobParams(params)
		childParams["identity"] = true
		childParams["identity_reference"] = false
		childParams["identity_strength"] = sequenceIdentityStrength
		childParams["ref_boost"] = 4.0
		childParams["grounding_px"] = 768
		childParams["steps"] = max(imageIntParam(childParams, "steps", 8), 10)
		childParams["enhanced_prompt"] = sequenceEditPrompt(sequencePrompts[index])
		childParams["parent_job_id"] = previousID
		childParams["sequence_previous_job_id"] = previousID
		childParams["sequence_id"] = id
		childParams["sequence_index"] = index + 1
		childParams["sequence_total"] = len(sequencePrompts)
		childParams["sequence_identity_strength"] = sequenceIdentityStrength
		childParams["sequence_region"] = sequenceRegions[index]
		if sequenceRegions[index] != "all" {
			childParams["identity"] = false
			childParams["anypaint"] = true
			childParams["anypaint_mask"] = true
			childParams["anypaint_strength"] = 1.0
			childParams["anypaint_boundary_redraw_px"] = 32
			childParams["styles"] = []styleSelection{}
			childParams["user_loras"] = []userLoRASelection{}
			childParams["style"] = ""
		}
		childTime := now.Add(time.Duration(index) * time.Nanosecond)
		childParams["queued_at"] = childTime.Format(time.RFC3339Nano)
		if seed >= 0 {
			childParams["seed"] = seed + int64(index)
		}
		child := jobs.Job{ID: childID, Kind: "image", Status: "queued", Prompt: sequencePrompts[index], Params: childParams, CreatedAt: childTime}
		if sequenceRegions[index] == "custom" {
			masks, uploadErr := saveUploads(r, fmt.Sprintf("sequence_mask_%d", index), filepath.Join(s.dataDir, "inputs", childID, "anypaint-mask"), 1)
			if uploadErr != nil || len(masks) != 1 {
				if uploadErr != nil {
					http.Error(w, uploadErr.Error(), http.StatusBadRequest)
				} else {
					http.Error(w, fmt.Sprintf("scene %d requires a painted mask", index+1), http.StatusBadRequest)
				}
				return
			}
		}
		if err := s.jobs.Save(child); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		created = append(created, child)
		previousID = childID
	}
	s.wakeGenerationQueue()
	if len(sequencePrompts) > 0 {
		writeJSON(w, http.StatusAccepted, map[string]any{"sequence_id": id, "jobs": created})
		return
	}
	writeJSON(w, http.StatusAccepted, j)
}

func parseImageSequencePrompts(raw string) ([]string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	var prompts []string
	if err := json.Unmarshal([]byte(raw), &prompts); err != nil {
		return nil, fmt.Errorf("invalid sequence prompts")
	}
	if len(prompts) < 2 || len(prompts) > 6 {
		return nil, fmt.Errorf("sequence generation requires 2 to 6 scenes")
	}
	for index := range prompts {
		prompts[index] = strings.TrimSpace(prompts[index])
		if prompts[index] == "" {
			return nil, fmt.Errorf("every sequence scene requires a prompt")
		}
		if len([]rune(prompts[index])) > 4000 {
			return nil, fmt.Errorf("sequence scene prompt is too long")
		}
	}
	return prompts, nil
}

func parseImageSequenceRegions(raw string, promptCount int) ([]string, error) {
	if promptCount == 0 {
		return nil, nil
	}
	regions := make([]string, promptCount)
	for index := range regions {
		regions[index] = "all"
	}
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return regions, nil
	}
	var provided []string
	if err := json.Unmarshal([]byte(raw), &provided); err != nil || len(provided) != promptCount {
		return nil, fmt.Errorf("invalid sequence regions")
	}
	valid := map[string]bool{"all": true, "left": true, "right": true, "upper": true, "lower": true, "left-arm": true, "right-arm": true, "custom": true}
	for index, region := range provided {
		region = strings.ToLower(strings.TrimSpace(region))
		if !valid[region] {
			return nil, fmt.Errorf("unsupported sequence region")
		}
		regions[index] = region
	}
	regions[0] = "all"
	return regions, nil
}

func cloneJobParams(source map[string]any) map[string]any {
	clone := make(map[string]any, len(source)+8)
	for key, value := range source {
		clone[key] = value
	}
	return clone
}

func sequenceEditPrompt(change string) string {
	return "Change: " + strings.TrimSpace(change) +
		"\nPose replacement rule: Treat every requested pose or body-part movement as a replacement, never an addition. Redraw each moved body part only in its new position and remove it from its previous position. Keep the anatomically correct number of limbs, with no duplicate arms, hands, legs, heads, or ghost body parts." +
		"\nFace continuity rule: Preserve the exact head and facial construction, including faceplate or screen type, eye count, eye shape, eye color, eye spacing, mouth design, and distinguishing details. If an expression change is explicitly requested, alter only that expression and never redesign the head or face." +
		"\nPreserve: the same character identity, face, hair, clothing details unless explicitly changed, visual style, lighting continuity, and scene elements that do not conflict with the requested movement. Do not preserve the previous pose where it conflicts with the new pose."
}

func (s *Server) createSpeech(w http.ResponseWriter, r *http.Request) {
	cfg := s.config()
	if err := r.ParseMultipartForm(40 << 20); err != nil {
		http.Error(w, "invalid form", 400)
		return
	}
	text := strings.TrimSpace(r.FormValue("text"))
	if text == "" {
		http.Error(w, "text is required", 400)
		return
	}
	id := newID()
	language := valueOr(r.FormValue("language"), cfg.Speech.DefaultLanguage)
	speaker := valueOr(r.FormValue("speaker"), cfg.Speech.DefaultSpeaker)
	instructions := strings.TrimSpace(r.FormValue("instructions"))
	seed := formInt64(r, "seed", -1)
	j := jobs.Job{ID: id, Kind: "speech", Status: "queued", Prompt: text, Params: map[string]any{"language": language, "speaker": speaker, "instructions": instructions, "seed": seed, "stage": "queued", "queued_at": time.Now().Format(time.RFC3339Nano)}, CreatedAt: time.Now()}
	if err := s.jobs.Save(j); err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	s.wakeGenerationQueue()
	writeJSON(w, 202, j)
}

func (s *Server) createImageDetailEnhance(w http.ResponseWriter, r *http.Request) {
	source, ok := s.jobs.Get(r.PathValue("id"))
	if !ok {
		http.NotFound(w, r)
		return
	}
	if source.Kind != "image" || source.Status != "completed" || source.OutputURL == "" {
		http.Error(w, "only a completed image can be detail-enhanced", http.StatusConflict)
		return
	}
	request := struct {
		Strength float64 `json:"strength"`
		Seed     int64   `json:"seed"`
		VAE      string  `json:"vae"`
	}{Strength: 1, Seed: -1, VAE: "wan"}
	decoder := json.NewDecoder(io.LimitReader(r.Body, 1<<20))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&request); err != nil && !errors.Is(err, io.EOF) {
		http.Error(w, "invalid detail enhancement request: "+err.Error(), http.StatusBadRequest)
		return
	}
	if request.Strength < 0 || request.Strength > 2 || (request.VAE != "wan" && request.VAE != "qwen") {
		http.Error(w, "detail strength must be 0..2 and VAE must be wan or qwen", http.StatusBadRequest)
		return
	}
	data, err := os.ReadFile(s.jobs.OutputPath(filepath.Base(source.OutputURL)))
	if err != nil {
		http.Error(w, "source image is no longer available", http.StatusNotFound)
		return
	}
	input, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		http.Error(w, "source image is invalid", http.StatusBadRequest)
		return
	}
	if input.Width < 512 || input.Width > 2048 || input.Height < 512 || input.Height > 2048 || input.Width%16 != 0 || input.Height%16 != 0 {
		http.Error(w, "detail enhancement requires a 512..2048 image with dimensions divisible by 16", http.StatusBadRequest)
		return
	}
	id := newID()
	params := map[string]any{
		"mode": "detail_enhance", "source_job_id": source.ID, "parent_job_id": source.ID,
		"model":           s.config().Image.Backends["create"].Model,
		"detail_strength": request.Strength, "detail_vae": request.VAE, "seed": request.Seed,
		"width": input.Width, "height": input.Height, "steps": 10,
		"sampling_preset": "detail", "sampler": "er_sde", "scheduler": "simple",
		"stage": "queued", "queued_at": time.Now().Format(time.RFC3339Nano),
	}
	j := jobs.Job{ID: id, Kind: "image", Status: "queued", Prompt: source.Prompt, Params: params, CreatedAt: time.Now()}
	if err := s.jobs.Save(j); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	s.wakeGenerationQueue()
	writeJSON(w, http.StatusAccepted, j)
}

func (s *Server) runImageDetailEnhance(j jobs.Job, source []byte, strength float64, seed int64, vae string) {
	cfg := s.config()
	s.heavyMu.Lock()
	defer s.heavyMu.Unlock()
	j.Status = "running"
	j.Params["started_at"] = time.Now().Format(time.RFC3339Nano)
	_ = s.jobs.Save(j)
	backend := cfg.Image.Backends["create"]
	request := map[string]any{
		"model":           backend.Model,
		"prompt":          "Enhance this image to high resolution while preserving the composition, subject identity, colors, lighting, and text. Improve natural skin texture, material detail, and fine background detail.",
		"size":            fmt.Sprintf("%dx%d", j.Params["width"], j.Params["height"]),
		"response_format": "b64_json", "output_format": "png",
		"detail_enhance_image": base64.StdEncoding.EncodeToString(source),
		"detail_strength":      strength, "detail_vae": vae, "steps": 10,
		"filter_mode": "balanced", "filter_strength": 1,
		"sampler_name": "er_sde", "scheduler": "simple",
	}
	if seed >= 0 {
		request["seed"] = seed
	}
	response, _, err := s.callJSON(backend.Endpoint+"/v1/images/generations", request)
	if err != nil {
		s.fail(j, err)
		return
	}
	if actualSeed, ok := decodeImageSeed(response); ok {
		j.Params["seed"] = actualSeed
	}
	data, err := decodeImage(response)
	if err != nil {
		s.fail(j, err)
		return
	}
	if err = s.writeImageResult(&j, data, j.Prompt); err != nil {
		s.fail(j, err)
		return
	}
}

func (s *Server) createImageUpscale(w http.ResponseWriter, r *http.Request) {
	source, ok := s.jobs.Get(r.PathValue("id"))
	if !ok {
		http.NotFound(w, r)
		return
	}
	if source.Kind != "image" || source.Status != "completed" || source.OutputURL == "" {
		http.Error(w, "only a completed image can be upscaled", http.StatusConflict)
		return
	}
	var request struct {
		Scale int   `json:"scale"`
		Seed  int64 `json:"seed"`
	}
	request.Scale = 2
	request.Seed = -1
	decoder := json.NewDecoder(io.LimitReader(r.Body, 1<<20))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&request); err != nil && !errors.Is(err, io.EOF) {
		http.Error(w, "invalid upscale request: "+err.Error(), http.StatusBadRequest)
		return
	}
	if request.Scale < 2 || request.Scale > 4 {
		http.Error(w, "upscale scale must be between 2 and 4", http.StatusBadRequest)
		return
	}
	sourcePath := s.jobs.OutputPath(filepath.Base(source.OutputURL))
	data, err := os.ReadFile(sourcePath)
	if err != nil {
		http.Error(w, "source image is no longer available", http.StatusNotFound)
		return
	}
	input, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		http.Error(w, "source image is invalid", http.StatusBadRequest)
		return
	}
	width, height := input.Width*request.Scale, input.Height*request.Scale
	if width > 4096 || height > 4096 {
		http.Error(w, "upscaled image must not exceed 4096 pixels on either edge", http.StatusBadRequest)
		return
	}
	id := newID()
	params := map[string]any{
		"mode": "upscale", "source_job_id": source.ID, "upscale_engine": "seedvr2-3b-fp8",
		"model":         "seedvr2-3b-fp8",
		"upscale_scale": request.Scale, "seed": request.Seed, "width": width, "height": height,
		"stage": "queued", "queued_at": time.Now().Format(time.RFC3339Nano),
	}
	if enhanced, exists := source.Params["enhanced_prompt"]; exists {
		params["source_enhanced_prompt"] = enhanced
	}
	j := jobs.Job{ID: id, Kind: "image", Status: "queued", Prompt: source.Prompt, Params: params, CreatedAt: time.Now()}
	if err := s.jobs.Save(j); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	s.wakeGenerationQueue()
	writeJSON(w, http.StatusAccepted, j)
}

func (s *Server) runImageUpscale(j jobs.Job, source []byte, scale int, seed int64) {
	cfg := s.config()
	s.heavyMu.Lock()
	defer s.heavyMu.Unlock()
	j.Status = "running"
	j.Params["started_at"] = time.Now().Format(time.RFC3339Nano)
	_ = s.jobs.Save(j)
	request := map[string]any{
		"model": "seedvr2-3b-fp8", "image": base64.StdEncoding.EncodeToString(source),
		"scale": scale, "response_format": "b64_json", "output_format": "png",
	}
	if seed >= 0 {
		request["seed"] = seed
	}
	response, _, err := s.callJSON(cfg.Engines["upscale"].Endpoint+"/v1/images/upscale", request)
	if err != nil {
		s.fail(j, err)
		return
	}
	if actualSeed, ok := decodeImageSeed(response); ok {
		j.Params["seed"] = actualSeed
	}
	data, err := decodeImage(response)
	if err != nil {
		s.fail(j, err)
		return
	}
	if err = s.writeImageResult(&j, data, j.Prompt); err != nil {
		s.fail(j, err)
		return
	}
}

func (s *Server) runImage(j jobs.Job, effectivePrompt string, refs []string, width, height int, seed int64, mode, controlType string, controlStrength float64, krea imageGenerationOptions) {
	cfg := s.config()
	s.heavyMu.Lock()
	defer s.heavyMu.Unlock()
	j.Status = "running"
	j.Params["started_at"] = time.Now().Format(time.RFC3339Nano)
	_ = s.jobs.Save(j)
	if krea.identityAutoPrompt && krea.identityPreset == "tryon" && len(krea.identityRefPaths) > 0 {
		outfit, describeErr := s.describeOutfitReference(krea.identityRefPaths[0])
		if describeErr != nil {
			s.fail(j, fmt.Errorf("outfit reference description: %w", describeErr))
			return
		}
		pronoun, describeErr := s.describeSubjectPronoun(krea.identityPath)
		if describeErr != nil {
			s.fail(j, fmt.Errorf("source subject description: %w", describeErr))
			return
		}
		outfit = strings.TrimSpace(strings.TrimSuffix(outfit, "."))
		lowerOutfit := strings.ToLower(outfit)
		switch {
		case strings.HasPrefix(lowerOutfit, "an "):
			outfit = "the " + strings.TrimSpace(outfit[3:])
		case strings.HasPrefix(lowerOutfit, "a "):
			outfit = "the " + strings.TrimSpace(outfit[2:])
		}
		pronoun = strings.ToLower(strings.TrimSpace(strings.Trim(pronoun, ".,;:!?\"'")))
		if pronoun != "she" && pronoun != "he" && pronoun != "they" && pronoun != "it" {
			pronoun = "they"
		}
		verb := "is"
		if pronoun == "they" {
			verb = "are"
		}
		// Preserve the lowercase training-style wording used by the published
		// Identity Edit workflow. With otherwise identical inputs and seed,
		// capitalizing this leading pronoun caused the source shirt to survive.
		modulePrompt := pronoun + " " + verb + " now wearing " + outfit + "."
		hasExtraUserPrompt := krea.identityUserPrompt && strings.TrimSpace(effectivePrompt) != "" && !isTryOnModuleFallback(j.Prompt, krea.depthPath != "")
		if hasExtraUserPrompt {
			if composed, composeErr := s.composeIdentityEditPrompt(pronoun, verb, outfit, effectivePrompt, krea.depthPath != ""); composeErr == nil {
				modulePrompt = composed
			} else {
				modulePrompt += "\n" + strings.TrimSpace(effectivePrompt)
				if krea.depthPath != "" {
					modulePrompt += "\n" + pronoun + " now holds the same pose."
				}
			}
		} else if krea.depthPath != "" {
			modulePrompt += "\n" + pronoun + " now holds the same pose."
		}
		effectivePrompt = modulePrompt
		j.Params["generated_edit_prompt"] = effectivePrompt
		_ = s.jobs.Save(j)
	}
	if krea.identityPath != "" && krea.depthPath != "" && krea.depthPrompt == "" && cfg.PromptEnhancement.VisionEnabled {
		poseDescription, describeErr := s.describePoseReference(krea.depthPath)
		if describeErr != nil {
			s.fail(j, fmt.Errorf("pose reference description: %w", describeErr))
			return
		}
		krea.depthPrompt = poseDescription
		j.Params["depth_pose_prompt"] = poseDescription
		_ = s.jobs.Save(j)
	}
	backend := cfg.Image.Backends[mode]
	endpoint := backend.Endpoint
	var response []byte
	var err error
	if mode == "control" {
		controlImage, readErr := os.ReadFile(refs[0])
		if readErr != nil {
			s.fail(j, readErr)
			return
		}
		request := map[string]any{
			"model": backend.Model, "prompt": effectivePrompt,
			"checkpoint": krea.checkpoint,
			"size":       fmt.Sprintf("%dx%d", width, height), "response_format": "b64_json", "output_format": "png",
			"control_image":    base64.StdEncoding.EncodeToString(controlImage),
			"control_strength": controlStrength, "control_strategy": "split4", "control_type": controlType,
		}
		if seed >= 0 {
			request["seed"] = seed
		}
		response, _, err = s.callJSON(endpoint+"/v1/images/generations", request)
	} else if mode == "edit" {
		fields := map[string]string{
			"model": backend.Model, "prompt": effectivePrompt,
			"size": fmt.Sprintf("%dx%d", width, height), "response_format": "b64_json", "output_format": "png",
		}
		if seed >= 0 {
			fields["seed"] = strconv.FormatInt(seed, 10)
		}
		response, _, err = s.callMultipart(endpoint+"/v1/images/edits", fields, "image", refs)
	} else {
		request := map[string]any{
			"model": backend.Model, "prompt": effectivePrompt,
			"checkpoint": krea.checkpoint,
			"size":       fmt.Sprintf("%dx%d", width, height), "response_format": "b64_json", "output_format": "png",
			"filter_mode": krea.filterMode, "filter_strength": krea.filterStrength,
			"prompt_enhancer": krea.promptEnhancer, "prompt_enhancer_strength": krea.promptEnhStrength,
			"prompt_text_scale": krea.promptTextScale,
			"sampler_name":      krea.sampler, "scheduler": krea.scheduler,
		}
		for field, path := range map[string]string{
			"source_image": krea.identityPath, "control_image": krea.depthPath, "nk2e_image": krea.nk2ePath,
			"identity_mask": krea.identityMaskPath, "strict_mask": krea.strictMaskPath,
			"anypaint_image": krea.anypaintPath, "anypaint_mask": krea.anypaintMaskPath,
		} {
			if path == "" {
				continue
			}
			image, readErr := os.ReadFile(path)
			if readErr != nil {
				s.fail(j, readErr)
				return
			}
			request[field] = base64.StdEncoding.EncodeToString(image)
		}
		for field, paths := range map[string][]string{
			"reference_images": krea.identityRefPaths, "vision_images": krea.visionPaths, "style_reference_images": krea.styleRefPaths,
		} {
			encoded := make([]string, 0, len(paths))
			for _, path := range paths {
				image, readErr := os.ReadFile(path)
				if readErr != nil {
					s.fail(j, readErr)
					return
				}
				encoded = append(encoded, base64.StdEncoding.EncodeToString(image))
			}
			if len(encoded) > 0 {
				request[field] = encoded
			}
		}
		if krea.identityPath != "" {
			request["identity_strength"] = krea.identityStrength
			request["ref_boost"] = krea.refBoost
			request["source_ref_boost"] = krea.sourceRefBoost
			request["grounding_px"] = krea.groundingPX
			request["strict_mask_grow"] = krea.strictMaskGrow
			request["strict_mask_feather"] = krea.strictMaskFeather
			request["vae_mode"] = krea.vaeMode
			request["identity_fit_mode"] = krea.identityFitMode
			request["identity_model"] = krea.identityModel
			request["identity_encoder"] = krea.identityEncoder
		}
		if krea.depthPath != "" {
			request["control_strength"] = krea.depthStrength
			request["control_prompt"] = krea.depthPrompt
			request["prepare_pose_reference"] = krea.preparePoseRef
		}
		if len(krea.styles) > 0 {
			request["styles"] = krea.styles
			request["style"] = krea.styles[0].Name
			request["style_strength"] = krea.styles[0].Strength
		}
		if len(krea.userLoras) > 0 {
			request["user_loras"] = krea.userLoras
		}
		if len(krea.visionPaths) > 0 {
			request["vision_mode"] = krea.visionMode
			request["vision_megapixels"] = krea.visionMegapixels
		}
		if len(krea.styleRefPaths) > 0 {
			request["style_reference_strength"] = krea.styleRefStrength
		}
		if krea.nk2ePath != "" {
			request["nk2e_mode"] = krea.nk2eMode
			request["nk2e_strength"] = krea.nk2eStrength
			request["nk2e_preprocessed"] = krea.nk2ePreprocessed
		}
		if krea.anypaintPath != "" {
			request["outpaint_left"] = krea.outpaintLeft
			request["outpaint_top"] = krea.outpaintTop
			request["outpaint_right"] = krea.outpaintRight
			request["outpaint_bottom"] = krea.outpaintBottom
			request["anypaint_strength"] = krea.anypaintStrength
			request["anypaint_boundary_redraw_px"] = krea.anypaintBoundary
			request["anypaint_reference_max_edge"] = 384
			request["anypaint_vlm_reference"] = true
		}
		if krea.steps > 0 {
			request["steps"] = krea.steps
		}
		if seed >= 0 {
			request["seed"] = seed
		}
		response, _, err = s.callJSON(endpoint+"/v1/images/generations", request)
	}
	if err != nil {
		s.fail(j, err)
		return
	}
	if actualSeed, ok := decodeImageSeed(response); ok {
		j.Params["seed"] = actualSeed
	}
	data, err := decodeImage(response)
	if err != nil {
		s.fail(j, err)
		return
	}
	if err = s.writeImageResult(&j, data, effectivePrompt); err != nil {
		s.fail(j, err)
		return
	}
}

func isTryOnModuleFallback(prompt string, withPose bool) bool {
	expected := "Use the complete outfit shown in the supporting clothing reference"
	if withPose {
		expected += ". Apply the pose, body orientation, framing, and camera viewpoint from the pose reference"
	}
	return strings.EqualFold(strings.TrimSpace(prompt), expected)
}

func (s *Server) writeImageResult(j *jobs.Job, data []byte, effectivePrompt string) error {
	if output, _, err := image.DecodeConfig(bytes.NewReader(data)); err == nil {
		j.Params["width"] = output.Width
		j.Params["height"] = output.Height
	}
	data = embedImageEXIF(data, metadataForImageJob(*j, effectivePrompt, s.config().ImageMetadata))
	name := j.ID + ".png"
	if err := os.WriteFile(s.jobs.OutputPath(name), data, 0o644); err != nil {
		return err
	}
	j.Status = "completed"
	j.OutputURL = "/api/outputs/" + name
	return s.jobs.Save(*j)
}

func (s *Server) imageJobEXIF(w http.ResponseWriter, r *http.Request) {
	j, ok := s.jobs.Get(r.PathValue("id"))
	if !ok || j.Kind != "image" || j.OutputURL == "" {
		http.NotFound(w, r)
		return
	}
	data, err := os.ReadFile(s.jobs.OutputPath(filepath.Base(j.OutputURL)))
	if err != nil {
		http.Error(w, "image file is no longer available", http.StatusNotFound)
		return
	}
	metadata, embedded := extractImageEXIF(data)
	writeJSON(w, http.StatusOK, map[string]any{"embedded": embedded, "metadata": metadata})
}

func (s *Server) runSpeech(j jobs.Job, language, speaker, instructions string, seed int64) {
	cfg := s.config()
	j.Status = "running"
	_ = s.jobs.Save(j)
	request := map[string]any{
		"model": cfg.Speech.CustomVoiceModel, "input": j.Prompt,
		"language": language, "voice": strings.ToLower(speaker),
		"instructions": instructions,
		"task_type":    "CustomVoice", "response_format": "wav", "stream": false,
	}
	if seed >= 0 {
		request["seed"] = seed
	}
	endpoint := cfg.Engines["speech"].Endpoint
	data, _, err := s.callJSON(endpoint+"/v1/audio/speech", request)
	if err != nil {
		s.fail(j, err)
		return
	}
	name := j.ID + ".wav"
	if err = os.WriteFile(s.jobs.OutputPath(name), data, 0o644); err != nil {
		s.fail(j, err)
		return
	}
	j.Status = "completed"
	j.OutputURL = "/api/outputs/" + name
	_ = s.jobs.Save(j)
}

func (s *Server) runVideo(j jobs.Job, effectivePrompt string, conditions []videoConditioningInput, width, height, frames int, fps float64, seed int64) {
	cfg := s.config()
	s.heavyMu.Lock()
	defer s.heavyMu.Unlock()
	j.Status = "running"
	j.Params["started_at"] = time.Now().Format(time.RFC3339Nano)
	_ = s.jobs.Save(j)
	fields := map[string]string{
		"prompt": effectivePrompt,
		"width":  strconv.Itoa(width), "height": strconv.Itoa(height),
		"num_frames": strconv.Itoa(frames), "fps": strconv.FormatFloat(fps, 'f', -1, 64),
		"seed": strconv.FormatInt(seed, 10),
	}
	motionStrength := 0.0
	if enabled, _ := j.Params["motion_lora_enabled"].(bool); enabled {
		motionStrength, _ = numberFromAny(j.Params["motion_lora_strength"])
	}
	fields["motion_lora_strength"] = strconv.FormatFloat(motionStrength, 'f', -1, 64)
	paths := make([]string, 0, len(conditions))
	frameIndices := make([]int, 0, len(conditions))
	strengths := make([]float64, 0, len(conditions))
	for _, condition := range conditions {
		paths = append(paths, condition.Path)
		frameIndices = append(frameIndices, condition.FrameIdx)
		strengths = append(strengths, condition.Strength)
	}
	frameJSON, _ := json.Marshal(frameIndices)
	strengthJSON, _ := json.Marshal(strengths)
	fields["frame_indices"] = string(frameJSON)
	fields["image_strengths"] = string(strengthJSON)
	name := j.ID + ".mp4"
	output := s.jobs.OutputPath(name)
	endpoint := cfg.Engines["video"].Endpoint
	if err := s.callMultipartToFile(endpoint+"/v1/videos/generations", fields, "images", paths, output); err != nil {
		_ = os.Remove(output)
		s.fail(j, err)
		return
	}
	j.Status = "completed"
	j.OutputURL = "/api/outputs/" + name
	_ = s.jobs.Save(j)
	go func() { _ = s.ensureVideoPreview(j.ID, output) }()
}

func numberFromAny(value any) (float64, bool) {
	switch number := value.(type) {
	case float64:
		return number, true
	case float32:
		return float64(number), true
	case int:
		return float64(number), true
	case int64:
		return float64(number), true
	case json.Number:
		parsed, err := number.Float64()
		return parsed, err == nil
	default:
		return 0, false
	}
}

func (s *Server) enhancePrompt(w http.ResponseWriter, r *http.Request) {
	cfg := s.config()
	if err := r.ParseMultipartForm(40 << 20); err != nil {
		http.Error(w, "invalid or oversized form", http.StatusBadRequest)
		return
	}
	original := strings.TrimSpace(r.FormValue("prompt"))
	if original == "" {
		http.Error(w, "prompt is required", http.StatusBadRequest)
		return
	}
	mode := strings.ToLower(strings.TrimSpace(r.FormValue("mode")))
	if mode == "" {
		mode = "t2v"
	}
	if mode != "t2v" && mode != "i2v" && mode != "t2i" && mode != "edit" && mode != "edit_control" && mode != "control" && mode != "paint" {
		http.Error(w, "mode must be t2i, edit, edit_control, control, paint, t2v or i2v", http.StatusBadRequest)
		return
	}
	if mode == "i2v" && !cfg.PromptEnhancement.VisionEnabled {
		http.Error(w, "I2V prompt enhancement requires a vision-enabled model bundle", http.StatusConflict)
		return
	}

	visionRequested := mode == "i2v" && cfg.PromptEnhancement.VisionEnabled
	userContent := any("User Raw Input Prompt: " + original)
	imageUsed := false
	if visionRequested {
		if file, header, err := r.FormFile("image"); err == nil {
			defer file.Close()
			data, readErr := io.ReadAll(io.LimitReader(file, (32<<20)+1))
			if readErr != nil || len(data) > 32<<20 {
				http.Error(w, "reference image is invalid or too large", http.StatusBadRequest)
				return
			}
			contentType := header.Header.Get("Content-Type")
			if contentType == "" {
				contentType = mime.TypeByExtension(strings.ToLower(filepath.Ext(header.Filename)))
			}
			if contentType == "" {
				contentType = http.DetectContentType(data)
			}
			userContent = []map[string]any{
				{"type": "image_url", "image_url": map[string]string{"url": "data:" + contentType + ";base64," + base64.StdEncoding.EncodeToString(data)}},
				{"type": "text", "text": "User Raw Input Prompt: " + original},
			}
			imageUsed = true
		} else {
			http.Error(w, "reference image is required for I2V prompt enhancement", http.StatusBadRequest)
			return
		}
	}

	systemPrompt := mediaprompt.System(mode, imageUsed)
	if mode == "edit" || mode == "edit_control" {
		preset := strings.TrimSpace(r.FormValue("identity_preset"))
		validPresets := map[string]bool{"": true, "restage": true, "sheet": true, "faceSwap": true, "headSwap": true, "personSwap": true, "tryon": true, "replace": true}
		if !validPresets[preset] {
			http.Error(w, "unsupported identity preset", http.StatusBadRequest)
			return
		}
		preserved := []string{}
		if raw := strings.TrimSpace(r.FormValue("identity_preserve_items")); raw != "" {
			if err := json.Unmarshal([]byte(raw), &preserved); err != nil {
				http.Error(w, "invalid identity preservation selection", http.StatusBadRequest)
				return
			}
			allowed := map[string]bool{"identity": true, "face": true, "hair": true, "body": true, "clothing": true, "pose": true, "background": true, "lighting": true, "composition": true, "untouched": true}
			for _, item := range preserved {
				if !allowed[item] {
					http.Error(w, "invalid identity preservation item", http.StatusBadRequest)
					return
				}
			}
		}
		systemPrompt += mediaprompt.EditModuleContext(preset, preserved)
	}
	payload := map[string]any{
		"model": cfg.PromptEnhancement.Model,
		"messages": []map[string]any{
			{"role": "system", "content": systemPrompt},
			{"role": "user", "content": userContent},
		},
		"max_completion_tokens": cfg.PromptEnhancement.MaxTokens,
		"temperature":           0,
		"top_k":                 1,
		"seed":                  42,
		"reasoning_effort":      "none",
	}
	endpoint := cfg.Engines["prompt"].Endpoint
	data, _, err := s.callJSON(endpoint+"/v1/chat/completions", payload)
	if err != nil {
		http.Error(w, "prompt enhancer: "+err.Error(), http.StatusBadGateway)
		return
	}
	var response struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(data, &response); err != nil || len(response.Choices) == 0 {
		http.Error(w, "prompt enhancer returned an invalid response", http.StatusBadGateway)
		return
	}
	enhanced := cleanEnhancedPrompt(response.Choices[0].Message.Content)
	if enhanced == "" || strings.EqualFold(enhanced, "IMAGE_NOT_AVAILABLE") {
		http.Error(w, "prompt enhancer returned no usable prompt", http.StatusBadGateway)
		return
	}
	fallback := false
	if !enhancementPreservesEditContract(mode, original, enhanced) {
		enhanced = original
		fallback = true
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"original_prompt": original,
		"enhanced_prompt": enhanced,
		"mode":            mode,
		"image_used":      imageUsed,
		"fallback":        fallback,
	})
}

func enhancementPreservesEditContract(mode, original, enhanced string) bool {
	if mode != "edit" && mode != "edit_control" {
		return true
	}
	source := strings.ToLower(original)
	result := strings.ToLower(enhanced)
	requireAny := func(trigger string, words ...string) bool {
		if !strings.Contains(source, trigger) {
			return true
		}
		for _, word := range words {
			if strings.Contains(result, word) {
				return true
			}
		}
		return false
	}
	if !requireAny("supporting reference", "supporting reference", "reference image", "reference outfit") ||
		!requireAny("depth", "depth") ||
		!requireAny("pose", "pose", "posture", "body orientation") ||
		!requireAny("clothing", "clothing", "outfit", "garment") {
		return false
	}
	if strings.Contains(source, "do not preserve") || strings.Contains(source, "do not retain") || strings.Contains(source, "do not restore") || strings.Contains(source, "may change") {
		for _, contradiction := range []string{
			"preserve original pose", "preserve the original pose", "preserving original pose", "preserving the original pose", "retain original pose", "retain the original pose", "maintain original pose", "maintain the original pose", "original pose remains", "original pose unchanged",
			"preserve original clothing", "preserve the original clothing", "preserving original clothing", "preserving the original clothing", "retain original clothing", "retain the original clothing", "maintain original clothing", "maintain the original clothing", "original clothing remains", "original clothing unchanged",
			"preserve original outfit", "preserve the original outfit", "preserving original outfit", "preserving the original outfit", "retain original outfit", "retain the original outfit", "maintain original outfit", "maintain the original outfit", "original outfit remains", "original outfit unchanged",
		} {
			if strings.Contains(result, contradiction) {
				return false
			}
		}
	}
	return true
}

func (s *Server) callJSON(url string, payload any) ([]byte, string, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, "", err
	}
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, "", err
	}
	req.Header.Set("Content-Type", "application/json")
	return s.do(req)
}

func (s *Server) callMultipart(url string, fields map[string]string, fileField string, paths []string) ([]byte, string, error) {
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	for k, v := range fields {
		_ = mw.WriteField(k, v)
	}
	for _, p := range paths {
		f, e := os.Open(p)
		if e != nil {
			return nil, "", e
		}
		part, e := mw.CreateFormFile(fileField, filepath.Base(p))
		if e == nil {
			_, e = io.Copy(part, f)
		}
		f.Close()
		if e != nil {
			return nil, "", e
		}
	}
	_ = mw.Close()
	req, _ := http.NewRequest(http.MethodPost, url, &body)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	return s.do(req)
}

func (s *Server) callMultipartToFile(url string, fields map[string]string, fileField string, paths []string, output string) error {
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	for k, v := range fields {
		_ = mw.WriteField(k, v)
	}
	for _, p := range paths {
		f, err := os.Open(p)
		if err != nil {
			return err
		}
		part, err := mw.CreateFormFile(fileField, filepath.Base(p))
		if err == nil {
			_, err = io.Copy(part, f)
		}
		_ = f.Close()
		if err != nil {
			return err
		}
	}
	if err := mw.Close(); err != nil {
		return err
	}
	req, err := http.NewRequest(http.MethodPost, url, &body)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())
	resp, err := s.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return fmt.Errorf("engine returned %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	dst, err := os.Create(output)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(dst, resp.Body)
	closeErr := dst.Close()
	if copyErr != nil {
		return copyErr
	}
	return closeErr
}

func (s *Server) do(req *http.Request) ([]byte, string, error) {
	resp, e := s.client.Do(req)
	if e != nil {
		return nil, "", e
	}
	defer resp.Body.Close()
	data, e := io.ReadAll(io.LimitReader(resp.Body, 100<<20))
	if e != nil {
		return nil, "", e
	}
	if resp.StatusCode/100 != 2 {
		return nil, "", fmt.Errorf("engine returned %d: %s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	return data, resp.Header.Get("Content-Type"), nil
}

func (s *Server) fail(j jobs.Job, err error) {
	if current, ok := s.jobs.Get(j.ID); ok && current.Status == "cancelled" {
		return
	}
	log.Printf("job %s failed: %v", j.ID, err)
	j.Status = "failed"
	j.Error = err.Error()
	_ = s.jobs.Save(j)
}

func (s *Server) jobCancelled(id string) bool {
	j, ok := s.jobs.Get(id)
	return ok && j.Status == "cancelled"
}
func (s *Server) engineStates(w http.ResponseWriter, _ *http.Request) {
	cfg := s.config()
	type state struct {
		Kind   string `json:"kind"`
		Status string `json:"status"`
	}
	states := make([]state, 0, 10)
	probe := func(endpoint, healthPath string) string {
		status := "offline"
		resp, err := s.health.Get(endpoint + healthPath)
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				status = "online"
			}
		}
		return status
	}
	defaultImageStatus := "offline"
	for _, mode := range []string{"create", "edit", "control"} {
		backend, ok := cfg.Image.Backends[mode]
		if !ok {
			continue
		}
		status := probe(backend.Endpoint, "/health")
		states = append(states, state{Kind: "image_" + mode, Status: status})
		if mode == cfg.Image.DefaultMode {
			defaultImageStatus = status
		}
	}
	states = append(states, state{Kind: "image", Status: defaultImageStatus})
	for _, kind := range []string{"speech", "recognition", "video", "prompt", "media", "trainer", "upscale", "garment"} {
		healthPath := "/health"
		if kind == "prompt" {
			healthPath = "/v1/models"
		}
		status := probe(cfg.Engines[kind].Endpoint, healthPath)
		states = append(states, state{Kind: kind, Status: status})
	}
	writeJSON(w, 200, states)
}

type imageJobInputInfo struct {
	Role string `json:"role"`
	Name string `json:"name"`
	URL  string `json:"url"`
	Ref  string `json:"ref"`
}

func (s *Server) imageJobInputs(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	job, ok := s.jobs.Get(id)
	if !ok || job.Kind != "image" {
		http.NotFound(w, r)
		return
	}
	inputs := make([]imageJobInputInfo, 0, 8)
	for _, role := range []string{"reference", "identity", "identity_reference", "identity_mask", "strict_mask", "depth", "vision", "style_reference", "nk2e", "anypaint", "anypaint_mask", "garment_source", "garment_reference"} {
		files, err := s.imageInputFiles(id, role)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		for index, path := range files {
			name := filepath.Base(path)
			switch role {
			case "identity":
				name = "identity" + filepath.Ext(path)
			case "identity_reference":
				name = fmt.Sprintf("identity-reference-%d%s", index+1, filepath.Ext(path))
			case "identity_mask":
				name = "identity-focus-mask" + filepath.Ext(path)
			case "strict_mask":
				name = "strict-change-mask" + filepath.Ext(path)
			case "depth":
				name = "depth" + filepath.Ext(path)
			case "vision":
				name = fmt.Sprintf("vision-%d%s", index+1, filepath.Ext(path))
			case "style_reference":
				name = fmt.Sprintf("style-reference-%d%s", index+1, filepath.Ext(path))
			case "nk2e":
				name = "nk2e" + filepath.Ext(path)
			case "anypaint":
				name = "anypaint-source" + filepath.Ext(path)
			case "anypaint_mask":
				name = "anypaint-mask" + filepath.Ext(path)
			case "garment_source":
				name = "garment-source" + filepath.Ext(path)
			case "garment_reference":
				name = fmt.Sprintf("garment-reference-%d%s", index+1, filepath.Ext(path))
			}
			inputs = append(inputs, imageJobInputInfo{
				Role: role,
				Name: name,
				URL:  fmt.Sprintf("/api/jobs/%s/inputs/%s/%d", id, role, index),
				Ref:  fmt.Sprintf("%s:%s:%d", id, role, index),
			})
		}
	}
	writeJSON(w, http.StatusOK, inputs)
}

func (s *Server) imageJobInput(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	job, ok := s.jobs.Get(id)
	if !ok || job.Kind != "image" {
		http.NotFound(w, r)
		return
	}
	index, err := strconv.Atoi(r.PathValue("index"))
	if err != nil || index < 0 {
		http.NotFound(w, r)
		return
	}
	files, err := s.imageInputFiles(id, r.PathValue("role"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if index >= len(files) {
		http.NotFound(w, r)
		return
	}
	http.ServeFile(w, r, files[index])
}

func (s *Server) imageInputFiles(id, role string) ([]string, error) {
	root := filepath.Join(s.dataDir, "inputs", id)
	dir := root
	switch role {
	case "output":
		job, ok := s.jobs.Get(id)
		if !ok || job.Kind != "image" || job.Status != "completed" || job.OutputURL == "" {
			return nil, nil
		}
		path := s.jobs.OutputPath(filepath.Base(job.OutputURL))
		if _, err := os.Stat(path); err != nil {
			if errors.Is(err, os.ErrNotExist) {
				return nil, nil
			}
			return nil, err
		}
		return []string{path}, nil
	case "reference":
	case "identity":
		dir = filepath.Join(root, "identity")
	case "identity_reference":
		dir = filepath.Join(root, "identity-reference")
	case "identity_mask":
		dir = filepath.Join(root, "identity-mask")
	case "strict_mask":
		dir = filepath.Join(root, "strict-mask")
	case "depth":
		dir = filepath.Join(root, "depth")
	case "vision":
		dir = filepath.Join(root, "vision")
	case "style_reference":
		dir = filepath.Join(root, "style-reference")
	case "nk2e":
		dir = filepath.Join(root, "nk2e")
	case "anypaint":
		dir = filepath.Join(root, "anypaint")
	case "anypaint_mask":
		dir = filepath.Join(root, "anypaint-mask")
	case "garment_source":
		dir = filepath.Join(root, "garment-source")
	case "garment_reference":
		dir = filepath.Join(root, "garment-reference")
	default:
		return nil, nil
	}
	entries, err := os.ReadDir(dir)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	files := make([]string, 0, len(entries))
	for _, entry := range entries {
		if entry.Type().IsRegular() {
			files = append(files, filepath.Join(dir, entry.Name()))
		}
	}
	sort.Strings(files)
	return files, nil
}

func (s *Server) appendReusedImageInputs(r *http.Request, field, dir string, max int, paths []string) ([]string, error) {
	tokens := r.MultipartForm.Value[field]
	if len(paths)+len(tokens) > max {
		return nil, fmt.Errorf("too many files for %s (maximum %d)", field, max)
	}
	if len(tokens) == 0 {
		return paths, nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	result := append([]string(nil), paths...)
	for _, token := range tokens {
		parts := strings.Split(token, ":")
		if len(parts) != 3 || parts[0] == "" || parts[1] == "" {
			return nil, fmt.Errorf("invalid stored image reference")
		}
		job, ok := s.jobs.Get(parts[0])
		if !ok || job.Kind != "image" {
			return nil, fmt.Errorf("stored image reference no longer exists")
		}
		index, err := strconv.Atoi(parts[2])
		if err != nil || index < 0 {
			return nil, fmt.Errorf("invalid stored image reference")
		}
		files, err := s.imageInputFiles(parts[0], parts[1])
		if err != nil {
			return nil, err
		}
		if index >= len(files) {
			return nil, fmt.Errorf("stored image reference no longer exists")
		}
		source := files[index]
		destination := filepath.Join(dir, fmt.Sprintf("%d%s", len(result), strings.ToLower(filepath.Ext(source))))
		if err := linkOrCopyFile(source, destination); err != nil {
			return nil, err
		}
		result = append(result, destination)
	}
	return result, nil
}

func linkOrCopyFile(source, destination string) error {
	if err := os.Link(source, destination); err == nil {
		return nil
	}
	input, err := os.Open(source)
	if err != nil {
		return err
	}
	defer input.Close()
	output, err := os.OpenFile(destination, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(output, input)
	closeErr := output.Close()
	if copyErr != nil {
		_ = os.Remove(destination)
		return copyErr
	}
	if closeErr != nil {
		_ = os.Remove(destination)
		return closeErr
	}
	return nil
}

func (s *Server) getJob(w http.ResponseWriter, r *http.Request) {
	j, ok := s.jobs.Get(r.PathValue("id"))
	if !ok {
		http.NotFound(w, r)
		return
	}
	writeJSON(w, 200, j)
}

func (s *Server) deleteJob(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	j, ok := s.jobs.Get(id)
	if !ok {
		http.NotFound(w, r)
		return
	}
	if j.Status == "queued" || j.Status == "running" {
		http.Error(w, jobs.ErrActive.Error(), http.StatusConflict)
		return
	}
	if err := s.deleteMediaAsset(j.MediaAssetID); err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	err := s.jobs.Delete(id)
	switch {
	case err == nil:
		_ = os.Remove(s.videoPreviewPath(id))
		w.WriteHeader(http.StatusNoContent)
	case errors.Is(err, jobs.ErrNotFound):
		http.NotFound(w, r)
	case errors.Is(err, jobs.ErrActive):
		http.Error(w, err.Error(), http.StatusConflict)
	default:
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func (s *Server) cancelJob(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	s.generationStateMu.Lock()
	defer s.generationStateMu.Unlock()
	j, ok := s.jobs.Get(id)
	if !ok {
		http.NotFound(w, r)
		return
	}
	if j.Status != "queued" && j.Status != "running" {
		http.Error(w, "job is not active", http.StatusConflict)
		return
	}
	if j.Kind != "recognition" && !(isGenerationKind(j.Kind) && j.Status == "queued") {
		http.Error(w, "running generation jobs cannot be cancelled safely", http.StatusConflict)
		return
	}
	if j.Params == nil {
		j.Params = map[string]any{}
	}
	j.Status = "cancelled"
	j.Error = ""
	j.Params["stage"] = "cancelled"
	delete(j.Params, "media_eta_seconds")
	if err := s.jobs.Save(j); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if j.Kind == "recognition" {
		s.cancelMediaPreparation(id)
	} else {
		s.wakeGenerationQueue()
	}
	writeJSON(w, http.StatusOK, j)
}

func (s *Server) deleteFinishedJobs(w http.ResponseWriter, _ *http.Request) {
	deleted := 0
	for _, j := range s.jobs.List() {
		if j.Status == "queued" || j.Status == "running" {
			continue
		}
		if err := s.deleteMediaAsset(j.MediaAssetID); err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
		if err := s.jobs.Delete(j.ID); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		_ = os.Remove(s.videoPreviewPath(j.ID))
		deleted++
	}
	writeJSON(w, http.StatusOK, map[string]int{"deleted": deleted})
}

func validAssetID(id string) bool {
	if len(id) != 32 {
		return false
	}
	for _, char := range id {
		if !strings.ContainsRune("0123456789abcdef", char) {
			return false
		}
	}
	return true
}

func (s *Server) proxyMediaAsset(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if !validAssetID(id) {
		http.NotFound(w, r)
		return
	}
	target := strings.TrimRight(s.config().Engines["media"].Endpoint, "/") + "/v1/media/assets/" + id
	request, err := http.NewRequestWithContext(r.Context(), r.Method, target, nil)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	for _, header := range []string{"Range", "If-Range", "If-Modified-Since", "If-None-Match"} {
		if value := r.Header.Get(header); value != "" {
			request.Header.Set(header, value)
		}
	}
	response, err := http.DefaultTransport.RoundTrip(request)
	if err != nil {
		http.Error(w, "media stream: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer response.Body.Close()
	for _, header := range []string{"Accept-Ranges", "Content-Disposition", "Content-Length", "Content-Range", "Content-Type", "ETag", "Last-Modified"} {
		if values := response.Header.Values(header); len(values) > 0 {
			w.Header()[header] = append([]string(nil), values...)
		}
	}
	w.WriteHeader(response.StatusCode)
	if r.Method != http.MethodHead {
		_, _ = io.Copy(w, response.Body)
	}
}

func (s *Server) deleteMediaAsset(id string) error {
	if id == "" {
		return nil
	}
	if !validAssetID(id) {
		return fmt.Errorf("invalid media asset id")
	}
	target := strings.TrimRight(s.config().Engines["media"].Endpoint, "/") + "/v1/media/assets/" + id
	request, err := http.NewRequest(http.MethodDelete, target, nil)
	if err != nil {
		return err
	}
	response, err := s.health.Do(request)
	if err != nil {
		return fmt.Errorf("delete media asset: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusNoContent && response.StatusCode != http.StatusNotFound {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 1<<20))
		return fmt.Errorf("delete media asset: engine returned %d: %s", response.StatusCode, strings.TrimSpace(string(body)))
	}
	return nil
}

func saveUploads(r *http.Request, field, dir string, max int) ([]string, error) {
	files := r.MultipartForm.File[field]
	if len(files) > max {
		return nil, fmt.Errorf("too many files (max %d)", max)
	}
	if len(files) == 0 {
		return nil, nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	out := make([]string, 0, len(files))
	for i, h := range files {
		src, e := h.Open()
		if e != nil {
			return nil, e
		}
		// Reference pixels are conditioning data, not merely previews. Saving a
		// PNG upload or a CDN response back as lossy WebP changed alpha edges and
		// was enough to make Krea Identity Edit retain the original clothing.
		// Decode every supported upload and persist one lossless PNG representation
		// so direct uploads, URL images and later job retries use identical pixels.
		data, readErr := io.ReadAll(io.LimitReader(src, (32<<20)+1))
		src.Close()
		if readErr != nil {
			return nil, readErr
		}
		if len(data) == 0 || len(data) > 32<<20 {
			return nil, fmt.Errorf("image upload must be between 1 byte and 32 MiB")
		}
		decoded, _, decodeErr := image.Decode(bytes.NewReader(data))
		if decodeErr != nil {
			// Preserve the old opaque-upload behavior for non-image engine test
			// fixtures and forward-compatible formats that this Go build cannot
			// decode. Recognized images always take the lossless PNG path below.
			name := fmt.Sprintf("%d%s", i, strings.ToLower(filepath.Ext(h.Filename)))
			dstPath := filepath.Join(dir, name)
			if writeErr := os.WriteFile(dstPath, data, 0o644); writeErr != nil {
				return nil, writeErr
			}
			out = append(out, dstPath)
			continue
		}
		name := fmt.Sprintf("%d.png", i)
		dstPath := filepath.Join(dir, name)
		dst, e := os.OpenFile(dstPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
		if e != nil {
			return nil, e
		}
		encodeErr := png.Encode(dst, decoded)
		closeErr := dst.Close()
		if encodeErr != nil {
			_ = os.Remove(dstPath)
			return nil, encodeErr
		}
		if closeErr != nil {
			_ = os.Remove(dstPath)
			return nil, closeErr
		}
		out = append(out, dstPath)
	}
	return out, nil
}

func newID() string { b := make([]byte, 12); _, _ = rand.Read(b); return hex.EncodeToString(b) }
func formInt(r *http.Request, k string, d int) int {
	v, e := strconv.Atoi(r.FormValue(k))
	if e != nil {
		return d
	}
	return v
}
func formInt64(r *http.Request, k string, d int64) int64 {
	v, e := strconv.ParseInt(r.FormValue(k), 10, 64)
	if e != nil {
		return d
	}
	return v
}
func formFloat64(r *http.Request, k string, d float64) float64 {
	v, e := strconv.ParseFloat(r.FormValue(k), 64)
	if e != nil {
		return d
	}
	return v
}
func valueOr(v, d string) string {
	if strings.TrimSpace(v) == "" {
		return d
	}
	return v
}

func valueIfDifferent(value, original string) string {
	if value == original {
		return ""
	}
	return value
}

func cleanEnhancedPrompt(value string) string {
	value = strings.NewReplacer(
		"\u2018", "'", "\u2019", "'", "\u201c", "\"", "\u201d", "\"",
		"\u2014", "--", "\u2013", "-", "\u00a0", " ", "\u2212", "-",
	).Replace(strings.TrimSpace(value))
	for index, char := range value {
		if unicode.IsLetter(char) {
			return strings.TrimSpace(value[index:])
		}
	}
	return ""
}

func decodeImage(data []byte) ([]byte, error) {
	var response struct {
		Data []struct {
			B64JSON string `json:"b64_json"`
		} `json:"data"`
	}
	if err := json.Unmarshal(data, &response); err != nil {
		return nil, fmt.Errorf("decode image response: %w", err)
	}
	if len(response.Data) == 0 || response.Data[0].B64JSON == "" {
		return nil, fmt.Errorf("image engine returned no image")
	}
	decoded, err := base64.StdEncoding.DecodeString(response.Data[0].B64JSON)
	if err != nil {
		return nil, fmt.Errorf("decode generated image: %w", err)
	}
	return decoded, nil
}

func decodeImageSeed(data []byte) (int64, bool) {
	var response struct {
		Seed *int64 `json:"seed"`
	}
	if err := json.Unmarshal(data, &response); err != nil || response.Seed == nil {
		return 0, false
	}
	return *response.Seed, true
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
func withLog(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start).Round(time.Millisecond))
	})
}

func spaHandler(root fs.FS) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p := strings.TrimPrefix(filepath.Clean(r.URL.Path), "/")
		if p == "." {
			p = "index.html"
		}
		if _, e := fs.Stat(root, p); e != nil {
			p = "index.html"
		}
		http.ServeFileFS(w, r, root, p)
	})
}
