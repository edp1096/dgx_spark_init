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
	_ "image/png"
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
	cfgMu      sync.RWMutex
	heavyMu    sync.Mutex
	cfg        config.Config
	configPath string
	dataDir    string
	jobs       *jobs.Store
	client     *http.Client
	health     *http.Client
	web        fs.FS
}

type imageGenerationOptions struct {
	identityPath      string
	identityRefPath   string
	identityMaskPath  string
	strictMaskPath    string
	strictMaskGrow    int
	strictMaskFeather float64
	vaeMode           string
	identityFitMode   string
	depthPath         string
	identityStrength  float64
	refBoost          float64
	groundingPX       int
	steps             int
	style             string
	styleStrength     float64
	styles            []styleSelection
	userLoras         []userLoRASelection
	depthStrength     float64
	visionPaths       []string
	visionMode        string
	visionMegapixels  float64
	styleRefPaths     []string
	styleRefStrength  float64
	nk2ePath          string
	nk2eMode          string
	nk2eStrength      float64
	nk2ePreprocessed  bool
	anypaintPath      string
	anypaintMaskPath  string
	outpaintLeft      int
	outpaintTop       int
	outpaintRight     int
	outpaintBottom    int
	anypaintStrength  float64
	anypaintBoundary  int
	filterMode        string
	filterStrength    float64
	promptEnhancer    bool
	promptEnhStrength float64
	promptTextScale   float64
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
		web:    web,
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
	mux.HandleFunc("GET /api/jobs", func(w http.ResponseWriter, _ *http.Request) { writeJSON(w, 200, s.jobs.List()) })
	mux.HandleFunc("DELETE /api/jobs", s.deleteFinishedJobs)
	mux.HandleFunc("GET /api/jobs/{id}", s.getJob)
	mux.HandleFunc("GET /api/jobs/{id}/exif", s.imageJobEXIF)
	mux.HandleFunc("GET /api/jobs/{id}/inputs", s.imageJobInputs)
	mux.HandleFunc("GET /api/jobs/{id}/inputs/{role}/{index}", s.imageJobInput)
	mux.HandleFunc("DELETE /api/jobs/{id}", s.deleteJob)
	mux.HandleFunc("POST /api/jobs/{id}/cancel", s.cancelJob)
	mux.HandleFunc("POST /api/jobs/{id}/retry", s.retrySubtitle)
	mux.HandleFunc("POST /api/jobs/image", s.createImage)
	mux.HandleFunc("POST /api/jobs/{id}/upscale", s.createImageUpscale)
	mux.HandleFunc("POST /api/jobs/{id}/detail-enhance", s.createImageDetailEnhance)
	mux.HandleFunc("POST /api/jobs/speech", s.createSpeech)
	mux.HandleFunc("POST /api/jobs/recognition", s.createSubtitle)
	mux.HandleFunc("POST /api/media/options", s.mediaOptions)
	mux.HandleFunc("GET /api/storage", s.mediaStorage)
	mux.HandleFunc("DELETE /api/storage/temp", s.cleanupMediaTemp)
	mux.HandleFunc("POST /api/jobs/video", s.createVideo)
	mux.HandleFunc("POST /api/prompts/enhance", s.enhancePrompt)
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

func (s *Server) proxyLoRA(w http.ResponseWriter, r *http.Request) {
	endpoint := s.config().Engines["trainer"].Endpoint
	target, err := url.Parse(endpoint)
	if err != nil || target.Host == "" {
		http.Error(w, "invalid LoRA trainer endpoint", http.StatusBadGateway)
		return
	}
	proxy := httputil.NewSingleHostReverseProxy(target)
	proxy.ErrorHandler = func(w http.ResponseWriter, _ *http.Request, err error) {
		http.Error(w, "LoRA trainer unavailable: "+err.Error(), http.StatusBadGateway)
	}
	originalDirector := proxy.Director
	proxy.Director = func(request *http.Request) {
		originalDirector(request)
		path := strings.TrimPrefix(request.URL.Path, "/api/lora")
		if path == "" {
			path = "/"
		}
		request.URL.Path = path
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
	if err := r.ParseMultipartForm(40 << 20); err != nil {
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
	refs, err := saveUploads(r, "image", inputDir, 1)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	imagePath := ""
	if len(refs) > 0 {
		imagePath = refs[0]
	}
	j := jobs.Job{
		ID: id, Kind: "video", Status: "queued", Prompt: originalPrompt,
		Params:    map[string]any{"width": width, "height": height, "num_frames": frames, "fps": fps, "seed": seed, "image_strength": strength, "image": imagePath != "", "enhanced_prompt": valueIfDifferent(effectivePrompt, originalPrompt)},
		CreatedAt: time.Now(),
	}
	if err := s.jobs.Save(j); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	go s.runVideo(j, effectivePrompt, imagePath, width, height, frames, fps, seed, strength)
	writeJSON(w, http.StatusAccepted, j)
}

func (s *Server) createImage(w http.ResponseWriter, r *http.Request) {
	cfg := s.config()
	if err := r.ParseMultipartForm(80 << 20); err != nil {
		http.Error(w, "invalid form", 400)
		return
	}
	effectivePrompt := strings.TrimSpace(r.FormValue("prompt"))
	if effectivePrompt == "" {
		http.Error(w, "prompt is required", 400)
		return
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
		identityStrength:  formFloat64(r, "identity_strength", 1),
		refBoost:          formFloat64(r, "ref_boost", 4),
		groundingPX:       formInt(r, "grounding_px", 768),
		steps:             formInt(r, "steps", 0),
		style:             strings.ToLower(strings.TrimSpace(r.FormValue("style"))),
		styleStrength:     formFloat64(r, "style_strength", 1),
		depthStrength:     formFloat64(r, "depth_strength", 0.8),
		visionMode:        strings.ToLower(strings.TrimSpace(r.FormValue("vision_mode"))),
		visionMegapixels:  formFloat64(r, "vision_megapixels", 1),
		styleRefStrength:  formFloat64(r, "style_reference_strength", 1),
		nk2eMode:          strings.ToLower(strings.TrimSpace(r.FormValue("nk2e_mode"))),
		nk2eStrength:      formFloat64(r, "nk2e_strength", 0.7),
		outpaintLeft:      formInt(r, "outpaint_left", 0),
		outpaintTop:       formInt(r, "outpaint_top", 0),
		outpaintRight:     formInt(r, "outpaint_right", 0),
		outpaintBottom:    formInt(r, "outpaint_bottom", 0),
		anypaintStrength:  formFloat64(r, "anypaint_strength", 1),
		anypaintBoundary:  formInt(r, "anypaint_boundary_redraw_px", 32),
		strictMaskGrow:    formInt(r, "strict_mask_grow", 0),
		strictMaskFeather: formFloat64(r, "strict_mask_feather", 0),
		vaeMode:           strings.ToLower(strings.TrimSpace(r.FormValue("vae_mode"))),
		identityFitMode:   strings.ToLower(strings.TrimSpace(r.FormValue("identity_fit_mode"))),
		nk2ePreprocessed:  strings.EqualFold(r.FormValue("nk2e_preprocessed"), "true"),
		filterMode:        strings.ToLower(strings.TrimSpace(r.FormValue("filter_mode"))),
		filterStrength:    formFloat64(r, "filter_strength", 1),
		promptEnhancer:    strings.EqualFold(r.FormValue("prompt_enhancer"), "true"),
		promptEnhStrength: formFloat64(r, "prompt_enhancer_strength", 1),
		promptTextScale:   formFloat64(r, "prompt_text_scale", 1.75),
	}
	if krea.vaeMode == "" {
		krea.vaeMode = "default"
	}
	if krea.identityFitMode == "" {
		krea.identityFitMode = "fit"
	}
	if krea.filterMode == "" {
		krea.filterMode = "balanced"
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
		identityRef, uploadErr := saveUploads(r, "identity_reference", filepath.Join(inputDir, "identity-reference"), 1)
		if uploadErr != nil {
			http.Error(w, uploadErr.Error(), http.StatusBadRequest)
			return
		}
		identityRef, uploadErr = s.appendReusedImageInputs(r, "reuse_identity_reference", filepath.Join(inputDir, "identity-reference"), 1, identityRef)
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
		if len(identityRef) > 0 {
			krea.identityRefPath = identityRef[0]
		}
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
		if krea.identityRefPath != "" && krea.identityPath == "" {
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
		if krea.filterMode != "off" && krea.filterMode != "balanced" && krea.filterMode != "strong" {
			http.Error(w, "filter mode must be off, balanced, or strong", http.StatusBadRequest)
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
			if selection.Filename == "" || filepath.Base(selection.Filename) != selection.Filename || !strings.HasSuffix(strings.ToLower(selection.Filename), ".safetensors") || seenUserLoras[selection.Filename] || selection.Strength < 0 || selection.Strength > 2 {
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
		params["identity"] = krea.identityPath != ""
		params["identity_reference"] = krea.identityRefPath != ""
		params["depth"] = krea.depthPath != ""
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
		params["strict_mask_grow"] = krea.strictMaskGrow
		params["strict_mask_feather"] = krea.strictMaskFeather
		params["filter_mode"] = krea.filterMode
		params["filter_strength"] = krea.filterStrength
		params["prompt_enhancer"] = krea.promptEnhancer
		params["prompt_enhancer_strength"] = krea.promptEnhStrength
		params["prompt_text_scale"] = krea.promptTextScale
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
			params["grounding_px"] = krea.groundingPX
			params["steps"] = krea.steps
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
	j := jobs.Job{ID: id, Kind: "image", Status: "queued", Prompt: originalPrompt, Params: params, CreatedAt: time.Now()}
	if err := s.jobs.Save(j); err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	go s.runImage(j, effectivePrompt, refs, width, height, seed, mode, controlType, controlStrength, krea)
	writeJSON(w, 202, j)
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
	j := jobs.Job{ID: id, Kind: "speech", Status: "queued", Prompt: text, Params: map[string]any{"language": language, "speaker": speaker, "instructions": instructions, "seed": seed}, CreatedAt: time.Now()}
	if err := s.jobs.Save(j); err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	go s.runSpeech(j, language, speaker, instructions, seed)
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
		"width": input.Width, "height": input.Height,
	}
	j := jobs.Job{ID: id, Kind: "image", Status: "queued", Prompt: source.Prompt, Params: params, CreatedAt: time.Now()}
	if err := s.jobs.Save(j); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	go s.runImageDetailEnhance(j, data, request.Strength, request.Seed, request.VAE)
	writeJSON(w, http.StatusAccepted, j)
}

func (s *Server) runImageDetailEnhance(j jobs.Job, source []byte, strength float64, seed int64, vae string) {
	cfg := s.config()
	s.heavyMu.Lock()
	defer s.heavyMu.Unlock()
	j.Status = "running"
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
	}
	if seed >= 0 {
		request["seed"] = seed
	}
	response, _, err := s.callJSON(backend.Endpoint+"/v1/images/generations", request)
	if err != nil {
		s.fail(j, err)
		return
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
	}
	if enhanced, exists := source.Params["enhanced_prompt"]; exists {
		params["source_enhanced_prompt"] = enhanced
	}
	j := jobs.Job{ID: id, Kind: "image", Status: "queued", Prompt: source.Prompt, Params: params, CreatedAt: time.Now()}
	if err := s.jobs.Save(j); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	go s.runImageUpscale(j, data, request.Scale, request.Seed)
	writeJSON(w, http.StatusAccepted, j)
}

func (s *Server) runImageUpscale(j jobs.Job, source []byte, scale int, seed int64) {
	cfg := s.config()
	s.heavyMu.Lock()
	defer s.heavyMu.Unlock()
	j.Status = "running"
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
	_ = s.jobs.Save(j)
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
			"size": fmt.Sprintf("%dx%d", width, height), "response_format": "b64_json", "output_format": "png",
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
			"size": fmt.Sprintf("%dx%d", width, height), "response_format": "b64_json", "output_format": "png",
			"filter_mode": krea.filterMode, "filter_strength": krea.filterStrength,
			"prompt_enhancer": krea.promptEnhancer, "prompt_enhancer_strength": krea.promptEnhStrength,
			"prompt_text_scale": krea.promptTextScale,
		}
		for field, path := range map[string]string{
			"source_image": krea.identityPath, "reference_image": krea.identityRefPath, "control_image": krea.depthPath, "nk2e_image": krea.nk2ePath,
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
			"vision_images": krea.visionPaths, "style_reference_images": krea.styleRefPaths,
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
			request["grounding_px"] = krea.groundingPX
			request["strict_mask_grow"] = krea.strictMaskGrow
			request["strict_mask_feather"] = krea.strictMaskFeather
			request["vae_mode"] = krea.vaeMode
			request["identity_fit_mode"] = krea.identityFitMode
		}
		if krea.depthPath != "" {
			request["control_strength"] = krea.depthStrength
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

func (s *Server) runVideo(j jobs.Job, effectivePrompt, imagePath string, width, height, frames int, fps float64, seed int64, strength float64) {
	cfg := s.config()
	s.heavyMu.Lock()
	defer s.heavyMu.Unlock()
	j.Status = "running"
	_ = s.jobs.Save(j)
	fields := map[string]string{
		"prompt": effectivePrompt,
		"width":  strconv.Itoa(width), "height": strconv.Itoa(height),
		"num_frames": strconv.Itoa(frames), "fps": strconv.FormatFloat(fps, 'f', -1, 64),
		"seed": strconv.FormatInt(seed, 10), "image_strength": strconv.FormatFloat(strength, 'f', -1, 64),
	}
	paths := []string{}
	if imagePath != "" {
		paths = append(paths, imagePath)
	}
	name := j.ID + ".mp4"
	output := s.jobs.OutputPath(name)
	endpoint := cfg.Engines["video"].Endpoint
	if err := s.callMultipartToFile(endpoint+"/v1/videos/generations", fields, "image", paths, output); err != nil {
		_ = os.Remove(output)
		s.fail(j, err)
		return
	}
	j.Status = "completed"
	j.OutputURL = "/api/outputs/" + name
	_ = s.jobs.Save(j)
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
	if mode != "t2v" && mode != "i2v" && mode != "t2i" && mode != "edit" {
		http.Error(w, "mode must be t2i, t2v or i2v", http.StatusBadRequest)
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

	payload := map[string]any{
		"model": cfg.PromptEnhancement.Model,
		"messages": []map[string]any{
			{"role": "system", "content": mediaprompt.System(mode, imageUsed)},
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
	writeJSON(w, http.StatusOK, map[string]any{
		"original_prompt": original,
		"enhanced_prompt": enhanced,
		"mode":            mode,
		"image_used":      imageUsed,
	})
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
	for _, kind := range []string{"speech", "recognition", "video", "prompt", "media", "trainer", "upscale"} {
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
	for _, role := range []string{"reference", "identity", "identity_reference", "identity_mask", "strict_mask", "depth", "vision", "style_reference", "nk2e", "anypaint", "anypaint_mask"} {
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
				name = "identity-reference" + filepath.Ext(path)
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
	j, ok := s.jobs.Get(id)
	if !ok {
		http.NotFound(w, r)
		return
	}
	if j.Status != "queued" && j.Status != "running" {
		http.Error(w, "job is not active", http.StatusConflict)
		return
	}
	if j.Kind != "recognition" {
		http.Error(w, "cancellation is currently supported for subtitle jobs", http.StatusConflict)
		return
	}
	j.Status = "cancelled"
	j.Error = ""
	j.Params["stage"] = "cancelled"
	delete(j.Params, "media_eta_seconds")
	if err := s.jobs.Save(j); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	endpoint := strings.TrimRight(s.config().Engines["media"].Endpoint, "/") + "/v1/media/prepare/" + id
	request, err := http.NewRequest(http.MethodDelete, endpoint, nil)
	if err == nil {
		response, requestErr := s.health.Do(request)
		if requestErr == nil {
			_ = response.Body.Close()
		}
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
		name := fmt.Sprintf("%d%s", i, strings.ToLower(filepath.Ext(h.Filename)))
		dstPath := filepath.Join(dir, name)
		dst, e := os.Create(dstPath)
		if e == nil {
			_, e = io.Copy(dst, io.LimitReader(src, 32<<20))
			dst.Close()
		}
		src.Close()
		if e != nil {
			return nil, e
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
