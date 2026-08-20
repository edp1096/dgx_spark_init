package server

import (
	"bytes"
	"crypto/rand"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log"
	"mime"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
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
	cfg        config.Config
	configPath string
	dataDir    string
	jobs       *jobs.Store
	client     *http.Client
	health     *http.Client
	web        fs.FS
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
	mux.HandleFunc("DELETE /api/jobs/{id}", s.deleteJob)
	mux.HandleFunc("POST /api/jobs/image", s.createImage)
	mux.HandleFunc("POST /api/jobs/speech", s.createSpeech)
	mux.HandleFunc("POST /api/jobs/recognition", s.createSubtitle)
	mux.HandleFunc("POST /api/media/options", s.mediaOptions)
	mux.HandleFunc("GET /api/storage", s.mediaStorage)
	mux.HandleFunc("DELETE /api/storage/temp", s.cleanupMediaTemp)
	mux.HandleFunc("POST /api/jobs/video", s.createVideo)
	mux.HandleFunc("POST /api/prompts/enhance", s.enhancePrompt)
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
	prompt := strings.TrimSpace(r.FormValue("prompt"))
	if prompt == "" {
		http.Error(w, "prompt is required", 400)
		return
	}
	id := newID()
	inputDir := filepath.Join(s.dataDir, "inputs", id)
	refs, err := saveUploads(r, "references", inputDir, cfg.Image.MaxReferenceImages)
	if err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	width := formInt(r, "width", cfg.Image.DefaultWidth)
	height := formInt(r, "height", cfg.Image.DefaultHeight)
	seed := formInt64(r, "seed", -1)
	j := jobs.Job{ID: id, Kind: "image", Status: "queued", Prompt: prompt, Params: map[string]any{"width": width, "height": height, "seed": seed, "references": len(refs)}, CreatedAt: time.Now()}
	if err := s.jobs.Save(j); err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	go s.runImage(j, refs, width, height, seed)
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

func (s *Server) runImage(j jobs.Job, refs []string, width, height int, seed int64) {
	cfg := s.config()
	j.Status = "running"
	_ = s.jobs.Save(j)
	endpoint := cfg.Engines["image"].Endpoint
	var response []byte
	var err error
	if len(refs) == 0 {
		request := map[string]any{
			"model": cfg.Image.Model, "prompt": j.Prompt,
			"size": fmt.Sprintf("%dx%d", width, height), "response_format": "b64_json", "output_format": "png",
		}
		if seed >= 0 {
			request["seed"] = seed
		}
		response, _, err = s.callJSON(endpoint+"/v1/images/generations", request)
	} else {
		fields := map[string]string{
			"model": cfg.Image.Model, "prompt": j.Prompt,
			"size": fmt.Sprintf("%dx%d", width, height), "response_format": "b64_json", "output_format": "png",
		}
		if seed >= 0 {
			fields["seed"] = strconv.FormatInt(seed, 10)
		}
		response, _, err = s.callMultipart(endpoint+"/v1/images/edits", fields, "image", refs)
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
	name := j.ID + ".png"
	if err = os.WriteFile(s.jobs.OutputPath(name), data, 0o644); err != nil {
		s.fail(j, err)
		return
	}
	j.Status = "completed"
	j.OutputURL = "/api/outputs/" + name
	_ = s.jobs.Save(j)
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
	if mode != "t2v" && mode != "i2v" {
		http.Error(w, "mode must be t2v or i2v", http.StatusBadRequest)
		return
	}
	if mode == "i2v" && !cfg.PromptEnhancement.VisionEnabled {
		http.Error(w, "I2V prompt enhancement requires a vision-enabled model bundle", http.StatusConflict)
		return
	}

	visionRequested := mode == "i2v" && cfg.PromptEnhancement.VisionEnabled
	userContent := any("user prompt: " + original)
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
	log.Printf("job %s failed: %v", j.ID, err)
	j.Status = "failed"
	j.Error = err.Error()
	_ = s.jobs.Save(j)
}
func (s *Server) engineStates(w http.ResponseWriter, _ *http.Request) {
	cfg := s.config()
	type state struct {
		Kind   string `json:"kind"`
		Status string `json:"status"`
	}
	states := make([]state, 0, 6)
	for _, kind := range []string{"image", "speech", "recognition", "video", "prompt", "media"} {
		status := "offline"
		healthPath := "/health"
		if kind == "prompt" {
			healthPath = "/v1/models"
		}
		resp, err := s.health.Get(cfg.Engines[kind].Endpoint + healthPath)
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				status = "online"
			}
		}
		states = append(states, state{Kind: kind, Status: status})
	}
	writeJSON(w, 200, states)
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
