package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"io"
	"mediaapp/internal/jobs"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

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

func (s *Server) runImageDetailEnhance(ctx context.Context, j jobs.Job, source []byte, strength float64, seed int64, vae string) {
	cfg := s.config()
	s.heavyMu.Lock()
	defer s.heavyMu.Unlock()
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
	if err := s.prepareKreaRequest(ctx, &j, generationModelPlan(j), request); err != nil {
		if s.requeueGenerationAfterEngineConflict(j, err) {
			return
		}
		s.fail(j, err)
		return
	}
	response, err := s.generateImageWithEngine(ctx, backend, request)
	if err != nil {
		if s.requeueGenerationAfterEngineConflict(j, err) {
			return
		}
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

func (s *Server) runImageUpscale(ctx context.Context, j jobs.Job, source []byte, scale int, seed int64) {
	s.heavyMu.Lock()
	defer s.heavyMu.Unlock()
	request := map[string]any{
		"model": "seedvr2-3b-fp8", "image": base64.StdEncoding.EncodeToString(source),
		"scale": scale, "response_format": "b64_json", "output_format": "png",
		"operation_id": j.ID,
	}
	if seed >= 0 {
		request["seed"] = seed
	}
	endpoint := s.config().Engines["upscale"].Endpoint
	observer := s.startRuntimeObserver(ctx, j.ID, endpoint)
	response, err := s.upscaleImageWithEngine(ctx, request)
	observer.Stop()
	if err != nil {
		if s.requeueGenerationAfterEngineConflict(j, err) {
			return
		}
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
