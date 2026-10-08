package server

import (
	"encoding/json"
	"errors"
	"io"
	"math"
	"mediaapp/internal/jobs"
	"net/http"
	"strings"
	"time"
)

func (s *Server) createVideoUpscale(w http.ResponseWriter, r *http.Request) {
	source, ok := s.jobs.Get(r.PathValue("id"))
	if !ok {
		http.NotFound(w, r)
		return
	}
	if (source.Kind != "video" && source.Kind != "recognition") || (source.Kind == "video" && source.Status != "completed") {
		http.Error(w, "only a completed video or a downloaded transcription source can be upscaled", http.StatusConflict)
		return
	}
	if source.Kind == "video" && source.OutputURL == "" {
		http.Error(w, "source video is no longer available", http.StatusConflict)
		return
	}
	if source.Kind == "recognition" && source.MediaAssetID == "" {
		http.Error(w, "saved transcription video is no longer available", http.StatusConflict)
		return
	}
	width, height := intParam(source.Params, "width", 0), intParam(source.Params, "height", 0)
	duration, _ := numberFromAny(source.Params["duration"])
	fps, _ := numberFromAny(source.Params["fps"])
	if duration <= 0 {
		frames := intParam(source.Params, "num_frames", 0)
		fps, _ := numberFromAny(source.Params["fps"])
		if frames > 0 && fps > 0 {
			duration = float64(frames) / fps
		}
	}
	if source.Kind == "recognition" {
		media, _ := source.Params["media"].(map[string]any)
		mediaType, _ := media["media_type"].(string)
		contentType, _ := media["content_type"].(string)
		if mediaType == "audio" || strings.HasPrefix(contentType, "audio/") {
			http.Error(w, "audio sources cannot be video-upscaled", http.StatusConflict)
			return
		}
		width, height = intParam(media, "width", 0), intParam(media, "height", 0)
		duration, _ = numberFromAny(media["duration"])
		fps, _ = numberFromAny(media["fps"])
	}
	var request struct {
		Scale           float64 `json:"scale"`
		Seed            int64   `json:"seed"`
		BatchSize       int     `json:"batch_size"`
		TemporalOverlap int     `json:"temporal_overlap"`
		StartTime       float64 `json:"start_time"`
		EndTime         float64 `json:"end_time"`
	}
	request.Scale, request.Seed, request.BatchSize, request.TemporalOverlap = 2, -1, 5, 1
	decoder := json.NewDecoder(io.LimitReader(r.Body, 1<<20))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&request); err != nil && !errors.Is(err, io.EOF) {
		http.Error(w, "invalid video upscale request: "+err.Error(), http.StatusBadRequest)
		return
	}
	if request.Scale <= 1 || request.Scale > 4 {
		http.Error(w, "scale must be greater than 1 and at most 4", http.StatusBadRequest)
		return
	}
	if request.BatchSize < 1 || (request.BatchSize-1)%4 != 0 || request.BatchSize > 21 {
		http.Error(w, "batch size must be one of 1, 5, 9, 13, 17 or 21", http.StatusBadRequest)
		return
	}
	if request.TemporalOverlap < 0 || request.TemporalOverlap > 4 {
		http.Error(w, "temporal overlap must be between 0 and 4", http.StatusBadRequest)
		return
	}
	if request.StartTime < 0 || request.EndTime < 0 || (request.EndTime > 0 && (request.EndTime <= request.StartTime || (duration > 0 && request.EndTime > duration+0.1))) {
		http.Error(w, "invalid video upscale time range", http.StatusBadRequest)
		return
	}
	if duration > 60 && request.EndTime <= 0 {
		http.Error(w, "videos longer than 60 seconds require a time range", http.StatusBadRequest)
		return
	}
	if request.EndTime > 0 && request.EndTime-request.StartTime > 60.1 {
		http.Error(w, "video upscale range must not exceed 60 seconds", http.StatusBadRequest)
		return
	}
	if width > 0 && height > 0 && float64(max(width, height))*request.Scale > 4096.5 {
		http.Error(w, "upscaled video must not exceed 4096 pixels on either edge", http.StatusBadRequest)
		return
	}
	id := newID()
	params := newVideoJobParams()
	params.Mode = "upscale"
	params.SourceJobID, params.SourceKind = source.ID, source.Kind
	params.UpscaleScale, params.BatchSize = request.Scale, request.BatchSize
	params.TemporalOverlap, params.Seed = request.TemporalOverlap, request.Seed
	params.Width = int(math.Round(float64(width) * request.Scale))
	params.Height = int(math.Round(float64(height) * request.Scale))
	params.SourceWidth, params.SourceHeight = width, height
	params.Duration, params.FPS = duration, fps
	params.Stage = "queued"
	now := time.Now()
	params.QueuedAt = now.Format(time.RFC3339Nano)
	if request.EndTime > 0 {
		params.SourceStartTime, params.SourceEndTime = request.StartTime, request.EndTime
		params.Duration = request.EndTime - request.StartTime
	}
	job := jobs.Job{ID: id, Kind: "video", Status: "queued", Prompt: source.Prompt, Params: params.toMap(), CreatedAt: now}
	if err := s.jobs.Save(job); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	s.wakeGenerationQueue()
	writeJSON(w, http.StatusAccepted, job)
}
