package server

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"mediaapp/internal/jobs"
)

func (s *Server) retryJob(w http.ResponseWriter, r *http.Request) {
	job, ok := s.jobs.Get(r.PathValue("id"))
	if !ok {
		http.NotFound(w, r)
		return
	}
	switch job.Kind {
	case "recognition":
		s.retrySubtitle(w, r)
	case "image":
		s.retryImage(w, job)
	case "video":
		s.retryVideo(w, job)
	case "speech":
		s.retrySpeech(w, job)
	default:
		http.Error(w, "this job type cannot be retried yet", http.StatusConflict)
	}
}

func (s *Server) retryVideo(w http.ResponseWriter, job jobs.Job) {
	if job.Status != "failed" && job.Status != "cancelled" {
		http.Error(w, "only failed or cancelled video jobs can be generated again", http.StatusConflict)
		return
	}
	if imageStringParam(job.Params, "mode", "") != "upscale" {
		if _, err := s.loadVideoExecution(job); err != nil {
			http.Error(w, err.Error(), http.StatusConflict)
			return
		}
	} else {
		sourceID := imageStringParam(job.Params, "source_job_id", "")
		source, ok := s.jobs.Get(sourceID)
		if !ok || (source.Kind == "video" && (source.Status != "completed" || source.OutputURL == "")) || (source.Kind == "recognition" && source.MediaAssetID == "") {
			http.Error(w, "saved upscale source is missing", http.StatusConflict)
			return
		}
	}
	if job.Params == nil {
		job.Params = map[string]any{}
	}
	job.Status = "queued"
	job.Error = ""
	job.OutputURL = ""
	delete(job.Params, "started_at")
	job.Params["stage"] = "queued"
	job.Params["queued_at"] = time.Now().Format(time.RFC3339Nano)
	job.Params["retried_at"] = time.Now().Format(time.RFC3339Nano)
	job.Params["retry_count"] = imageIntParam(job.Params, "retry_count", 0) + 1
	if err := s.jobs.Save(job); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	s.wakeGenerationQueue()
	writeJSON(w, http.StatusAccepted, job)
}

func (s *Server) savedVideoInput(id, role string) (string, error) {
	dir := filepath.Join(s.dataDir, "inputs", id, role)
	entries, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		return "", nil
	}
	if err != nil {
		return "", err
	}
	for _, entry := range entries {
		if entry.Type().IsRegular() {
			return filepath.Join(dir, entry.Name()), nil
		}
	}
	return "", nil
}

func (s *Server) retryImage(w http.ResponseWriter, job jobs.Job) {
	if job.Status != "failed" && job.Status != "cancelled" {
		http.Error(w, "only failed or cancelled image jobs can be generated again", http.StatusConflict)
		return
	}
	if operation := imageStringParam(job.Params, "operation", ""); operation != "" {
		http.Error(w, "failed post-processing jobs cannot be generated again", http.StatusConflict)
		return
	}
	mode := imageStringParam(job.Params, "mode", "create")
	if mode == "garment_extract" {
		paths, err := s.imageInputFiles(job.ID, "garment_source")
		if err != nil || len(paths) != 1 {
			http.Error(w, "saved garment source is missing", http.StatusConflict)
			return
		}
	} else if mode != "upscale" && mode != "detail_enhance" {
		if _, err := s.loadImageExecution(job); err != nil {
			http.Error(w, err.Error(), http.StatusConflict)
			return
		}
	} else {
		source, ok := s.jobs.Get(imageStringParam(job.Params, "source_job_id", ""))
		if !ok || source.Kind != "image" || source.Status != "completed" || source.OutputURL == "" {
			http.Error(w, "saved source image is missing", http.StatusConflict)
			return
		}
	}

	job.Status = "queued"
	job.Error = ""
	job.OutputURL = ""
	if job.Params == nil {
		job.Params = map[string]any{}
	}
	delete(job.Params, "started_at")
	job.Params["stage"] = "queued"
	job.Params["queued_at"] = time.Now().Format(time.RFC3339Nano)
	job.Params["retried_at"] = time.Now().Format(time.RFC3339Nano)
	job.Params["retry_count"] = imageIntParam(job.Params, "retry_count", 0) + 1
	if err := s.jobs.Save(job); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	s.wakeGenerationQueue()
	writeJSON(w, http.StatusAccepted, job)
}

func (s *Server) retrySpeech(w http.ResponseWriter, job jobs.Job) {
	if job.Status != "failed" && job.Status != "cancelled" {
		http.Error(w, "only failed or cancelled speech jobs can be generated again", http.StatusConflict)
		return
	}
	job.Status, job.Error, job.OutputURL = "queued", "", ""
	if job.Params == nil {
		job.Params = map[string]any{}
	}
	delete(job.Params, "started_at")
	job.Params["stage"] = "queued"
	job.Params["queued_at"] = time.Now().Format(time.RFC3339Nano)
	job.Params["retried_at"] = time.Now().Format(time.RFC3339Nano)
	job.Params["retry_count"] = imageIntParam(job.Params, "retry_count", 0) + 1
	if err := s.jobs.Save(job); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	s.wakeGenerationQueue()
	writeJSON(w, http.StatusAccepted, job)
}

func firstImagePath(paths []string) string {
	if len(paths) > 0 {
		return paths[0]
	}
	return ""
}

func imageStringParam(params map[string]any, key, fallback string) string {
	if value, ok := params[key].(string); ok && strings.TrimSpace(value) != "" {
		return value
	}
	return fallback
}

func imageIntParam(params map[string]any, key string, fallback int) int {
	switch value := params[key].(type) {
	case int:
		return value
	case int64:
		return int(value)
	case float64:
		return int(value)
	case json.Number:
		if number, err := value.Int64(); err == nil {
			return int(number)
		}
	}
	return fallback
}

func imageInt64Param(params map[string]any, key string, fallback int64) int64 {
	switch value := params[key].(type) {
	case int:
		return int64(value)
	case int64:
		return value
	case float64:
		return int64(value)
	case json.Number:
		if number, err := value.Int64(); err == nil {
			return number
		}
	}
	return fallback
}

func imageFloatParam(params map[string]any, key string, fallback float64) float64 {
	switch value := params[key].(type) {
	case float64:
		return value
	case float32:
		return float64(value)
	case int:
		return float64(value)
	case int64:
		return float64(value)
	case json.Number:
		if number, err := value.Float64(); err == nil {
			return number
		}
	}
	return fallback
}

func imageBoolParam(params map[string]any, key string, fallback bool) bool {
	if value, ok := params[key].(bool); ok {
		return value
	}
	return fallback
}

func decodeImageParam(params map[string]any, key string, target any) {
	data, err := json.Marshal(params[key])
	if err == nil {
		_ = json.Unmarshal(data, target)
	}
}
