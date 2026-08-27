package server

import (
	"mediaapp/internal/jobs"
	"net/http"
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
	params := decodeVideoJobParams(job.Params)
	if params.Mode != "upscale" {
		if _, err := s.loadVideoExecution(job); err != nil {
			http.Error(w, err.Error(), http.StatusConflict)
			return
		}
	} else {
		source, ok := s.jobs.Get(params.SourceJobID)
		if !ok || (source.Kind == "video" && (source.Status != "completed" || source.OutputURL == "")) || (source.Kind == "recognition" && source.MediaAssetID == "") {
			http.Error(w, "saved upscale source is missing", http.StatusConflict)
			return
		}
	}
	if job.Params == nil {
		job.Params = map[string]any{}
	}
	s.queueGenerationRetry(w, job)
}

func (s *Server) retryImage(w http.ResponseWriter, job jobs.Job) {
	if job.Status != "failed" && job.Status != "cancelled" {
		http.Error(w, "only failed or cancelled image jobs can be generated again", http.StatusConflict)
		return
	}
	if operation := stringParam(job.Params, "operation", ""); operation != "" {
		http.Error(w, "failed post-processing jobs cannot be generated again", http.StatusConflict)
		return
	}
	mode := decodeImageJobParams(job.Params).Mode
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
		source, ok := s.jobs.Get(stringParam(job.Params, "source_job_id", ""))
		if !ok || source.Kind != "image" || source.Status != "completed" || source.OutputURL == "" {
			http.Error(w, "saved source image is missing", http.StatusConflict)
			return
		}
	}

	s.queueGenerationRetry(w, job)
}

func (s *Server) retrySpeech(w http.ResponseWriter, job jobs.Job) {
	if job.Status != "failed" && job.Status != "cancelled" {
		http.Error(w, "only failed or cancelled speech jobs can be generated again", http.StatusConflict)
		return
	}
	s.queueGenerationRetry(w, job)
}
