package server

import (
	"errors"
	"log"
	"mediaapp/internal/jobs"
	"net/http"
	"strings"
)

func (s *Server) fail(j jobs.Job, err error) {
	if current, ok := s.jobs.Get(j.ID); ok && current.Status == "cancelled" {
		return
	}
	log.Printf("job %s failed: %v", j.ID, err)
	transitionJobFailed(&j, err.Error())
	_ = s.jobs.Save(j)
}

func (s *Server) jobCancelled(id string) bool {
	j, ok := s.jobs.Get(id)
	return ok && j.Status == "cancelled"
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
	if j.Kind == "recognition" || j.MediaAssetID != "" {
		if err := s.deleteMediaJobArtifacts(id); err != nil {
			http.Error(w, err.Error(), http.StatusBadGateway)
			return
		}
	}
	if err := s.deleteVideoPreview(id); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
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
	s.generationStateMu.Lock()
	j, ok := s.jobs.Get(id)
	if !ok {
		s.generationStateMu.Unlock()
		http.NotFound(w, r)
		return
	}
	if j.Status != "queued" && j.Status != "running" {
		s.generationStateMu.Unlock()
		http.Error(w, "job is not active", http.StatusConflict)
		return
	}
	if j.Kind != "recognition" && !isGenerationKind(j.Kind) {
		s.generationStateMu.Unlock()
		http.Error(w, "job cannot be cancelled", http.StatusConflict)
		return
	}
	wasRunning := j.Status == "running"
	transitionJobCancelled(&j)
	if err := s.jobs.Save(j); err != nil {
		s.generationStateMu.Unlock()
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	s.generationStateMu.Unlock()

	if j.Kind == "recognition" {
		s.cancelMediaPreparation(id)
	} else {
		if wasRunning {
			s.generationCancelMu.Lock()
			cancel := s.generationCancels[id]
			s.generationCancelMu.Unlock()
			if cancel != nil {
				cancel()
			}
			if j.Kind == "video" && decodeVideoJobParams(j.Params).Mode != "upscale" {
				s.engineDrainMu.Lock()
				s.engineDraining["video"] = true
				s.engineDrainMu.Unlock()
			}
			go s.interruptGenerationEngine(j)
		}
		s.wakeGenerationQueue()
	}
	writeJSON(w, http.StatusOK, j)
}

// interruptGenerationEngine asks engines that support cooperative interruption
// to stop GPU work as well as cancelling this server's HTTP request. Engines
// without the endpoint simply return 404 and retain their own safety behavior.
func (s *Server) interruptGenerationEngine(job jobs.Job) {
	cfg := s.config()
	endpoint := ""
	switch job.Kind {
	case "image":
		mode := stringParam(job.Params, "mode", cfg.Image.DefaultMode)
		if mode == "upscale" {
			endpoint = cfg.Engines["upscale"].Endpoint
		} else if mode == "garment_extract" {
			endpoint = cfg.Engines["garment"].Endpoint
		} else if mode == "face_swap" {
			endpoint = cfg.Engines["faceswap"].Endpoint
		} else if backend, ok := cfg.Image.Backends[mode]; ok {
			endpoint = backend.Endpoint
		} else if backend, ok := cfg.Image.Backends[cfg.Image.DefaultMode]; ok {
			endpoint = backend.Endpoint
		}
	case "video":
		if decodeVideoJobParams(job.Params).Mode == "upscale" {
			endpoint = cfg.Engines["upscale"].Endpoint
		} else {
			endpoint = cfg.Engines["video"].Endpoint
		}
	case "speech":
		endpoint = cfg.Engines["speech"].Endpoint
	}
	endpoint = strings.TrimRight(endpoint, "/")
	if endpoint == "" {
		return
	}
	request, err := http.NewRequest(http.MethodPost, endpoint+"/v1/cancel", nil)
	if err != nil {
		return
	}
	response, err := s.health.Do(request)
	if err == nil {
		_ = response.Body.Close()
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
		if j.Kind == "recognition" || j.MediaAssetID != "" {
			if err := s.deleteMediaJobArtifacts(j.ID); err != nil {
				http.Error(w, err.Error(), http.StatusBadGateway)
				return
			}
		}
		if err := s.deleteVideoPreview(j.ID); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
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
