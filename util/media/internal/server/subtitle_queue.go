package server

import (
	"errors"
	"fmt"
	"mediaapp/internal/jobs"
	"net/http"
	"path/filepath"
	"strings"
	"time"
)

var errJobCancelled = errors.New("job cancelled")

// wakeSubtitleQueue starts one durable FIFO worker and notifies it that work
// may be available. The worker owns the complete media-to-subtitle pipeline,
// so downloads, ASR, and translation never overlap across subtitle jobs.
func (s *Server) wakeSubtitleQueue() {
	s.subtitleQueueOnce.Do(func() { go s.subtitleQueueLoop() })
	select {
	case s.subtitleQueueWake <- struct{}{}:
	default:
	}
}

func (s *Server) subtitleQueueLoop() {
	for {
		job, ok := s.nextQueuedSubtitle()
		if !ok {
			<-s.subtitleQueueWake
			continue
		}
		s.executeQueuedSubtitle(job)
	}
}

func (s *Server) nextQueuedSubtitle() (jobs.Job, bool) {
	var next jobs.Job
	found := false
	for _, job := range s.jobs.List() {
		if job.Kind != "recognition" || job.Status != "queued" {
			continue
		}
		if !found || subtitleQueueTime(job).Before(subtitleQueueTime(next)) ||
			(subtitleQueueTime(job).Equal(subtitleQueueTime(next)) && job.ID < next.ID) {
			next = job
			found = true
		}
	}
	return next, found
}

func subtitleQueueTime(job jobs.Job) time.Time {
	if value, ok := job.Params["queued_at"].(string); ok {
		if parsed, err := time.Parse(time.RFC3339Nano, value); err == nil {
			return parsed
		}
	}
	return job.CreatedAt
}

func (s *Server) executeQueuedSubtitle(job jobs.Job) {
	current, ok := s.jobs.Get(job.ID)
	if !ok || current.Kind != "recognition" || current.Status != "queued" {
		return
	}
	now := time.Now()
	transitionJobRunning(&current, "media", now)
	current.Params["stage_started_at"] = now.Format(time.RFC3339Nano)
	if err := s.jobs.Save(current); err != nil {
		return
	}
	inputDir := filepath.Join(s.dataDir, "inputs", current.ID)
	params := decodeSubtitleJobParams(current.Params, s.config().Recognition)
	sourceURL := ""
	inputPath := ""
	if params.Source == "url" {
		sourceURL = current.Prompt
	} else {
		matches, _ := filepath.Glob(filepath.Join(inputDir, "source.*"))
		if len(matches) == 0 {
			s.fail(current, fmt.Errorf("saved source media is missing"))
			return
		}
		inputPath = matches[0]
	}
	s.runSubtitle(
		current, inputDir, inputPath, sourceURL,
		params.Language, params.Context, params.OutputFormats,
		params.TranslationMode, params.TargetLanguage, params.MediaPart, params.MediaSource,
	)
}

// CancelActiveMediaPreparations stops remote download/FFmpeg processes before
// interrupted subtitle jobs are resumed. Without this handshake, restarting
// the app can submit the same durable request ID while the previous Media API
// handler is still writing to its partial file.
func (s *Server) CancelActiveMediaPreparations() int {
	cancelled := 0
	for _, job := range s.jobs.List() {
		if job.Kind != "recognition" || job.Status != "running" {
			continue
		}
		job.Status = "queued"
		if job.Params == nil {
			job.Params = map[string]any{}
		}
		if _, ok := job.Params["queued_at"]; !ok {
			job.Params["queued_at"] = job.CreatedAt.Format(time.RFC3339Nano)
		}
		job.Params["stage"] = "queued"
		_ = s.jobs.Save(job)
		if s.cancelMediaPreparation(job.ID) {
			cancelled++
		}
	}
	return cancelled
}

func (s *Server) cancelMediaPreparation(requestID string) bool {
	base := strings.TrimRight(s.config().Engines["media"].Endpoint, "/")
	if base == "" {
		return false
	}
	request, err := http.NewRequest(http.MethodDelete, base+"/v1/media/prepare/"+requestID, nil)
	if err != nil {
		return false
	}
	response, err := (&http.Client{Timeout: 10 * time.Second}).Do(request)
	if err != nil {
		return false
	}
	defer response.Body.Close()
	return response.StatusCode == http.StatusAccepted
}
