package server

import (
	"mediaapp/internal/jobs"
	"time"
)

func ensureJobParams(job *jobs.Job) {
	if job.Params == nil {
		job.Params = map[string]any{}
	}
}

func transitionJobRunning(job *jobs.Job, stage string, now time.Time) {
	ensureJobParams(job)
	job.Status = "running"
	job.Error = ""
	if stage != "" {
		job.Params["stage"] = stage
	}
	job.Params["started_at"] = now.Format(time.RFC3339Nano)
}

func transitionJobFailed(job *jobs.Job, message string) {
	ensureJobParams(job)
	job.Status = "failed"
	job.Error = message
}

func transitionJobCancelled(job *jobs.Job) {
	ensureJobParams(job)
	job.Status = "cancelled"
	job.Error = ""
	job.Params["stage"] = "cancelled"
	delete(job.Params, "media_eta_seconds")
}

func transitionJobCompleted(job *jobs.Job, outputURL string) {
	ensureJobParams(job)
	job.Status = "completed"
	job.Error = ""
	job.OutputURL = outputURL
}

// completeGenerationJob atomically observes cancellation before publishing a
// generated file. cleanup removes an output that lost the cancellation race.
func (s *Server) completeGenerationJob(job *jobs.Job, outputURL string, cleanup func()) (bool, error) {
	s.generationStateMu.Lock()
	defer s.generationStateMu.Unlock()
	if current, ok := s.jobs.Get(job.ID); ok && current.Status == "cancelled" {
		if cleanup != nil {
			cleanup()
		}
		return false, nil
	}
	transitionJobCompleted(job, outputURL)
	return true, s.jobs.Save(*job)
}
