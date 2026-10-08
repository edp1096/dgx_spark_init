package server

import (
	"mediaapp/internal/jobs"
	"net/http"
	"time"
)

func resetGenerationJobForRetry(job jobs.Job, now time.Time) jobs.Job {
	job.Status = "queued"
	job.Error = ""
	job.OutputURL = ""
	if job.Params == nil {
		job.Params = map[string]any{}
	}
	delete(job.Params, "started_at")
	job.Params["stage"] = "queued"
	job.Params["queued_at"] = now.Format(time.RFC3339Nano)
	job.Params["retried_at"] = now.Format(time.RFC3339Nano)
	job.Params["retry_count"] = intParam(job.Params, "retry_count", 0) + 1
	return job
}

func (s *Server) queueGenerationRetry(w http.ResponseWriter, job jobs.Job) {
	job = resetGenerationJobForRetry(job, time.Now())
	if err := s.jobs.Save(job); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	s.wakeGenerationQueue()
	writeJSON(w, http.StatusAccepted, job)
}
