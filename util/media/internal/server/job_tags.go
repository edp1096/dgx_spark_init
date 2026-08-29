package server

import (
	"encoding/json"
	"errors"
	"mediaapp/internal/jobs"
	"net/http"
)

type updateJobTagsRequest struct {
	Tags []string `json:"tags"`
}

func (s *Server) listTags(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, s.jobs.Tags())
}

func (s *Server) updateJobTags(w http.ResponseWriter, r *http.Request) {
	var request updateJobTagsRequest
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 16<<10))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&request); err != nil {
		http.Error(w, "invalid tag request", http.StatusBadRequest)
		return
	}
	job, err := s.jobs.UpdateTags(r.PathValue("id"), request.Tags)
	switch {
	case err == nil:
		writeJSON(w, http.StatusOK, job)
	case errors.Is(err, jobs.ErrNotFound):
		http.NotFound(w, r)
	case errors.Is(err, jobs.ErrInvalidTags):
		http.Error(w, "태그는 항목당 24개, 이름당 32자까지이며 쉼표를 포함할 수 없습니다.", http.StatusBadRequest)
	default:
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}
