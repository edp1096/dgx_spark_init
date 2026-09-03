package server

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"unicode/utf8"

	"sparktalk/internal/db"
)

type memoryRequest struct {
	Kind            string `json:"kind"`
	Priority        string `json:"priority"`
	Title           string `json:"title"`
	Content         string `json:"content"`
	Enabled         *bool  `json:"enabled,omitempty"`
	SourceSessionID string `json:"source_session_id,omitempty"`
	SourceMessageID int64  `json:"source_message_id,omitempty"`
}

func (s *Server) memories(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		items, err := s.db.Memories()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, http.StatusOK, items)
	case http.MethodPost:
		var req memoryRequest
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10)).Decode(&req); err != nil {
			http.Error(w, "invalid memory", http.StatusBadRequest)
			return
		}
		if message := normalizeMemoryRequest(&req); message != "" {
			http.Error(w, message, http.StatusBadRequest)
			return
		}
		item, err := s.db.AddMemory(req.Kind, req.Priority, req.Title, req.Content, req.SourceSessionID, req.SourceMessageID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, http.StatusCreated, item)
	default:
		methodNotAllowed(w)
	}
}

func (s *Server) memory(w http.ResponseWriter, r *http.Request) {
	rawID := strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/memories/"), "/")
	id, err := strconv.ParseInt(rawID, 10, 64)
	if err != nil || id < 1 {
		http.Error(w, "invalid memory id", http.StatusBadRequest)
		return
	}
	switch r.Method {
	case http.MethodPut:
		var req memoryRequest
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10)).Decode(&req); err != nil {
			http.Error(w, "invalid memory", http.StatusBadRequest)
			return
		}
		if message := normalizeMemoryRequest(&req); message != "" {
			http.Error(w, message, http.StatusBadRequest)
			return
		}
		enabled := true
		if req.Enabled != nil {
			enabled = *req.Enabled
		}
		item, err := s.db.UpdateMemory(id, req.Kind, req.Priority, req.Title, req.Content, enabled)
		if err != nil {
			writeMemoryError(w, err)
			return
		}
		writeJSON(w, http.StatusOK, item)
	case http.MethodDelete:
		if err := s.db.DeleteMemory(id); err != nil {
			writeMemoryError(w, err)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	default:
		methodNotAllowed(w)
	}
}

func normalizeMemoryRequest(req *memoryRequest) string {
	req.Kind = strings.ToLower(strings.TrimSpace(req.Kind))
	req.Priority = strings.ToLower(strings.TrimSpace(req.Priority))
	if req.Priority == "" {
		req.Priority = "preferred"
	}
	req.Title = strings.TrimSpace(req.Title)
	req.Content = strings.TrimSpace(req.Content)
	req.SourceSessionID = strings.TrimSpace(req.SourceSessionID)
	if req.Kind != "user" && req.Kind != "memory" {
		return "memory kind must be user or memory"
	}
	if req.Priority != "reference" && req.Priority != "preferred" {
		return "memory priority must be reference or preferred"
	}
	if req.Content == "" || utf8.RuneCountInString(req.Content) > 8000 {
		return "memory content must be between 1 and 8000 characters"
	}
	if utf8.RuneCountInString(req.Title) > 120 {
		return "memory title supports at most 120 characters"
	}
	return ""
}

func writeMemoryError(w http.ResponseWriter, err error) {
	if db.IsMemoryNotFound(err) {
		http.Error(w, "memory not found", http.StatusNotFound)
		return
	}
	http.Error(w, err.Error(), http.StatusInternalServerError)
}
