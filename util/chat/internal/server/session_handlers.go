package server

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"strings"
	"unicode/utf8"
)

func (s *Server) sessions(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		items, err := s.db.Sessions()
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		writeJSON(w, 200, items)
	case http.MethodPost:
		var req struct {
			Title     string `json:"title"`
			Model     string `json:"model"`
			Reasoning string `json:"reasoning_effort"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		if strings.TrimSpace(req.Title) == "" {
			req.Title = "새 대화"
		}
		cfg, _ := s.snapshot()
		if req.Model == "" {
			req.Model = cfg.Model.DefaultModel
		}
		if req.Reasoning == "" {
			req.Reasoning = cfg.Model.ReasoningEffort
		}
		item, err := s.db.CreateSession(newID(), req.Title, req.Model, req.Reasoning)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		writeJSON(w, 201, item)
	default:
		methodNotAllowed(w)
	}
}

func (s *Server) groups(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		items, err := s.db.Groups()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, http.StatusOK, items)
	case http.MethodPost:
		var req struct {
			Name string `json:"name"`
		}
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 16<<10)).Decode(&req); err != nil {
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		req.Name = strings.TrimSpace(req.Name)
		if req.Name == "" || utf8.RuneCountInString(req.Name) > 60 {
			http.Error(w, "group name must be between 1 and 60 characters", http.StatusBadRequest)
			return
		}
		item, err := s.db.CreateGroup(newID(), req.Name)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, http.StatusCreated, item)
	default:
		methodNotAllowed(w)
	}
}

func (s *Server) group(w http.ResponseWriter, r *http.Request) {
	rest := strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/groups/"), "/")
	parts := strings.Split(rest, "/")
	if len(parts) == 0 || parts[0] == "" {
		http.NotFound(w, r)
		return
	}
	id := parts[0]
	var err error
	switch {
	case len(parts) == 1 && r.Method == http.MethodPatch:
		var req struct {
			Name string `json:"name"`
		}
		if decodeErr := json.NewDecoder(http.MaxBytesReader(w, r.Body, 16<<10)).Decode(&req); decodeErr != nil {
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		req.Name = strings.TrimSpace(req.Name)
		if req.Name == "" || utf8.RuneCountInString(req.Name) > 60 {
			http.Error(w, "group name must be between 1 and 60 characters", http.StatusBadRequest)
			return
		}
		err = s.db.RenameGroup(id, req.Name)
		if err == nil {
			writeJSON(w, http.StatusOK, map[string]string{"name": req.Name})
			return
		}
	case len(parts) == 2 && parts[1] == "move" && r.Method == http.MethodPost:
		var req struct {
			Direction string `json:"direction"`
		}
		if decodeErr := json.NewDecoder(http.MaxBytesReader(w, r.Body, 16<<10)).Decode(&req); decodeErr != nil || (req.Direction != "up" && req.Direction != "down") {
			http.Error(w, "direction must be up or down", http.StatusBadRequest)
			return
		}
		err = s.db.MoveGroup(id, req.Direction)
		if err == nil {
			w.WriteHeader(http.StatusNoContent)
			return
		}
	case len(parts) == 1 && r.Method == http.MethodDelete:
		err = s.db.DeleteGroup(id)
		if err == nil {
			w.WriteHeader(http.StatusNoContent)
			return
		}
	default:
		http.NotFound(w, r)
		return
	}
	if err == sql.ErrNoRows {
		http.NotFound(w, r)
	} else {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func (s *Server) session(w http.ResponseWriter, r *http.Request) {
	rest := strings.TrimPrefix(r.URL.Path, "/api/sessions/")
	parts := strings.Split(strings.Trim(rest, "/"), "/")
	if len(parts) == 0 || parts[0] == "" {
		http.NotFound(w, r)
		return
	}
	id := parts[0]
	if len(parts) == 2 && parts[1] == "messages" && r.Method == http.MethodGet {
		items, err := s.db.Messages(id)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		writeJSON(w, 200, items)
		return
	}
	if len(parts) == 2 && parts[1] == "group" && r.Method == http.MethodPut {
		var req struct {
			GroupID string `json:"group_id"`
		}
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 16<<10)).Decode(&req); err != nil {
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		if err := s.db.SetSessionGroup(id, strings.TrimSpace(req.GroupID)); err != nil {
			if err == sql.ErrNoRows {
				http.NotFound(w, r)
			} else {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		}
		w.WriteHeader(http.StatusNoContent)
		return
	}
	if len(parts) == 1 && r.Method == http.MethodDelete {
		if err := s.db.DeleteSession(id); err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		w.WriteHeader(http.StatusNoContent)
		return
	}
	if len(parts) == 1 && r.Method == http.MethodPatch {
		var req struct {
			Title string `json:"title"`
		}
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 16<<10)).Decode(&req); err != nil {
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		req.Title = strings.TrimSpace(req.Title)
		if req.Title == "" || utf8.RuneCountInString(req.Title) > 120 {
			http.Error(w, "title must be between 1 and 120 characters", http.StatusBadRequest)
			return
		}
		if err := s.db.RenameSession(id, req.Title); err != nil {
			if err == sql.ErrNoRows {
				http.NotFound(w, r)
			} else {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		}
		writeJSON(w, http.StatusOK, map[string]string{"title": req.Title})
		return
	}
	http.NotFound(w, r)
}
