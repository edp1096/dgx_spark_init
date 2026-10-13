package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
)

func (s *Server) documentTaskAPI(w http.ResponseWriter, r *http.Request) {
	session := r.URL.Query().Get("session_id")
	if session == "" {
		http.Error(w, "session_id required", 400)
		return
	}
	parts := strings.Split(strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/document-tasks"), "/"), "/")
	if len(parts) == 1 && parts[0] == "" {
		if r.Method != "POST" {
			methodNotAllowed(w)
			return
		}
		var a documentStart
		if e := json.NewDecoder(http.MaxBytesReader(w, r.Body, 20000)).Decode(&a); e != nil {
			http.Error(w, e.Error(), 400)
			return
		}
		a.Action = "start"
		a.Key = r.Header.Get("Idempotency-Key")
		run, e := s.documentAction(r.Context(), session, a)
		if e != nil {
			http.Error(w, e.Error(), 400)
			return
		}
		writeJSON(w, 202, s.documentResultView(run))
		return
	}
	id := parts[0]
	run, e := s.db.DocumentRun(session, id)
	if e != nil {
		http.Error(w, "document task not found", 404)
		return
	}
	if len(parts) == 1 && r.Method == "DELETE" {
		if _, e = s.documentAction(r.Context(), session, documentStart{Action: "cancel", RunID: id}); e != nil {
			http.Error(w, e.Error(), 409)
			return
		}
		if e = s.db.DeleteDocumentRun(session, id); e != nil {
			http.Error(w, e.Error(), 500)
			return
		}
		w.WriteHeader(204)
		return
	}
	if len(parts) == 1 && r.Method == "GET" {
		view := s.documentResultView(run)
		units, e := s.db.DocumentUnits(session, id)
		if e != nil {
			http.Error(w, e.Error(), 500)
			return
		}
		offset, _ := strconv.Atoi(r.URL.Query().Get("offset"))
		limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
		if offset < 0 {
			offset = 0
		}
		if limit < 1 || limit > 50 {
			limit = 10
		}
		offset = min(offset, len(units))
		units = units[offset:min(len(units), offset+limit)]
		for i := range units {
			units[i].Inputs = nil
		}
		view["units"] = units
		writeJSON(w, 200, view)
		return
	}
	draftDownload := len(parts) == 3 && parts[1] == "draft" && strings.HasPrefix(parts[2], "output.")
	if (len(parts) == 2 && strings.HasPrefix(parts[1], "output.") || draftDownload) && r.Method == "GET" {
		name := parts[len(parts)-1]
		published := run.Document.Pipeline == "" && !draftDownload
		manifest := run.Document.Publication
		if draftDownload {
			manifest = run.Document.DraftPublication
			if manifest == nil {
				manifest = run.Document.Publication
			}
		}
		if manifest != nil {
			for _, file := range manifest.Files {
				published = published || file.Name == name
			}
		}
		jobs, err := s.db.DocumentConversions(session, id)
		if err != nil {
			http.Error(w, "document conversion state unavailable", 500)
			return
		}
		for _, job := range jobs {
			if "output."+job.Format == name && job.Status == "completed" && job.Current {
				published = true
			}
		}
		if run.Current < 2 || !published {
			http.Error(w, "output not available yet", 404)
			return
		}
		data, mime, e := s.db.ConvertedDocumentOutput(session, id, name)
		if e != nil {
			http.Error(w, "output not available yet", 404)
			return
		}
		w.Header().Set("Cache-Control", "no-store")
		w.Header().Set("Content-Type", mime)
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=%q", name))
		w.Write(data)
		return
	}
	if len(parts) == 2 && parts[1] == "reviews" && r.Method == "GET" {
		reviews, err := s.db.DocumentReviews(session, id)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		writeJSON(w, 200, reviews)
		return
	}
	if len(parts) == 2 && r.Method == "POST" {
		var a documentStart
		if r.ContentLength != 0 {
			if e := json.NewDecoder(http.MaxBytesReader(w, r.Body, 20000)).Decode(&a); e != nil {
				http.Error(w, e.Error(), 400)
				return
			}
		}
		a.Action = parts[1]
		a.RunID = id
		next, e := s.documentAction(r.Context(), session, a)
		if e != nil {
			http.Error(w, e.Error(), 409)
			return
		}
		code := http.StatusOK
		if a.Action == "reexport" {
			code = http.StatusAccepted
		}
		writeJSON(w, code, s.documentResultView(next))
		return
	}
	methodNotAllowed(w)
}
func (s *Server) sourceAPI(w http.ResponseWriter, r *http.Request) {
	session := r.URL.Query().Get("session_id")
	id := strings.TrimPrefix(r.URL.Path, "/api/sources/")
	if session == "" || id == "" {
		http.Error(w, "session_id and source required", 400)
		return
	}
	switch r.Method {
	case "GET":
		b, e := s.db.Source(session, id, false)
		if e != nil {
			http.Error(w, "source not found", 404)
			return
		}
		writeJSON(w, 200, b)
	case "DELETE":
		if e := s.db.DeleteSource(session, id); e != nil {
			http.Error(w, e.Error(), 409)
			return
		}
		w.WriteHeader(204)
	default:
		methodNotAllowed(w)
	}
}
