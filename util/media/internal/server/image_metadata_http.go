package server

import (
	"net/http"
	"os"
	"path/filepath"
)

func (s *Server) imageJobEXIF(w http.ResponseWriter, r *http.Request) {
	j, ok := s.jobs.Get(r.PathValue("id"))
	if !ok || j.Kind != "image" || j.OutputURL == "" {
		http.NotFound(w, r)
		return
	}
	data, err := os.ReadFile(s.jobs.OutputPath(filepath.Base(j.OutputURL)))
	if err != nil {
		http.Error(w, "image file is no longer available", http.StatusNotFound)
		return
	}
	metadata, embedded := extractImageEXIF(data)
	writeJSON(w, http.StatusOK, map[string]any{"embedded": embedded, "metadata": metadata})
}
