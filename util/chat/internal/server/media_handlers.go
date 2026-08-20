package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"path/filepath"
	"strings"
	"time"

	"sparktalk/internal/media"
)

func (s *Server) uploadSource(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	var input struct {
		URL string `json:"url"`
	}
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&input); err != nil || strings.TrimSpace(input.URL) == "" {
		http.Error(w, "url is required", http.StatusBadRequest)
		return
	}
	cfg, _ := s.snapshot()
	endpoint := strings.TrimRight(cfg.ASR.FFmpegEndpoint, "/")
	if endpoint == "" {
		http.Error(w, "SparkTalk Media API endpoint is not configured", http.StatusServiceUnavailable)
		return
	}
	payload, _ := json.Marshal(map[string]any{
		"url":             input.URL,
		"max_download_mb": media.MaxAttachmentBytes >> 20,
		"max_height":      720,
	})
	request, err := http.NewRequestWithContext(r.Context(), http.MethodPost, endpoint+"/v1/source/download", bytes.NewReader(payload))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	request.Header.Set("Content-Type", "application/json")
	timeout, _ := time.ParseDuration(cfg.ASR.Timeout)
	if timeout <= 0 {
		timeout = 30 * time.Minute
	}
	response, err := (&http.Client{Timeout: timeout}).Do(request)
	if err != nil {
		http.Error(w, "SparkTalk Media API: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, 64<<10))
		http.Error(w, fmt.Sprintf("SparkTalk Media API HTTP %d: %s", response.StatusCode, strings.TrimSpace(string(detail))), http.StatusBadGateway)
		return
	}
	name := sourceResponseName(response)
	item, err := s.media.SaveReader(response.Body, name, response.Header.Get("Content-Type"), media.MaxAttachmentBytes)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	writeJSON(w, http.StatusCreated, item)
}

func sourceResponseName(response *http.Response) string {
	name := "remote-media"
	if disposition, params, err := mime.ParseMediaType(response.Header.Get("Content-Disposition")); err == nil && disposition == "attachment" && params["filename"] != "" {
		name = filepath.Base(params["filename"])
	}
	extension := filepath.Ext(name)
	if encoded := response.Header.Get("X-Media-Title"); encoded != "" {
		if title, err := url.QueryUnescape(encoded); err == nil && strings.TrimSpace(title) != "" {
			name = strings.TrimSpace(title) + extension
		}
	}
	return name
}

func (s *Server) mediaUsage(w http.ResponseWriter, r *http.Request) {
	referenced, err := s.db.ReferencedAttachmentIDs()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	keep := make(map[string]struct{})
	cfg, _ := s.snapshot()
	for _, avatar := range []string{cfg.Appearance.AssistantAvatar, cfg.Appearance.UserAvatar} {
		if id := strings.TrimPrefix(avatar, "/api/images/"); id != avatar && id != "" {
			keep[id] = struct{}{}
		}
	}
	if r.Method == http.MethodDelete {
		var req struct {
			KeepIDs []string `json:"keep_ids"`
		}
		if r.Body != nil {
			_ = json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10)).Decode(&req)
		}
		for _, id := range req.KeepIDs {
			keep[id] = struct{}{}
		}
		removed, err := s.media.Cleanup(referenced, keep)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		usage, err := s.media.Usage(referenced, keep)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"removed": removed, "usage": usage})
		return
	}
	if r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	usage, err := s.media.Usage(referenced, keep)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusOK, usage)
}

func (s *Server) uploadImage(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, 16<<20)
	if err := r.ParseMultipartForm(16 << 20); err != nil {
		http.Error(w, "invalid image upload or image is too large", http.StatusBadRequest)
		return
	}
	defer r.MultipartForm.RemoveAll()
	file, header, err := r.FormFile("image")
	if err != nil {
		http.Error(w, "image is required", http.StatusBadRequest)
		return
	}
	_ = file.Close()
	item, err := s.media.SaveImage(header)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	writeJSON(w, http.StatusCreated, item)
}

func (s *Server) image(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	s.media.Serve(w, r, strings.TrimPrefix(r.URL.Path, "/api/images/"), "image", "")
}

func (s *Server) uploadFile(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, media.MaxAttachmentBytes+(1<<20))
	if err := r.ParseMultipartForm(8 << 20); err != nil {
		http.Error(w, "invalid media upload or file is too large", http.StatusBadRequest)
		return
	}
	defer r.MultipartForm.RemoveAll()
	file, header, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "media file is required", http.StatusBadRequest)
		return
	}
	_ = file.Close()
	item, err := s.media.SaveAttachment(header)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	writeJSON(w, http.StatusCreated, item)
}

func (s *Server) file(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet && r.Method != http.MethodHead {
		methodNotAllowed(w)
		return
	}
	rest := strings.TrimPrefix(r.URL.Path, "/api/files/")
	parts := strings.SplitN(rest, "/", 2)
	if len(parts) != 2 {
		http.NotFound(w, r)
		return
	}
	name, err := url.PathUnescape(parts[1])
	if err != nil {
		http.NotFound(w, r)
		return
	}
	s.media.Serve(w, r, parts[0], name, r.URL.Query().Get("type"))
}
