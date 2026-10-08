package server

import (
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

func validAssetID(id string) bool {
	if len(id) != 32 {
		return false
	}
	for _, char := range id {
		if !strings.ContainsRune("0123456789abcdef", char) {
			return false
		}
	}
	return true
}

func (s *Server) proxyMediaAsset(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if !validAssetID(id) {
		http.NotFound(w, r)
		return
	}
	target := strings.TrimRight(s.config().Engines["media"].Endpoint, "/") + "/v1/media/assets/" + id
	request, err := http.NewRequestWithContext(r.Context(), r.Method, target, nil)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	for _, header := range []string{"Range", "If-Range", "If-Modified-Since", "If-None-Match"} {
		if value := r.Header.Get(header); value != "" {
			request.Header.Set(header, value)
		}
	}
	response, err := http.DefaultTransport.RoundTrip(request)
	if err != nil {
		http.Error(w, "media stream: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer response.Body.Close()
	for _, header := range []string{"Accept-Ranges", "Content-Disposition", "Content-Length", "Content-Range", "Content-Type", "ETag", "Last-Modified"} {
		if values := response.Header.Values(header); len(values) > 0 {
			w.Header()[header] = append([]string(nil), values...)
		}
	}
	w.WriteHeader(response.StatusCode)
	if r.Method != http.MethodHead {
		_, _ = io.Copy(w, response.Body)
	}
}

func (s *Server) mediaJobFrame(w http.ResponseWriter, r *http.Request) {
	job, ok := s.jobs.Get(r.PathValue("id"))
	if !ok {
		http.NotFound(w, r)
		return
	}
	timeSeconds, err := strconv.ParseFloat(valueOr(r.URL.Query().Get("time"), "0"), 64)
	if err != nil || timeSeconds < 0 {
		http.Error(w, "time must be a non-negative number", http.StatusBadRequest)
		return
	}
	endpoint := strings.TrimRight(s.config().Engines["media"].Endpoint, "/")
	if job.Kind == "recognition" {
		if job.MediaAssetID == "" {
			http.Error(w, "saved source video is no longer available", http.StatusConflict)
			return
		}
		if media, ok := job.Params["media"].(map[string]any); ok {
			mediaType, _ := media["media_type"].(string)
			contentType, _ := media["content_type"].(string)
			if mediaType == "audio" || strings.HasPrefix(contentType, "audio/") {
				http.Error(w, "audio sources do not contain video frames", http.StatusConflict)
				return
			}
		}
		target := fmt.Sprintf("%s/v1/media/assets/%s/frame?time_seconds=%s", endpoint, job.MediaAssetID, url.QueryEscape(strconv.FormatFloat(timeSeconds, 'f', 6, 64)))
		s.proxyFrameResponse(w, r, target, "")
		return
	}
	if job.Kind != "video" || job.Status != "completed" || job.OutputURL == "" {
		http.Error(w, "only a completed video or transcription source can provide frames", http.StatusConflict)
		return
	}
	source := s.jobs.OutputPath(filepath.Base(job.OutputURL))
	if _, err := os.Stat(source); err != nil {
		http.Error(w, "source video is no longer available", http.StatusNotFound)
		return
	}
	s.proxyFrameResponse(w, r, endpoint+"/v1/media/frame", source)
}

func (s *Server) proxyFrameResponse(w http.ResponseWriter, r *http.Request, target, source string) {
	var request *http.Request
	var err error
	if source == "" {
		request, err = http.NewRequestWithContext(r.Context(), http.MethodGet, target, nil)
	} else {
		reader, writer := io.Pipe()
		multipartWriter := multipart.NewWriter(writer)
		go func() {
			writeErr := multipartWriter.WriteField("time_seconds", valueOr(r.URL.Query().Get("time"), "0"))
			if writeErr == nil {
				var file *os.File
				file, writeErr = os.Open(source)
				if writeErr == nil {
					var part io.Writer
					part, writeErr = multipartWriter.CreateFormFile("video", filepath.Base(source))
					if writeErr == nil {
						_, writeErr = io.Copy(part, file)
					}
					_ = file.Close()
				}
			}
			if closeErr := multipartWriter.Close(); writeErr == nil {
				writeErr = closeErr
			}
			_ = writer.CloseWithError(writeErr)
		}()
		request, err = http.NewRequestWithContext(r.Context(), http.MethodPost, target, reader)
		if err == nil {
			request.Header.Set("Content-Type", multipartWriter.FormDataContentType())
		}
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	response, err := s.client.Do(request)
	if err != nil {
		http.Error(w, "frame extraction service unavailable: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer response.Body.Close()
	if response.StatusCode/100 != 2 {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 1<<20))
		http.Error(w, strings.TrimSpace(string(body)), response.StatusCode)
		return
	}
	w.Header().Set("Content-Type", valueOr(response.Header.Get("Content-Type"), "image/jpeg"))
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Frame-Time", response.Header.Get("X-Frame-Time"))
	w.Header().Set("Content-Disposition", fmt.Sprintf("inline; filename=frame-%0.3fs.jpg", timeSecondsFromQuery(r)))
	_, _ = io.Copy(w, response.Body)
}

func timeSecondsFromQuery(r *http.Request) float64 {
	value, _ := strconv.ParseFloat(valueOr(r.URL.Query().Get("time"), "0"), 64)
	return value
}

func (s *Server) deleteMediaAsset(id string) error {
	if id == "" {
		return nil
	}
	if !validAssetID(id) {
		return fmt.Errorf("invalid media asset id")
	}
	target := strings.TrimRight(s.config().Engines["media"].Endpoint, "/") + "/v1/media/assets/" + id
	request, err := http.NewRequest(http.MethodDelete, target, nil)
	if err != nil {
		return err
	}
	response, err := s.health.Do(request)
	if err != nil {
		return fmt.Errorf("delete media asset: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusNoContent && response.StatusCode != http.StatusNotFound {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 1<<20))
		return fmt.Errorf("delete media asset: engine returned %d: %s", response.StatusCode, strings.TrimSpace(string(body)))
	}
	return nil
}

func (s *Server) deleteMediaJobArtifacts(id string) error {
	if id == "" || filepath.Base(id) != id {
		return fmt.Errorf("invalid media job id")
	}
	endpoint := strings.TrimRight(s.config().Engines["media"].Endpoint, "/")
	if endpoint == "" {
		return nil
	}
	target := endpoint + "/v1/media/jobs/" + url.PathEscape(id)
	request, err := http.NewRequest(http.MethodDelete, target, nil)
	if err != nil {
		return err
	}
	response, err := s.client.Do(request)
	if err != nil {
		return fmt.Errorf("delete media job artifacts: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusNoContent && response.StatusCode != http.StatusNotFound {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 1<<20))
		return fmt.Errorf("delete media job artifacts: engine returned %d: %s", response.StatusCode, strings.TrimSpace(string(body)))
	}
	return nil
}
