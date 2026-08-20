package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

type mediaStorageStatus struct {
	TemporaryDirectories   int   `json:"temporary_directories"`
	TemporaryBytes         int64 `json:"temporary_bytes"`
	ActiveDirectories      int   `json:"active_directories"`
	ReclaimableDirectories int   `json:"reclaimable_directories"`
	ReclaimableBytes       int64 `json:"reclaimable_bytes"`
}

type mediaCleanupResult struct {
	RemovedDirectories int   `json:"removed_directories"`
	RemovedBytes       int64 `json:"removed_bytes"`
}

func (s *Server) mediaStorage(w http.ResponseWriter, _ *http.Request) {
	target := strings.TrimRight(s.config().Engines["media"].Endpoint, "/") + "/v1/media/storage"
	response, err := s.client.Get(target)
	if err != nil {
		http.Error(w, "storage status: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 1<<20))
		http.Error(w, fmt.Sprintf("storage status returned %d: %s", response.StatusCode, strings.TrimSpace(string(body))), http.StatusBadGateway)
		return
	}
	var status mediaStorageStatus
	if err := json.NewDecoder(io.LimitReader(response.Body, 1<<20)).Decode(&status); err != nil {
		http.Error(w, "storage status: invalid response", http.StatusBadGateway)
		return
	}
	writeJSON(w, http.StatusOK, status)
}

func (s *Server) requestMediaTempCleanup(olderThanHours int) (mediaCleanupResult, error) {
	endpoint := strings.TrimRight(s.config().Engines["media"].Endpoint, "/") + "/v1/media/storage/temp"
	if olderThanHours > 0 {
		endpoint += "?" + url.Values{"older_than_hours": {strconv.Itoa(olderThanHours)}}.Encode()
	}
	request, err := http.NewRequest(http.MethodDelete, endpoint, nil)
	if err != nil {
		return mediaCleanupResult{}, err
	}
	response, err := s.client.Do(request)
	if err != nil {
		return mediaCleanupResult{}, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(response.Body, 1<<20))
		return mediaCleanupResult{}, fmt.Errorf("engine returned %d: %s", response.StatusCode, strings.TrimSpace(string(body)))
	}
	var result mediaCleanupResult
	if err := json.NewDecoder(io.LimitReader(response.Body, 1<<20)).Decode(&result); err != nil {
		return mediaCleanupResult{}, fmt.Errorf("invalid cleanup response: %w", err)
	}
	return result, nil
}

func (s *Server) cleanupMediaTemp(w http.ResponseWriter, _ *http.Request) {
	result, err := s.requestMediaTempCleanup(0)
	if err != nil {
		http.Error(w, "temporary cleanup: "+err.Error(), http.StatusBadGateway)
		return
	}
	writeJSON(w, http.StatusOK, result)
}

// CleanupStaleMediaTemp applies the saved retention policy when the app starts.
func (s *Server) CleanupStaleMediaTemp() (mediaCleanupResult, error) {
	cfg := s.config()
	if !cfg.Storage.CleanupOnStartup {
		return mediaCleanupResult{}, nil
	}
	return s.requestMediaTempCleanup(cfg.Storage.TempRetentionHours)
}
