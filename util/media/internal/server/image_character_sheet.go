package server

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// createImageSequenceCharacterSheet produces an experimental QuadView candidate.
// It is returned for explicit user review and is never promoted to a ReID anchor
// automatically because CharacterSheet may reinterpret the source identity.
func (s *Server) createImageSequenceCharacterSheet(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(64 << 20); err != nil {
		http.Error(w, "invalid or oversized character sheet form", http.StatusBadRequest)
		return
	}
	tempParent := filepath.Join(s.dataDir, "temp")
	if err := os.MkdirAll(tempParent, 0o755); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	root, err := os.MkdirTemp(tempParent, "character-sheet-")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer os.RemoveAll(root)
	paths, err := saveUploads(r, "image", root, 1)
	if err == nil {
		paths, err = s.appendReusedImageInputs(r, "reuse_image", root, 1, paths)
	}
	if err != nil || len(paths) != 1 {
		http.Error(w, "select exactly one representative character image", http.StatusBadRequest)
		return
	}
	data, err := os.ReadFile(paths[0])
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	seed := time.Now().UnixNano() & 0x7fffffffffffffff
	if parsed, parseErr := strconv.ParseInt(strings.TrimSpace(r.FormValue("seed")), 10, 64); parseErr == nil && parsed >= 0 {
		seed = parsed
	}
	cfg := s.config()
	backend, ok := cfg.Image.Backends["create"]
	if !ok || strings.TrimSpace(backend.Endpoint) == "" {
		http.Error(w, "Krea image backend is not configured", http.StatusServiceUnavailable)
		return
	}
	operationID := strings.TrimSpace(r.FormValue("operation_id"))
	if !strings.HasPrefix(operationID, "character-sheet-") || len(operationID) > 128 {
		operationID = fmt.Sprintf("character-sheet-%d", time.Now().UnixNano())
	}
	payload := map[string]any{
		"model": backend.Model, "prompt": "Create a four-view character sheet candidate for user review.",
		"checkpoint": "official", "size": "1536x1024", "steps": 10, "seed": seed,
		"response_format": "b64_json", "output_format": "png", "filter_mode": "balanced",
		"character_sheet_image": base64.StdEncoding.EncodeToString(data),
		"operation_id":          operationID,
	}
	s.heavyMu.Lock()
	response, err := s.generateImageWithEngine(r.Context(), backend, payload)
	s.heavyMu.Unlock()
	if err != nil {
		http.Error(w, "character sheet: "+err.Error(), http.StatusBadGateway)
		return
	}
	imageData, err := decodeImage(response)
	if err != nil {
		http.Error(w, "character sheet returned invalid image data", http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", "image/png")
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Character-Sheet-Seed", strconv.FormatInt(seed, 10))
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(imageData)
}

// imageSequenceCharacterSheetStatus exposes only the correlated Krea runtime
// operation used by the character-sheet modal. The engine remains the source
// of truth for model loading, conditioning, sampling and decoding progress.
func (s *Server) imageSequenceCharacterSheetStatus(w http.ResponseWriter, r *http.Request) {
	operationID := strings.TrimSpace(r.URL.Query().Get("operation_id"))
	if !strings.HasPrefix(operationID, "character-sheet-") || len(operationID) > 128 {
		http.Error(w, "invalid character sheet operation id", http.StatusBadRequest)
		return
	}
	backend, ok := s.config().Image.Backends["create"]
	if !ok || strings.TrimSpace(backend.Endpoint) == "" {
		http.Error(w, "Krea image backend is not configured", http.StatusServiceUnavailable)
		return
	}
	statusURL := strings.TrimRight(backend.Endpoint, "/") + "/v1/models/runtime/status?operation_id=" + url.QueryEscape(operationID)
	request, err := http.NewRequestWithContext(r.Context(), http.MethodGet, statusURL, nil)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	response, err := s.client.Do(request)
	if err != nil {
		http.Error(w, "character sheet progress unavailable", http.StatusBadGateway)
		return
	}
	defer response.Body.Close()
	if response.StatusCode/100 != 2 {
		http.Error(w, "character sheet progress unavailable", http.StatusBadGateway)
		return
	}
	var status map[string]any
	if err := json.NewDecoder(response.Body).Decode(&status); err != nil {
		http.Error(w, "invalid character sheet progress", http.StatusBadGateway)
		return
	}
	writeJSON(w, http.StatusOK, status)
}
