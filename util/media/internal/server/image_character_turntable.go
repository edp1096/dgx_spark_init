package server

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type characterTurntableFrame struct {
	Direction  string `json:"direction"`
	FrameIndex int    `json:"frame_index"`
	MimeType   string `json:"mime_type"`
	Data       string `json:"data"`
}

type characterTurntableResponse struct {
	OperationID string                    `json:"operation_id"`
	Seed        int64                     `json:"seed"`
	Frames      []characterTurntableFrame `json:"frames"`
}

// createImageSequenceCharacterTurntable asks the dedicated MMH3 runtime for a
// temporal 360-degree orbit. The runtime extracts calibrated direction frames;
// SparkMedia only validates and forwards those candidates for explicit review.
func (s *Server) createImageSequenceCharacterTurntable(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(64 << 20); err != nil {
		http.Error(w, "invalid or oversized character turntable form", http.StatusBadRequest)
		return
	}
	if err := os.MkdirAll(filepath.Join(s.dataDir, "temp"), 0o755); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	root, err := os.MkdirTemp(filepath.Join(s.dataDir, "temp"), "character-turntable-")
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
	operationID := strings.TrimSpace(r.FormValue("operation_id"))
	if !strings.HasPrefix(operationID, "character-turntable-") || len(operationID) > 128 {
		operationID = fmt.Sprintf("character-turntable-%d", time.Now().UnixNano())
	}
	endpoint := strings.TrimRight(s.config().Engines["character"].Endpoint, "/")
	if endpoint == "" {
		http.Error(w, "MMH3 character runtime is not configured", http.StatusServiceUnavailable)
		return
	}
	fields := map[string]string{"operation_id": operationID}
	if seed := strings.TrimSpace(r.FormValue("seed")); seed != "" {
		fields["seed"] = seed
	}
	s.heavyMu.Lock()
	data, _, callErr := s.callMultipartContext(r.Context(), endpoint+"/v1/character/turntable", fields, "image", paths)
	s.heavyMu.Unlock()
	if callErr != nil {
		http.Error(w, "character turntable: "+callErr.Error(), http.StatusBadGateway)
		return
	}
	var result characterTurntableResponse
	if err := json.Unmarshal(data, &result); err != nil || len(result.Frames) != 8 {
		http.Error(w, "character turntable returned invalid frame data", http.StatusBadGateway)
		return
	}
	for _, frame := range result.Frames {
		decoded, decodeErr := base64.StdEncoding.DecodeString(frame.Data)
		if decodeErr != nil || len(decoded) < 128 || !strings.HasPrefix(frame.MimeType, "image/") {
			http.Error(w, "character turntable returned an invalid frame", http.StatusBadGateway)
			return
		}
	}
	w.Header().Set("Cache-Control", "no-store")
	writeJSON(w, http.StatusOK, result)
}

func (s *Server) imageSequenceCharacterTurntableStatus(w http.ResponseWriter, r *http.Request) {
	operationID := strings.TrimSpace(r.URL.Query().Get("operation_id"))
	if !strings.HasPrefix(operationID, "character-turntable-") || len(operationID) > 128 {
		http.Error(w, "invalid character turntable operation id", http.StatusBadRequest)
		return
	}
	endpoint := strings.TrimRight(s.config().Engines["character"].Endpoint, "/")
	request, err := http.NewRequestWithContext(r.Context(), http.MethodGet, endpoint+"/v1/character/turntable/status?operation_id="+url.QueryEscape(operationID), nil)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	response, err := s.client.Do(request)
	if err != nil {
		http.Error(w, "character turntable progress unavailable", http.StatusBadGateway)
		return
	}
	defer response.Body.Close()
	if response.StatusCode/100 != 2 {
		http.Error(w, "character turntable progress unavailable", http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	_, _ = io.Copy(w, response.Body)
}
