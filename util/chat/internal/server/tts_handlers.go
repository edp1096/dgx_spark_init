package server

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
)

const maxTTSRequestBytes = 128 << 10

func (s *Server) synthesizeSpeech(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	var request struct {
		Text string `json:"text"`
		Seed *int64 `json:"seed,omitempty"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxTTSRequestBytes)).Decode(&request); err != nil {
		http.Error(w, "invalid TTS request", http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(request.Text) == "" {
		http.Error(w, "text is required", http.StatusBadRequest)
		return
	}
	if request.Seed != nil && (*request.Seed < 0 || *request.Seed > 2147483647) {
		http.Error(w, "seed must be between 0 and 2147483647", http.StatusBadRequest)
		return
	}
	s.ttsMu.Lock()
	defer s.ttsMu.Unlock()
	stream, err := s.ttsSnapshot().SpeechStream(r.Context(), request.Text, request.Seed)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	defer stream.Body.Close()
	w.Header().Set("Content-Type", "audio/pcm")
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Audio-Sample-Rate", "24000")
	w.Header().Set("X-Audio-Channels", "1")
	w.Header().Set("X-Audio-Sample-Format", "s16le")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	buffer := make([]byte, 32<<10)
	for {
		n, readErr := stream.Body.Read(buffer)
		if n > 0 {
			if _, writeErr := w.Write(buffer[:n]); writeErr != nil {
				return
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if readErr != nil {
			if readErr != io.EOF {
				return
			}
			break
		}
	}
}
