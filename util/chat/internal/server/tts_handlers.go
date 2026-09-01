package server

import (
	"encoding/json"
	"io"
	"net/http"
	"strconv"
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
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxTTSRequestBytes)).Decode(&request); err != nil {
		http.Error(w, "invalid TTS request", http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(request.Text) == "" {
		http.Error(w, "text is required", http.StatusBadRequest)
		return
	}
	s.ttsMu.Lock()
	defer s.ttsMu.Unlock()
	client := s.ttsSnapshot()
	parts := client.SpeechParts(request.Text)
	if len(parts) == 0 {
		http.Error(w, "text is required", http.StatusBadRequest)
		return
	}
	stream, err := client.SpeechStreamLanguage(r.Context(), parts[0].Text, parts[0].Language)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", "audio/pcm")
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Audio-Sample-Rate", strconv.Itoa(stream.SampleRate))
	w.Header().Set("X-Audio-Channels", "1")
	w.Header().Set("X-Audio-Sample-Format", "s16le")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	buffer := make([]byte, 32<<10)
	for index := range parts {
		if index > 0 {
			stream, err = client.SpeechStreamLanguage(r.Context(), parts[index].Text, parts[index].Language)
			if err != nil {
				return
			}
		}
		for {
			n, readErr := stream.Body.Read(buffer)
			if n > 0 {
				if _, writeErr := w.Write(buffer[:n]); writeErr != nil {
					stream.Body.Close()
					return
				}
				if flusher != nil {
					flusher.Flush()
				}
			}
			if readErr != nil {
				stream.Body.Close()
				if readErr != io.EOF {
					return
				}
				break
			}
		}
	}
}
