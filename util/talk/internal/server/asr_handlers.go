package server

import (
	"errors"
	"net/http"
	"strings"

	"sparktalk/internal/asr"
)

const maxVoiceRecordingBytes = 32 << 20

func (s *Server) transcribeVoice(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	cfg, _ := s.snapshot()
	if !cfg.ASR.Enabled {
		http.Error(w, "ASR is disabled", http.StatusServiceUnavailable)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxVoiceRecordingBytes+(1<<20))
	if err := r.ParseMultipartForm(8 << 20); err != nil {
		http.Error(w, "invalid voice recording or recording is too large", http.StatusBadRequest)
		return
	}
	defer r.MultipartForm.RemoveAll()
	file, header, err := r.FormFile("audio")
	if err != nil {
		http.Error(w, "audio recording is required", http.StatusBadRequest)
		return
	}
	defer file.Close()
	mimeType := strings.TrimSpace(header.Header.Get("Content-Type"))
	if mimeType == "" {
		mimeType = "application/octet-stream"
	}

	// The local ASR service is intentionally serialized. Voice dictation and
	// attachment transcription share the same queue but use separate languages.
	s.asrMu.Lock()
	defer s.asrMu.Unlock()
	result, err := s.asrSnapshot().TranscribeVoice(r.Context(), file, header.Filename, mimeType)
	if err != nil {
		if errors.Is(err, asr.ErrNoAudio) {
			http.Error(w, "recording has no audio", http.StatusUnprocessableEntity)
			return
		}
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	writeJSON(w, http.StatusOK, result)
}
