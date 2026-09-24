package server

import (
	"net/http"
	"strings"
)

// Read stored results only; opening the transcript must never launch inference.
func (s *Server) attachmentTranscript(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	id := strings.TrimPrefix(r.URL.Path, "/api/media/transcript/")
	cached, ok, err := s.media.LoadTranscript(id, "")
	if err != nil || !ok {
		http.Error(w, "저장된 전사가 없습니다. ASR을 활성화한 뒤 첨부를 다시 분석해 주세요.", http.StatusNotFound)
		return
	}
	writeJSON(w, http.StatusOK, cached)
}
