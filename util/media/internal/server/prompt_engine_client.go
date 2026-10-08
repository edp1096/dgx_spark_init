package server

import "strings"

func (s *Server) chatWithPromptEngine(payload any) ([]byte, error) {
	endpoint := strings.TrimRight(s.config().Engines["prompt"].Endpoint, "/") + "/v1/chat/completions"
	data, _, err := s.callJSON(endpoint, payload)
	return data, err
}
