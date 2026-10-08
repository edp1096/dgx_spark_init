package server

import "context"

func (s *Server) generateSpeechWithEngine(ctx context.Context, request map[string]any) ([]byte, error) {
	endpoint := s.config().Engines["speech"].Endpoint + "/v1/audio/speech"
	data, _, err := s.callJSONContext(ctx, endpoint, request)
	return data, err
}

func (s *Server) transcribeWithEngine(fields map[string]string, path string) ([]byte, error) {
	endpoint := s.config().Engines["recognition"].Endpoint + "/v1/audio/transcriptions"
	data, _, err := s.callMultipart(endpoint, fields, "file", []string{path})
	return data, err
}
