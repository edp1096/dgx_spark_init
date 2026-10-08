package server

import (
	"context"
	"net/http"
)

func (s *Server) generateVideoWithEngine(ctx context.Context, fields map[string]string, files map[string][]string, output string) (http.Header, error) {
	endpoint := s.config().Engines["video"].Endpoint + "/v1/videos/generations"
	return s.callMultipartFilesToFileContext(ctx, endpoint, fields, files, output)
}
