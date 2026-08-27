package server

import (
	"context"
	"mediaapp/internal/config"
)

func (s *Server) generateImageWithEngine(ctx context.Context, backend config.ImageBackend, request map[string]any) ([]byte, error) {
	data, _, err := s.callJSONContext(ctx, backend.Endpoint+"/v1/images/generations", request)
	return data, err
}

func (s *Server) editImageWithEngine(ctx context.Context, backend config.ImageBackend, fields map[string]string, references []string) ([]byte, error) {
	data, _, err := s.callMultipartContext(ctx, backend.Endpoint+"/v1/images/edits", fields, "image", references)
	return data, err
}

func (s *Server) upscaleImageWithEngine(ctx context.Context, request map[string]any) ([]byte, error) {
	data, _, err := s.callJSONContext(ctx, s.config().Engines["upscale"].Endpoint+"/v1/images/upscale", request)
	return data, err
}
