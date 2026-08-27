package server

import (
	"net/http"
)

func (s *Server) engineStates(w http.ResponseWriter, _ *http.Request) {
	cfg := s.config()
	type state struct {
		Kind   string `json:"kind"`
		Status string `json:"status"`
	}
	states := make([]state, 0, 10)
	probe := func(endpoint, healthPath string) string {
		status := "offline"
		resp, err := s.health.Get(endpoint + healthPath)
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				status = "online"
			}
		}
		return status
	}
	defaultImageStatus := "offline"
	for _, mode := range []string{"create", "edit", "control"} {
		backend, ok := cfg.Image.Backends[mode]
		if !ok {
			continue
		}
		status := probe(backend.Endpoint, "/health")
		states = append(states, state{Kind: "image_" + mode, Status: status})
		if mode == cfg.Image.DefaultMode {
			defaultImageStatus = status
		}
	}
	states = append(states, state{Kind: "image", Status: defaultImageStatus})
	for _, kind := range []string{"speech", "recognition", "video", "prompt", "media", "trainer", "upscale", "garment"} {
		healthPath := "/health"
		if kind == "prompt" {
			healthPath = "/v1/models"
		}
		status := probe(cfg.Engines[kind].Endpoint, healthPath)
		states = append(states, state{Kind: kind, Status: status})
	}
	writeJSON(w, 200, states)
}
