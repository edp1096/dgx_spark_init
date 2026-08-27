package server

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
)

func (s *Server) videoModelStatus(w http.ResponseWriter, r *http.Request) {
	endpoint := s.config().Engines["video"].Endpoint
	req, err := http.NewRequestWithContext(r.Context(), http.MethodGet, endpoint+"/v1/models/status", nil)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	resp, err := s.health.Do(req)
	if err != nil {
		http.Error(w, "LTX model service unavailable: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, io.LimitReader(resp.Body, 1<<20))
}

func (s *Server) prepareVideoModels(w http.ResponseWriter, r *http.Request) {
	var request struct {
		HFToken string `json:"hf_token"`
	}
	decoder := json.NewDecoder(io.LimitReader(r.Body, 16<<10))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&request); err != nil {
		http.Error(w, "invalid model preparation request", http.StatusBadRequest)
		return
	}
	payload := map[string]string{"hf_token": strings.TrimSpace(request.HFToken)}
	data, _, err := s.callJSON(s.config().Engines["video"].Endpoint+"/v1/models/prepare", payload)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	_, _ = w.Write(data)
}

func (s *Server) imageCheckpointStatus(w http.ResponseWriter, r *http.Request) {
	endpoint := s.config().Engines["image"].Endpoint
	req, err := http.NewRequestWithContext(r.Context(), http.MethodGet, endpoint+"/v1/checkpoints/status", nil)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	resp, err := s.health.Do(req)
	if err != nil {
		http.Error(w, "Krea model service unavailable: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, io.LimitReader(resp.Body, 1<<20))
}

func (s *Server) prepareImageCheckpoints(w http.ResponseWriter, r *http.Request) {
	var request struct {
		CivitaiToken string   `json:"civitai_token"`
		HFToken      string   `json:"hf_token"`
		Variants     []string `json:"variants"`
	}
	decoder := json.NewDecoder(io.LimitReader(r.Body, 32<<10))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&request); err != nil {
		http.Error(w, "invalid checkpoint preparation request", http.StatusBadRequest)
		return
	}
	payload := map[string]any{
		"civitai_token": strings.TrimSpace(request.CivitaiToken),
		"hf_token":      strings.TrimSpace(request.HFToken),
		"variants":      request.Variants,
	}
	data, _, err := s.callJSON(s.config().Engines["image"].Endpoint+"/v1/checkpoints/prepare", payload)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	_, _ = w.Write(data)
}

func (s *Server) convertImageCheckpointsNVFP4(w http.ResponseWriter, r *http.Request) {
	var request struct {
		CivitaiToken     string   `json:"civitai_token"`
		Variants         []string `json:"variants"`
		RemoveBF16Source bool     `json:"remove_bf16_sources"`
	}
	decoder := json.NewDecoder(io.LimitReader(r.Body, 32<<10))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&request); err != nil {
		http.Error(w, "invalid NVFP4 conversion request", http.StatusBadRequest)
		return
	}
	payload := map[string]any{
		"civitai_token":       strings.TrimSpace(request.CivitaiToken),
		"variants":            request.Variants,
		"remove_bf16_sources": request.RemoveBF16Source,
	}
	data, _, err := s.callJSON(s.config().Engines["image"].Endpoint+"/v1/checkpoints/convert-nvfp4", payload)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	_, _ = w.Write(data)
}
