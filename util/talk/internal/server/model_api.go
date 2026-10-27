package server

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strconv"
	"strings"
)

const modelAPIAlias = "sparktalk"

func modelAPIError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{"message": message, "type": "api_error"}})
}

func (s *Server) modelAPIInfo(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		modelAPIError(w, 405, "Method not allowed")
		return
	}
	cfg, _ := s.snapshot()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{"base_path": "/v1", "model": modelAPIAlias, "active_model": cfg.Model.DefaultModel})
}

// Each request uses one configuration snapshot: switching models affects new
// requests, never the destination of an in-flight stream. No Talk tool loop runs.
func (s *Server) modelAPI(w http.ResponseWriter, r *http.Request) {
	if r.Header.Get("X-SparkTalk-Proxy") != "" {
		modelAPIError(w, 508, "Model API routing loop")
		return
	}
	path := r.URL.Path
	if path != "/v1/models" && path != "/v1/chat/completions" && path != "/v1/completions" && path != "/v1/responses" {
		modelAPIError(w, 404, "Unsupported model API endpoint")
		return
	}
	method := http.MethodPost
	if path == "/v1/models" {
		method = http.MethodGet
	}
	if r.Method != method {
		w.Header().Set("Allow", method)
		modelAPIError(w, 405, "Method not allowed")
		return
	}
	cfg, _ := s.snapshot()
	target, err := url.Parse(strings.TrimRight(cfg.Model.Endpoint, "/"))
	if err != nil || target.Host == "" || (target.Scheme != "http" && target.Scheme != "https") || cfg.Model.DefaultModel == "" {
		modelAPIError(w, 503, "No model API configured")
		return
	}
	if path == "/v1/models" {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": []any{map[string]any{"id": modelAPIAlias, "object": "model", "created": 0, "owned_by": "sparktalk"}}})
		return
	}
	// Decode only the envelope. Raw fields preserve tools, images, reasoning and
	// engine-specific options without converting numbers through float64.
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 64<<20))
	if err != nil {
		modelAPIError(w, 413, "Request body exceeds limit or could not be read")
		return
	}
	var fields map[string]json.RawMessage
	if json.Unmarshal(body, &fields) != nil || fields == nil {
		modelAPIError(w, 400, "Expected a JSON object")
		return
	}
	var requested string
	if raw, ok := fields["model"]; ok {
		if json.Unmarshal(raw, &requested) != nil {
			modelAPIError(w, 400, "model must be a string")
			return
		}
	}
	if requested != "" && requested != modelAPIAlias && requested != cfg.Model.DefaultModel {
		modelAPIError(w, 400, "Use model sparktalk to follow the active model")
		return
	}
	fields["model"], _ = json.Marshal(cfg.Model.DefaultModel)
	body, _ = json.Marshal(fields)
	target.Path = strings.TrimSuffix(strings.TrimRight(target.Path, "/"), "/v1")
	proxy := &httputil.ReverseProxy{
		FlushInterval: -1,
		Rewrite: func(pr *httputil.ProxyRequest) {
			pr.SetURL(target)
			pr.Out.Header.Del("Authorization")
			pr.Out.Header.Del("Cookie")
			pr.Out.Header.Del("X-Api-Key")
			pr.Out.Header.Set("X-SparkTalk-Proxy", "1")
			if cfg.Model.APIKey != "" {
				pr.Out.Header.Set("Authorization", "Bearer "+cfg.Model.APIKey)
			}
			pr.Out.Header.Set("Content-Type", "application/json")
			pr.Out.Header.Set("Content-Length", strconv.Itoa(len(body)))
			pr.Out.Body = io.NopCloser(bytes.NewReader(body))
			pr.Out.ContentLength = int64(len(body))
		},
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			modelAPIError(w, 502, "Active model server is unavailable")
		},
	}
	proxy.ServeHTTP(w, r)
}
