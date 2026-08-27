package server

import (
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
)

func (s *Server) proxyLoRA(w http.ResponseWriter, r *http.Request) {
	endpoint := s.config().Engines["image"].Endpoint
	target, err := url.Parse(endpoint)
	if err != nil || target.Host == "" {
		http.Error(w, "invalid LoRA manager endpoint", http.StatusBadGateway)
		return
	}
	proxy := httputil.NewSingleHostReverseProxy(target)
	proxy.ErrorHandler = func(w http.ResponseWriter, _ *http.Request, err error) {
		http.Error(w, "LoRA manager unavailable: "+err.Error(), http.StatusBadGateway)
	}
	originalDirector := proxy.Director
	proxy.Director = func(request *http.Request) {
		originalDirector(request)
		path := strings.TrimPrefix(request.URL.Path, "/api/lora")
		request.URL.Path = "/v1/user-loras" + path
		request.URL.RawPath = ""
	}
	proxy.ServeHTTP(w, r)
}
