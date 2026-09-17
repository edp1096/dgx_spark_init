package server

import (
	"context"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

func (s *Server) mediaRuntime(w http.ResponseWriter, r *http.Request) {
	path := "/v1/runtime/yt-dlp"
	if r.Method == http.MethodGet {
		if r.URL.Query().Get("check") == "1" {
			path += "?check=1"
		}
	} else if r.Method == http.MethodPost {
		if origin := r.Header.Get("Origin"); origin != "" {
			u, e := url.Parse(origin)
			if e != nil || u.Host != r.Host {
				http.Error(w, "cross-origin update rejected", 403)
				return
			}
		}
		switch strings.TrimPrefix(r.URL.Path, "/api/media/yt-dlp/") {
		case "update":
			path += "/update"
		case "rollback":
			path += "/rollback"
		default:
			http.NotFound(w, r)
			return
		}
	} else {
		methodNotAllowed(w)
		return
	}
	cfg, _ := s.snapshot()
	ctx, cancel := context.WithTimeout(r.Context(), 3*time.Minute)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, r.Method, strings.TrimRight(cfg.SupportEndpoint("media"), "/")+path, nil)
	if err != nil {
		http.Error(w, err.Error(), 502)
		return
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		http.Error(w, err.Error(), 502)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode == 404 {
		http.Error(w, "현재 미디어 서비스가 yt-dlp 업데이트 API를 제공하지 않습니다. 미디어 서비스를 갱신하세요.", 502)
		return
	}
	w.Header().Set("Content-Type", resp.Header.Get("Content-Type"))
	w.Header().Set("Cache-Control", "no-store")
	w.WriteHeader(resp.StatusCode)
	io.Copy(w, io.LimitReader(resp.Body, 1<<20))
}
