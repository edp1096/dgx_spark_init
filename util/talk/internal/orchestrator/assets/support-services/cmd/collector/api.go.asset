package main

import (
	"context"
	"encoding/json"
	"log"
	"mime"
	"net/http"
	"strconv"
	"strings"
	"time"
)

type collectorAPI struct {
	cfg config
	sem chan struct{}
}

func newCollectorAPI(cfg config) *collectorAPI {
	return &collectorAPI{cfg: cfg, sem: make(chan struct{}, cfg.MaxConcurrency)}
}

func (a *collectorAPI) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", a.health)
	mux.HandleFunc("POST /v1/collect", a.collect)
	return collectorLog(mux)
}

func (a *collectorAPI) health(w http.ResponseWriter, _ *http.Request) {
	writeCollectorJSON(w, http.StatusOK, map[string]any{
		"status": "ok", "browser": collectorExecutableAvailable(a.cfg.ChromiumPath),
		"active": len(a.sem), "max_concurrency": cap(a.sem),
	})
}

func (a *collectorAPI) collect(w http.ResponseWriter, r *http.Request) {
	var input struct {
		URL      string `json:"url"`
		Mode     string `json:"mode"`
		MaxBytes int64  `json:"max_bytes"`
	}
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&input); err != nil {
		writeCollectorError(w, http.StatusBadRequest, "invalid collection request")
		return
	}
	input.URL = strings.TrimSpace(input.URL)
	input.Mode = strings.ToLower(strings.TrimSpace(input.Mode))
	if input.Mode == "" {
		input.Mode = "auto"
	}
	if input.Mode != "auto" && input.Mode != "direct" && input.Mode != "browser" {
		writeCollectorError(w, http.StatusBadRequest, "mode must be auto, direct, or browser")
		return
	}
	if input.MaxBytes <= 0 || input.MaxBytes > a.cfg.MaxBytes {
		input.MaxBytes = a.cfg.MaxBytes
	}
	select {
	case a.sem <- struct{}{}:
		defer func() { <-a.sem }()
	case <-r.Context().Done():
		writeCollectorError(w, 499, "request canceled")
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), a.cfg.Timeout)
	defer cancel()
	item, err := collectURL(ctx, a.cfg, input.URL, input.Mode, input.MaxBytes)
	if err != nil {
		writeCollectorError(w, http.StatusBadGateway, err.Error())
		return
	}
	bundle, err := tempBundle(item)
	if err != nil {
		writeCollectorError(w, http.StatusInternalServerError, err.Error())
		return
	}
	w.Header().Set("Content-Type", "application/zip")
	w.Header().Set("Content-Disposition", mime.FormatMediaType("attachment", map[string]string{"filename": bundleName(item.Manifest.Title)}))
	w.Header().Set("X-SparkTalk-Collector-Method", item.Manifest.Method)
	w.Header().Set("X-SparkTalk-Collector-Final-URL", item.Manifest.FinalURL)
	w.Header().Set("Content-Length", strconv.Itoa(len(bundle)))
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(bundle)
}

func writeCollectorJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func writeCollectorError(w http.ResponseWriter, status int, message string) {
	writeCollectorJSON(w, status, map[string]string{"error": message})
}

func collectorLog(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		started := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(started).Round(time.Millisecond))
	})
}
