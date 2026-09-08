package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

type config struct {
	ListenAddr     string
	ChromiumPath   string
	MaxBytes       int64
	MaxConcurrency int
	Timeout        time.Duration
	BrowserWait    time.Duration
}

func loadConfig() (config, error) {
	cfg := config{
		ListenAddr:     env("SPARKTALK_EXTRA_COLLECTOR_LISTEN_ADDR", "0.0.0.0:8695"),
		ChromiumPath:   env("SPARKTALK_EXTRA_COLLECTOR_CHROMIUM_PATH", "/usr/bin/chromium"),
		MaxBytes:       int64(envInt("SPARKTALK_EXTRA_COLLECTOR_MAX_MB", 256)) << 20,
		MaxConcurrency: envInt("SPARKTALK_EXTRA_COLLECTOR_MAX_CONCURRENCY", 1),
		Timeout:        time.Duration(envInt("SPARKTALK_EXTRA_COLLECTOR_TIMEOUT_SECONDS", 120)) * time.Second,
		BrowserWait:    time.Duration(envInt("SPARKTALK_EXTRA_COLLECTOR_BROWSER_WAIT_MS", 1200)) * time.Millisecond,
	}
	if cfg.MaxBytes < 1<<20 || cfg.MaxBytes > 512<<20 {
		return config{}, fmt.Errorf("collector max size must be between 1 and 512 MB")
	}
	if cfg.MaxConcurrency < 1 || cfg.MaxConcurrency > 4 {
		return config{}, fmt.Errorf("collector max concurrency must be between 1 and 4")
	}
	if cfg.Timeout < 10*time.Second || cfg.Timeout > 30*time.Minute {
		return config{}, fmt.Errorf("collector timeout must be between 10 and 1800 seconds")
	}
	return cfg, nil
}

func env(name, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(name)); value != "" {
		return value
	}
	return fallback
}

func envInt(name string, fallback int) int {
	value, err := strconv.Atoi(strings.TrimSpace(os.Getenv(name)))
	if err != nil {
		return fallback
	}
	return value
}
