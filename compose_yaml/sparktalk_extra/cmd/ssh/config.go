package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

type config struct {
	ListenAddr     string
	KeyDir         string
	KnownHostsPath string
	MaxConcurrency int
	MaxOutputBytes int64
	CommandTimeout time.Duration
}

func loadConfig() (config, error) {
	concurrency, err := envInt("SPARKTALK_EXTRA_SSH_MAX_CONCURRENCY", 2)
	if err != nil || concurrency < 1 || concurrency > 16 {
		return config{}, fmt.Errorf("invalid SPARKTALK_EXTRA_SSH_MAX_CONCURRENCY")
	}
	maxOutputMB, err := envInt64("SPARKTALK_EXTRA_SSH_MAX_OUTPUT_MB", 4)
	if err != nil || maxOutputMB < 1 || maxOutputMB > 128 {
		return config{}, fmt.Errorf("invalid SPARKTALK_EXTRA_SSH_MAX_OUTPUT_MB")
	}
	timeoutSeconds, err := envInt64("SPARKTALK_EXTRA_SSH_TIMEOUT_SECONDS", 300)
	if err != nil || timeoutSeconds < 1 || timeoutSeconds > 86400 {
		return config{}, fmt.Errorf("invalid SPARKTALK_EXTRA_SSH_TIMEOUT_SECONDS")
	}
	cfg := config{
		ListenAddr:     env("SPARKTALK_EXTRA_SSH_LISTEN_ADDR", "0.0.0.0:8699"),
		KeyDir:         env("SPARKTALK_EXTRA_SSH_KEY_DIR", "/run/sparktalk-extra/keys"),
		KnownHostsPath: env("SPARKTALK_EXTRA_SSH_KNOWN_HOSTS", "/var/lib/sparktalk-extra/known_hosts"),
		MaxConcurrency: concurrency,
		MaxOutputBytes: maxOutputMB * 1024 * 1024,
		CommandTimeout: time.Duration(timeoutSeconds) * time.Second,
	}
	if err := os.MkdirAll(filepath.Dir(cfg.KnownHostsPath), 0o700); err != nil {
		return config{}, fmt.Errorf("create SSH state directory: %w", err)
	}
	file, err := os.OpenFile(cfg.KnownHostsPath, os.O_CREATE|os.O_APPEND, 0o600)
	if err != nil {
		return config{}, fmt.Errorf("open known_hosts: %w", err)
	}
	if err := file.Close(); err != nil {
		return config{}, fmt.Errorf("close known_hosts: %w", err)
	}
	return cfg, nil
}

func env(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}

func envInt(name string, fallback int) (int, error) {
	value := os.Getenv(name)
	if value == "" {
		return fallback, nil
	}
	return strconv.Atoi(value)
}

func envInt64(name string, fallback int64) (int64, error) {
	value := os.Getenv(name)
	if value == "" {
		return fallback, nil
	}
	return strconv.ParseInt(value, 10, 64)
}
