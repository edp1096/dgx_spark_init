package main

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

type config struct {
	ListenAddr     string
	TempDir        string
	MaxUploadBytes int64
	MaxConcurrency int
	ProcessTimeout time.Duration
	FFmpegPath     string
	FFprobePath    string
	YtDLPPath      string
	MaxDownloadMB  int64
	MaxDurationSec int64
	MaxVideoHeight int
}

func loadConfig() (config, error) {
	maxUploadMB, err := envInt64Any([]string{"SPARKTALK_MEDIA_API_MAX_UPLOAD_MB", "SPARK_MEDIA_API_MAX_UPLOAD_MB", "FFMPEG_API_MAX_UPLOAD_MB"}, 512)
	if err != nil || maxUploadMB < 1 {
		return config{}, fmt.Errorf("invalid SPARKTALK_MEDIA_API_MAX_UPLOAD_MB")
	}
	maxConcurrency, err := envIntAny([]string{"SPARKTALK_MEDIA_API_MAX_CONCURRENCY", "SPARK_MEDIA_API_MAX_CONCURRENCY", "FFMPEG_API_MAX_CONCURRENCY"}, 2)
	if err != nil || maxConcurrency < 1 {
		return config{}, fmt.Errorf("invalid SPARKTALK_MEDIA_API_MAX_CONCURRENCY")
	}
	timeoutSeconds, err := envInt64Any([]string{"SPARKTALK_MEDIA_API_TIMEOUT_SECONDS", "SPARK_MEDIA_API_TIMEOUT_SECONDS", "FFMPEG_API_TIMEOUT_SECONDS"}, 1800)
	if err != nil || timeoutSeconds < 1 {
		return config{}, fmt.Errorf("invalid SPARKTALK_MEDIA_API_TIMEOUT_SECONDS")
	}
	maxDownloadMB, err := envInt64Any([]string{"SPARKTALK_MEDIA_API_MAX_DOWNLOAD_MB", "SPARK_MEDIA_API_MAX_DOWNLOAD_MB"}, 4096)
	if err != nil || maxDownloadMB < 1 {
		return config{}, fmt.Errorf("invalid SPARKTALK_MEDIA_API_MAX_DOWNLOAD_MB")
	}
	maxDurationSec, err := envInt64Any([]string{"SPARKTALK_MEDIA_API_MAX_DURATION_SECONDS", "SPARK_MEDIA_API_MAX_DURATION_SECONDS"}, 14400)
	if err != nil || maxDurationSec < 1 {
		return config{}, fmt.Errorf("invalid SPARKTALK_MEDIA_API_MAX_DURATION_SECONDS")
	}
	maxVideoHeight, err := envIntAny([]string{"SPARKTALK_MEDIA_API_MAX_VIDEO_HEIGHT", "SPARK_MEDIA_API_MAX_VIDEO_HEIGHT"}, 720)
	if err != nil || maxVideoHeight < 144 {
		return config{}, fmt.Errorf("invalid SPARKTALK_MEDIA_API_MAX_VIDEO_HEIGHT")
	}
	ytDLPPath := env("YTDLP_PATH", "/usr/local/bin/yt-dlp")
	if override := "/var/lib/sparktalk-media-api/bin/yt-dlp"; isExecutable(override) {
		ytDLPPath = override
	}
	return config{
		ListenAddr:     envAny([]string{"SPARKTALK_MEDIA_API_LISTEN_ADDR", "SPARK_MEDIA_API_LISTEN_ADDR", "FFMPEG_API_LISTEN_ADDR"}, "0.0.0.0:8698"),
		TempDir:        envAny([]string{"SPARKTALK_MEDIA_API_TEMP_DIR", "SPARK_MEDIA_API_TEMP_DIR", "FFMPEG_API_TEMP_DIR"}, "/tmp/sparktalk-media-api"),
		MaxUploadBytes: maxUploadMB * 1024 * 1024,
		MaxConcurrency: maxConcurrency,
		ProcessTimeout: time.Duration(timeoutSeconds) * time.Second,
		FFmpegPath:     env("FFMPEG_PATH", "ffmpeg"),
		FFprobePath:    env("FFPROBE_PATH", "ffprobe"),
		YtDLPPath:      ytDLPPath,
		MaxDownloadMB:  maxDownloadMB,
		MaxDurationSec: maxDurationSec,
		MaxVideoHeight: maxVideoHeight,
	}, nil
}

func isExecutable(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir() && info.Mode()&0o111 != 0
}

func envAny(names []string, fallback string) string {
	for _, name := range names {
		if value := os.Getenv(name); value != "" {
			return value
		}
	}
	return fallback
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

func envIntAny(names []string, fallback int) (int, error) {
	for _, name := range names {
		if value := os.Getenv(name); value != "" {
			return strconv.Atoi(value)
		}
	}
	return fallback, nil
}

func envInt64Any(names []string, fallback int64) (int64, error) {
	for _, name := range names {
		if value := os.Getenv(name); value != "" {
			return strconv.ParseInt(value, 10, 64)
		}
	}
	return fallback, nil
}
