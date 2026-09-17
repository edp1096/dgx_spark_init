package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const ytReleaseAPI = "https://api.github.com/repos/yt-dlp/yt-dlp/releases/latest"

func (a *api) updateDir() string {
	if a.cfg.UpdateDir != "" {
		return a.cfg.UpdateDir
	}
	return "/var/lib/sparktalk-extra/media/bin"
}
func (a *api) currentYtdlp() string {
	p := filepath.Join(a.updateDir(), "yt-dlp")
	if isExecutable(p) {
		return p
	}
	return a.cfg.YtDLPPath
}
func (a *api) ytdlpStatus() map[string]any {
	version, err := ytDLPVersion(a.currentYtdlp())
	out := map[string]any{"current": version, "can_rollback": isExecutable(filepath.Join(a.updateDir(), "yt-dlp.previous"))}
	if err != nil {
		out["error"] = err.Error()
	}
	return out
}

type ytRelease struct {
	Tag    string `json:"tag_name"`
	Assets []struct {
		Name   string `json:"name"`
		URL    string `json:"browser_download_url"`
		Digest string `json:"digest"`
	} `json:"assets"`
}

func downloadRuntime(ctx context.Context, url string, limit int64) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "SparkTalk-runtime-updater")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("download HTTP %d", resp.StatusCode)
	}
	b, err := io.ReadAll(io.LimitReader(resp.Body, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(b)) > limit {
		return nil, fmt.Errorf("download exceeds size limit")
	}
	return b, nil
}
func latestYtdlp(ctx context.Context) (ytRelease, error) {
	var release ytRelease
	b, err := downloadRuntime(ctx, ytReleaseAPI, 2<<20)
	if err != nil {
		return release, err
	}
	err = json.Unmarshal(b, &release)
	if err == nil && release.Tag == "" {
		err = fmt.Errorf("release tag missing")
	}
	return release, err
}
func (a *api) ytdlpRuntime(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 3*time.Minute)
	defer cancel()
	if r.Method == http.MethodGet {
		out := a.ytdlpStatus()
		if r.URL.Query().Get("check") == "1" {
			release, err := latestYtdlp(ctx)
			if err != nil {
				http.Error(w, err.Error(), 502)
				return
			}
			out["latest"] = release.Tag
			out["update_available"] = release.Tag != out["current"]
		}
		writeRuntimeJSON(w, out)
		return
	}
	if !a.updateMu.TryLock() {
		http.Error(w, "yt-dlp update already in progress", 409)
		return
	}
	defer a.updateMu.Unlock()
	var err error
	switch r.URL.Path {
	case "/v1/runtime/yt-dlp/update":
		err = a.installYtdlp(ctx)
	case "/v1/runtime/yt-dlp/rollback":
		err = a.rollbackYtdlp(ctx)
	default:
		http.NotFound(w, r)
		return
	}
	if err != nil {
		http.Error(w, err.Error(), 502)
		return
	}
	writeRuntimeJSON(w, a.ytdlpStatus())
}
func writeRuntimeJSON(w http.ResponseWriter, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	json.NewEncoder(w).Encode(value)
}
func (a *api) installYtdlp(ctx context.Context) error {
	release, err := latestYtdlp(ctx)
	if err != nil {
		return err
	}
	current, err := ytDLPVersion(a.currentYtdlp())
	if err != nil {
		return err
	}
	if current == release.Tag {
		return nil
	}
	name := "yt-dlp_linux"
	if runtime.GOARCH == "arm64" {
		name = "yt-dlp_linux_aarch64"
	} else if runtime.GOARCH != "amd64" {
		return fmt.Errorf("unsupported architecture")
	}
	var url, digest string
	for _, asset := range release.Assets {
		if asset.Name == name {
			url, digest = asset.URL, strings.TrimPrefix(asset.Digest, "sha256:")
			break
		}
	}
	if !strings.HasPrefix(url, "https://github.com/yt-dlp/yt-dlp/releases/download/") || len(digest) != 64 {
		return fmt.Errorf("verified release asset unavailable")
	}
	data, err := downloadRuntime(ctx, url, 128<<20)
	if err != nil {
		return err
	}
	sum := sha256.Sum256(data)
	if hex.EncodeToString(sum[:]) != digest {
		return fmt.Errorf("yt-dlp SHA256 mismatch; active version unchanged")
	}
	candidate, err := a.stageBinary(data)
	if err != nil {
		return err
	}
	defer os.Remove(candidate)
	command := exec.CommandContext(ctx, candidate, "--version")
	out, err := command.Output()
	if err != nil || strings.TrimSpace(string(out)) != release.Tag {
		return fmt.Errorf("candidate version validation failed; active version unchanged")
	}
	return a.activateYtdlp(candidate)
}
func (a *api) stageBinary(data []byte) (string, error) {
	if err := os.MkdirAll(a.updateDir(), 0755); err != nil {
		return "", err
	}
	f, err := os.CreateTemp(a.updateDir(), ".yt-dlp-")
	if err != nil {
		return "", err
	}
	name := f.Name()
	_, err = f.Write(data)
	if err == nil {
		err = f.Chmod(0755)
	}
	if err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err == nil {
		err = closeErr
	}
	if err != nil {
		os.Remove(name)
		return "", err
	}
	return name, nil
}
func (a *api) activateYtdlp(candidate string) error {
	old, err := os.ReadFile(a.currentYtdlp())
	if err != nil {
		return err
	}
	backup, err := a.stageBinary(old)
	if err != nil {
		return err
	}
	defer os.Remove(backup)
	if err = os.Rename(backup, filepath.Join(a.updateDir(), "yt-dlp.previous")); err != nil {
		return err
	}
	return os.Rename(candidate, filepath.Join(a.updateDir(), "yt-dlp"))
}
func (a *api) rollbackYtdlp(ctx context.Context) error {
	data, err := os.ReadFile(filepath.Join(a.updateDir(), "yt-dlp.previous"))
	if err != nil {
		return fmt.Errorf("rollback version unavailable: %w", err)
	}
	candidate, err := a.stageBinary(data)
	if err != nil {
		return err
	}
	defer os.Remove(candidate)
	if _, err = exec.CommandContext(ctx, candidate, "--version").Output(); err != nil {
		return fmt.Errorf("rollback binary invalid: %w", err)
	}
	return a.activateYtdlp(candidate)
}
