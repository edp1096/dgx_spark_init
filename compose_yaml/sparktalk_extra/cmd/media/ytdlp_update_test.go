package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

type runtimeTransport func(*http.Request) (*http.Response, error)

func (f runtimeTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
func TestYtdlpVerifiedUpdateRollbackAndRestart(t *testing.T) {
	dir := t.TempDir()
	base := filepath.Join(dir, "base")
	os.WriteFile(base, []byte("#!/bin/sh\necho 2026.08.19\n"), 0755)
	a := &api{cfg: config{YtDLPPath: base, UpdateDir: filepath.Join(dir, "runtime")}}
	candidate := []byte("#!/bin/sh\necho 2026.09.01\n")
	sum := sha256.Sum256(candidate)
	digest := "sha256:" + hex.EncodeToString(sum[:])
	bad := false
	original := http.DefaultTransport
	defer func() { http.DefaultTransport = original }()
	http.DefaultTransport = runtimeTransport(func(r *http.Request) (*http.Response, error) {
		var body []byte
		if r.URL.String() == ytReleaseAPI {
			name := "yt-dlp_linux"
			if runtime.GOARCH == "arm64" {
				name += "_aarch64"
			}
			d := digest
			if bad {
				d = "sha256:" + strings.Repeat("0", 64)
			}
			body, _ = json.Marshal(map[string]any{"tag_name": "2026.09.01", "assets": []map[string]string{{"name": name, "digest": d, "browser_download_url": "https://github.com/yt-dlp/yt-dlp/releases/download/2026.09.01/" + name}}})
		} else {
			body = candidate
		}
		return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(string(body))), Header: make(http.Header)}, nil
	})
	bad = true
	if err := a.installYtdlp(context.Background()); err == nil {
		t.Fatal("bad checksum accepted")
	}
	if v, _ := ytDLPVersion(a.currentYtdlp()); v != "2026.08.19" {
		t.Fatal("failed update changed active binary")
	}
	bad = false
	if err := a.installYtdlp(context.Background()); err != nil {
		t.Fatal(err)
	}
	restarted := &api{cfg: a.cfg}
	if v, _ := ytDLPVersion(restarted.currentYtdlp()); v != "2026.09.01" {
		t.Fatal("update did not persist")
	}
	if err := a.rollbackYtdlp(context.Background()); err != nil {
		t.Fatal(err)
	}
	if v, _ := ytDLPVersion(a.currentYtdlp()); v != "2026.08.19" {
		t.Fatal("rollback failed")
	}
	if err := a.rollbackYtdlp(context.Background()); err != nil {
		t.Fatal(err)
	}
	if v, _ := ytDLPVersion(a.currentYtdlp()); v != "2026.09.01" {
		t.Fatal("rollback swap failed")
	}
}
