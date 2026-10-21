package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const testDownloadID = "0123456789abcdef0123456789abcdef"

func progressSnapshot(t *testing.T, a *api) downloadProgress {
	t.Helper()
	w := httptest.NewRecorder()
	a.sourceProgress(w, httptest.NewRequest("GET", "/v1/source/progress?id="+testDownloadID, nil))
	if w.Code != 200 {
		t.Fatal(w.Code, w.Body.String())
	}
	if strings.Contains(w.Body.String(), "private-url") {
		t.Fatal("leaked extractor data")
	}
	var p downloadProgress
	if err := json.Unmarshal(w.Body.Bytes(), &p); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestDownloadProgressSplitLinesAndStaleETA(t *testing.T) {
	a := &api{}
	a.beginDownload(testDownloadID)
	a.selectDownloadStreams(testDownloadID, sourceInfo{Formats: []sourceFormat{{ID: "v", Height: 240, VideoCodec: "avc1", AudioCodec: "none", FileSize: 100}, {ID: "a", VideoCodec: "none", AudioCodec: "mp4a"}}}, "v+a")
	w := &downloadProgressWriter{a: a, id: testDownloadID, diagnostics: limitedBuffer{Limit: 1024}}
	line := progressPrefix + `{"format_id":"v","progress":{"status":"finished","downloaded_bytes":100,"total_bytes":100,"filename":"private-url"}}` + "\n" + progressPrefix + `{"format_id":"a","progress":{"status":"downloading","downloaded_bytes":50,"total_bytes_estimate":200.5,"speed":10,"eta":15,"fragment_index":5,"fragment_count":20}}` + "\r"
	for i := 0; i < len(line); i += 7 {
		end := i + 7
		if end > len(line) {
			end = len(line)
		}
		w.Write([]byte(line[i:end]))
	}
	p := progressSnapshot(t, a)
	if p.Streams[0].Status != "finished" || p.Streams[1].Kind != "audio" || p.Streams[1].DownloadedBytes != 50 || p.Streams[1].ETA == nil || *p.Streams[1].ETA != 15 {
		t.Fatalf("lost live streams: %+v", p)
	}
	a.updateDownload(testDownloadID, func(p *downloadProgress) { p.Streams[1].updated = time.Now().Add(-20 * time.Second) })
	p = progressSnapshot(t, a)
	if !p.Streams[1].Stalled || p.Streams[1].ETA != nil || p.Streams[1].Speed != 0 {
		t.Fatal("stale ETA displayed", p)
	}
	if a.beginDownload(testDownloadID) {
		t.Fatal("duplicate ID accepted")
	}
	a.downloadStage(testDownloadID, "failed")
	if progressSnapshot(t, a).Stage != "failed" {
		t.Fatal("failure not reported")
	}
	old := a.downloads[testDownloadID]
	old.updated = time.Now().Add(-11 * time.Minute)
	a.downloads[testDownloadID] = old
	a.beginDownload("abcdef0123456789abcdef0123456789")
	if _, exists := a.downloads[testDownloadID]; exists {
		t.Fatal("completed progress not expired")
	}
}

// Uses the same flags and native progress template as production, with real
// yt-dlp and local HTTP data. No remote video, browser or model is required.
func TestNativeYTDLPDownloadProgress(t *testing.T) {
	binary := os.Getenv("YTDLP_TEST_BINARY")
	if binary == "" {
		t.Skip("requires local yt-dlp binary")
	}
	data := strings.Repeat("0123456789abcdef", 16384)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write([]byte(data)) }))
	defer server.Close()
	dir := t.TempDir()
	fixture := filepath.Join(dir, "info.json")
	info := map[string]any{"id": "offline", "title": "offline", "extractor": "generic", "webpage_url": "https://example.com/video", "formats": []map[string]any{
		{"format_id": "v", "url": server.URL + "/video", "ext": "mp4", "vcodec": "avc1", "acodec": "none", "height": 240},
		{"format_id": "a", "url": server.URL + "/audio", "ext": "m4a", "vcodec": "none", "acodec": "mp4a"},
	}}
	raw, _ := json.Marshal(info)
	if err := os.WriteFile(fixture, raw, 0600); err != nil {
		t.Fatal(err)
	}
	a := &api{cfg: config{YtDLPPath: binary}}
	a.beginDownload(testDownloadID)
	a.selectDownloadStreams(testDownloadID, sourceInfo{Formats: []sourceFormat{{ID: "v", Height: 240, VideoCodec: "avc1", AudioCodec: "none"}, {ID: "a", VideoCodec: "none", AudioCodec: "mp4a"}}}, "v+a")
	for _, format := range []string{"v", "a"} {
		args := append(downloadProgressArgs(), "--ignore-config", "--no-warnings", "--no-part", "--fixup", "never", "--load-info-json", fixture, "--format", format, "--output", filepath.Join(dir, "source.%(format_id)s.%(ext)s"))
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		stderr, err := a.runDownload(ctx, testDownloadID, args...)
		cancel()
		if err != nil {
			t.Fatalf("native downloader: %v: %s", err, stderr)
		}
	}
	p := progressSnapshot(t, a)
	if p.Stage != "processing" {
		t.Fatalf("native template did not update progress: %+v", p)
	}
	for _, s := range p.Streams {
		if s.Status != "finished" || s.DownloadedBytes != int64(len(data)) {
			t.Fatalf("bad native stream: %+v", s)
		}
	}
}
