package main

import (
	"bytes"
	"context"
	"errors"
	"mime/multipart"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunRejectsTruncatedOutput(t *testing.T) {
	stdout, stderr, err := run(context.Background(), "sh", "-c", "echo warning >&2; head -c 8388609 /dev/zero")
	if len(stdout) != 8*1024*1024 || !strings.Contains(string(stderr), "warning") {
		t.Fatalf("unexpected capture: stdout %d, stderr %q", len(stdout), stderr)
	}
	var apiErr *httpError
	if !errors.As(processError("test", err, stderr), &apiErr) || !strings.Contains(apiErr.Message, "output was truncated") {
		t.Fatalf("overflow hidden by warning: %v", err)
	}
}

func TestLimitedBufferExactLimitIsNotTruncated(t *testing.T) {
	b := &limitedBuffer{Limit: 3}
	_, _ = b.Write([]byte("abc"))
	if b.Truncated || string(b.Bytes()) != "abc" {
		t.Fatalf("exact limit treated as overflow: %+v", b)
	}
	_, _ = b.Write([]byte("d"))
	if !b.Truncated || string(b.Bytes()) != "abc" {
		t.Fatalf("overflow not detected: %+v", b)
	}
}

func TestSafeExtension(t *testing.T) {
	tests := map[string]string{
		"movie.MP4":        ".mp4",
		"voice.wav":        ".wav",
		"no-extension":     ".media",
		"unsafe.foo-bar":   ".media",
		"long.abcdefghijk": ".media",
	}
	for input, want := range tests {
		if got := safeExtension(input); got != want {
			t.Errorf("safeExtension(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestSaveRawUpload(t *testing.T) {
	dir := t.TempDir()
	req := httptest.NewRequest("POST", "/v1/probe", bytes.NewBufferString("media-data"))
	req.Header.Set("Content-Type", "video/mp4")
	recorder := httptest.NewRecorder()
	path, err := saveUpload(recorder, req, dir, 1024)
	if err != nil {
		t.Fatal(err)
	}
	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "media-data" || filepath.Ext(path) != ".mp4" {
		t.Fatalf("unexpected saved upload: path=%s content=%q", path, content)
	}
}

func TestSaveMultipartUpload(t *testing.T) {
	dir := t.TempDir()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("file", "clip.webm")
	if err != nil {
		t.Fatal(err)
	}
	_, _ = part.Write([]byte("webm-data"))
	_ = writer.Close()
	req := httptest.NewRequest("POST", "/v1/probe", &body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	path, err := saveUpload(httptest.NewRecorder(), req, dir, 1024)
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Ext(path) != ".webm" {
		t.Fatalf("unexpected extension: %s", path)
	}
}

func TestSaveUploadLimit(t *testing.T) {
	req := httptest.NewRequest("POST", "/v1/probe", bytes.NewBufferString("too-large"))
	_, err := saveUpload(httptest.NewRecorder(), req, t.TempDir(), 3)
	if err == nil {
		t.Fatal("expected upload limit error")
	}
}
