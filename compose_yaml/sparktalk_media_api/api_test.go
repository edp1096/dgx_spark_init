package main

import (
	"bytes"
	"mime/multipart"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

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
