package server

import (
	"bytes"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
)

func TestFetchRemoteImageReturnsValidatedImage(t *testing.T) {
	var source bytes.Buffer
	canvas := image.NewRGBA(image.Rect(0, 0, 8, 6))
	canvas.Set(2, 2, color.RGBA{R: 255, A: 255})
	if err := png.Encode(&source, canvas); err != nil {
		t.Fatal(err)
	}
	remote := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.Header.Get("User-Agent"), "SparkMedia") {
			t.Errorf("user agent=%q", r.Header.Get("User-Agent"))
		}
		w.Header().Set("Content-Type", "application/octet-stream")
		_, _ = w.Write(source.Bytes())
	}))
	defer remote.Close()

	dataDir := t.TempDir()
	store, err := jobs.New(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(config.Config{DataDir: dataDir}, store, nil).Handler()
	payload, _ := json.Marshal(remoteImageRequest{URL: remote.URL + "/sample.png?size=large"})
	request := httptest.NewRequest(http.MethodPost, "/api/images/fetch", bytes.NewReader(payload))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	if response.Header().Get("Content-Type") != "image/png" || response.Header().Get("X-Image-Filename") != "sample.png" {
		t.Fatalf("headers=%v", response.Header())
	}
	if !bytes.Equal(response.Body.Bytes(), source.Bytes()) {
		t.Fatal("downloaded image changed")
	}
}

func TestFetchRemoteImageRejectsNonImageAndUnsupportedScheme(t *testing.T) {
	remote := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write([]byte("<html>not an image</html>"))
	}))
	defer remote.Close()
	dataDir := t.TempDir()
	store, _ := jobs.New(dataDir)
	handler := New(config.Config{DataDir: dataDir}, store, nil).Handler()
	for _, test := range []struct {
		url  string
		code int
	}{
		{url: "file:///etc/passwd", code: http.StatusBadRequest},
		{url: remote.URL + "/page", code: http.StatusUnsupportedMediaType},
	} {
		payload, _ := json.Marshal(remoteImageRequest{URL: test.url})
		request := httptest.NewRequest(http.MethodPost, "/api/images/fetch", bytes.NewReader(payload))
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != test.code {
			t.Fatalf("url=%s status=%d body=%s", test.url, response.Code, response.Body.String())
		}
	}
}
