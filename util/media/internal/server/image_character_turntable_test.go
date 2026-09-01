package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCharacterTurntableProxiesEightReviewFrames(t *testing.T) {
	frame := base64.StdEncoding.EncodeToString(bytes.Repeat([]byte{1}, 256))
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/character/turntable" {
			t.Fatalf("path=%q", r.URL.Path)
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil || r.FormValue("operation_id") != "character-turntable-test" {
			t.Fatalf("invalid turntable form: %v", err)
		}
		frames := make([]map[string]any, 8)
		for index := range frames {
			frames[index] = map[string]any{"direction": "front", "frame_index": index, "mime_type": "image/jpeg", "data": frame}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"operation_id": "character-turntable-test", "seed": 7, "frames": frames})
	}))
	defer engine.Close()
	dataDir := t.TempDir()
	store, _ := jobs.New(dataDir)
	cfg := config.Config{DataDir: dataDir, Engines: map[string]config.Engine{"character": {Endpoint: engine.URL}}}
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	part, _ := form.CreateFormFile("image", "source.png")
	_, _ = part.Write(bytes.Repeat([]byte{2}, 256))
	_ = form.WriteField("operation_id", "character-turntable-test")
	_ = form.Close()
	request := httptest.NewRequest(http.MethodPost, "/api/images/character-turntable", &body)
	request.Header.Set("Content-Type", form.FormDataContentType())
	response := httptest.NewRecorder()
	New(cfg, store, nil).Handler().ServeHTTP(response, request)
	if response.Code != http.StatusOK || !bytes.Contains(response.Body.Bytes(), []byte(`"frames"`)) {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
}
