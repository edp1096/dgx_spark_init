package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
)

func TestFaceSwapPersistsBothInputsAndResult(t *testing.T) {
	result := testPNG(t, 40, 30)
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/faces/swap" {
			http.NotFound(w, r)
			return
		}
		if err := r.ParseMultipartForm(4 << 20); err != nil {
			t.Fatal(err)
		}
		if len(r.MultipartForm.File["target"]) != 1 || len(r.MultipartForm.File["source"]) != 1 {
			t.Fatalf("missing face swap files: %#v", r.MultipartForm.File)
		}
		if r.FormValue("target_face_index") != "1" || r.FormValue("source_face_index") != "2" {
			t.Fatalf("unexpected face indexes target=%q source=%q", r.FormValue("target_face_index"), r.FormValue("source_face_index"))
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"data":  []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString(result)}},
			"model": "inswapper_128.onnx", "target_face_index": 1, "source_face_index": 2,
		})
	}))
	defer worker.Close()

	dataDir := t.TempDir()
	store, err := jobs.New(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(config.Config{DataDir: dataDir, Engines: map[string]config.Engine{"faceswap": {Endpoint: worker.URL}}}, store, nil).Handler()
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("target_face_index", "1")
	_ = form.WriteField("source_face_index", "2")
	for _, field := range []string{"target", "source"} {
		part, partErr := form.CreateFormFile(field, field+".png")
		if partErr != nil {
			t.Fatal(partErr)
		}
		_, _ = part.Write(testPNG(t, 32, 24))
	}
	_ = form.Close()
	request := httptest.NewRequest(http.MethodPost, "/api/jobs/face-swap", &body)
	request.Header.Set("Content-Type", form.FormDataContentType())
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		list := store.List()
		if len(list) == 1 && list[0].Status == "completed" {
			job := list[0]
			if job.Params["model"] != "inswapper_128.onnx" || intParam(job.Params, "target_face_index", -1) != 1 {
				t.Fatalf("unexpected face swap metadata: %#v", job.Params)
			}
			if _, err := os.Stat(store.OutputPath(job.ID + ".png")); err != nil {
				t.Fatal(err)
			}
			inputs := httptest.NewRecorder()
			handler.ServeHTTP(inputs, httptest.NewRequest(http.MethodGet, "/api/jobs/"+job.ID+"/inputs", nil))
			if !bytes.Contains(inputs.Body.Bytes(), []byte("face_swap_target")) || !bytes.Contains(inputs.Body.Bytes(), []byte("face_swap_source")) {
				t.Fatalf("face swap inputs are not reusable: %s", inputs.Body.String())
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("face swap did not complete: %#v", store.List())
}
