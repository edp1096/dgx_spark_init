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

func TestGarmentExtractionPersistsCutoutMaskAndInputs(t *testing.T) {
	pngData := testPNG(t, 32, 24)
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/garments/extract" {
			http.NotFound(w, r)
			return
		}
		if err := r.ParseMultipartForm(2 << 20); err != nil {
			t.Fatal(err)
		}
		if r.FormValue("target") != "upper,lower,shoes" || len(r.MultipartForm.File["images"]) != 2 {
			t.Fatalf("unexpected garment request target=%q images=%d", r.FormValue("target"), len(r.MultipartForm.File["images"]))
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": "test-parser", "target": "upper,lower,shoes", "selected_index": 1,
			"width": 32, "height": 24, "coverage": .25,
			"cutout_b64": base64.StdEncoding.EncodeToString(pngData),
			"mask_b64":   base64.StdEncoding.EncodeToString(pngData),
			"candidates": []map[string]any{{"index": 0, "score": .1}, {"index": 1, "score": .2}},
		})
	}))
	defer worker.Close()

	dataDir := t.TempDir()
	store, err := jobs.New(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(config.Config{DataDir: dataDir, Engines: map[string]config.Engine{"garment": {Endpoint: worker.URL}}}, store, nil).Handler()
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("target", "shoes,upper,lower")
	for index, field := range []string{"source", "references"} {
		part, partErr := form.CreateFormFile(field, "person.png")
		if partErr != nil {
			t.Fatal(partErr)
		}
		if _, partErr = part.Write(pngData); partErr != nil {
			t.Fatal(partErr)
		}
		_ = index
	}
	_ = form.Close()
	request := httptest.NewRequest(http.MethodPost, "/api/jobs/garment-extract", &body)
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
			if imageIntParam(job.Params, "selected_source_index", -1) != 1 || job.Outputs["mask"] == "" {
				t.Fatalf("missing extraction metadata: %#v", job)
			}
			if _, err := os.Stat(store.OutputPath(job.ID + ".png")); err != nil {
				t.Fatal(err)
			}
			if _, err := os.Stat(store.OutputPath(job.ID + "-mask.png")); err != nil {
				t.Fatal(err)
			}
			inputsResponse := httptest.NewRecorder()
			handler.ServeHTTP(inputsResponse, httptest.NewRequest(http.MethodGet, "/api/jobs/"+job.ID+"/inputs", nil))
			if !bytes.Contains(inputsResponse.Body.Bytes(), []byte("garment_source")) || !bytes.Contains(inputsResponse.Body.Bytes(), []byte("garment_reference")) {
				t.Fatalf("missing garment inputs: %s", inputsResponse.Body.String())
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("garment extraction did not complete: %#v", store.List())
}
