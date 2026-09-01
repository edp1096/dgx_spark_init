package server

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
)

func TestSanitizeTransparentRGBRemovesHiddenSourcePixels(t *testing.T) {
	source := image.NewNRGBA(image.Rect(0, 0, 2, 1))
	source.SetNRGBA(0, 0, color.NRGBA{R: 210, G: 80, B: 35, A: 0})
	source.SetNRGBA(1, 0, color.NRGBA{R: 50, G: 90, B: 130, A: 255})
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, source); err != nil {
		t.Fatal(err)
	}

	cleaned, err := sanitizeTransparentRGB(encoded.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	decoded, _, err := image.Decode(bytes.NewReader(cleaned))
	if err != nil {
		t.Fatal(err)
	}
	hidden := color.NRGBAModel.Convert(decoded.At(0, 0)).(color.NRGBA)
	visible := color.NRGBAModel.Convert(decoded.At(1, 0)).(color.NRGBA)
	if hidden != (color.NRGBA{}) {
		t.Fatalf("transparent pixel retained hidden RGB: %#v", hidden)
	}
	if visible != (color.NRGBA{R: 50, G: 90, B: 130, A: 255}) {
		t.Fatalf("visible garment pixel changed: %#v", visible)
	}
}

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
			"cutout_b64":    base64.StdEncoding.EncodeToString(pngData),
			"mask_b64":      base64.StdEncoding.EncodeToString(pngData),
			"reference_b64": base64.StdEncoding.EncodeToString(pngData),
			"candidates":    []map[string]any{{"index": 0, "score": .1}, {"index": 1, "score": .2}},
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
			if intParam(job.Params, "selected_source_index", -1) != 1 || job.Outputs["mask"] == "" || job.Outputs["reference"] == "" {
				t.Fatalf("missing extraction metadata: %#v", job)
			}
			if _, err := os.Stat(store.OutputPath(job.ID + ".png")); err != nil {
				t.Fatal(err)
			}
			if _, err := os.Stat(store.OutputPath(job.ID + "-mask.png")); err != nil {
				t.Fatal(err)
			}
			if _, err := os.Stat(store.OutputPath(job.ID + "-reference.png")); err != nil {
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
