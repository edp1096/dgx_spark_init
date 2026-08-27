package server

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"io"
	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestVideoJobStreamsEngineOutput(t *testing.T) {
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/videos/generations" {
			http.NotFound(w, r)
			return
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		if r.FormValue("prompt") != "waves under moonlight" || r.FormValue("width") != "768" || r.FormValue("height") != "512" || r.FormValue("num_frames") != "121" || r.FormValue("fps") != "24" || r.FormValue("seed") != "42" {
			t.Fatalf("unexpected fields: %#v", r.MultipartForm.Value)
		}
		if r.FormValue("frame_indices") != "[0,60,120]" || r.FormValue("image_strengths") != "[0.8,0.7,0.9]" || len(r.MultipartForm.File["images"]) != 3 {
			t.Fatalf("unexpected conditioning: values=%#v files=%#v", r.MultipartForm.Value, r.MultipartForm.File)
		}
		w.Header().Set("Content-Type", "video/mp4")
		_, _ = w.Write([]byte("fake mp4"))
	}))
	defer worker.Close()

	cfg := config.Config{
		DataDir: t.TempDir(),
		Engines: map[string]config.Engine{"video": {Endpoint: worker.URL}},
		Video:   config.Video{Model: "test-video", DefaultWidth: 768, DefaultHeight: 512, DefaultFrames: 121, DefaultFPS: 24},
	}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()

	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("prompt", "waves under moonlight")
	_ = form.WriteField("seed", "42")
	_ = form.WriteField("image_strength", "0.8")
	_ = form.WriteField("end_image_strength", "0.9")
	_ = form.WriteField("keyframe_count", "1")
	_ = form.WriteField("keyframe_time_0", "2.5")
	_ = form.WriteField("keyframe_strength_0", "0.7")
	for _, field := range []string{"start_image", "keyframe_image_0", "end_image"} {
		part, err := form.CreateFormFile(field, field+".png")
		if err != nil {
			t.Fatal(err)
		}
		_, _ = part.Write([]byte("fake image"))
	}
	_ = form.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/video", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		list := store.List()
		if len(list) == 1 && list[0].Status == "completed" {
			got, err := os.ReadFile(store.OutputPath(list[0].ID + ".mp4"))
			if err != nil || string(got) != "fake mp4" {
				t.Fatalf("output=%q err=%v", got, err)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("job did not complete: %#v", store.List())
}

func TestCustomVoiceUsesOpenAICompatibleRequest(t *testing.T) {
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		if r.URL.Path != "/v1/audio/speech" {
			http.NotFound(w, r)
			return
		}
		var request map[string]any
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		if request["task_type"] != "CustomVoice" || request["voice"] != "sohee" || request["model"] != "test-custom" || request["instructions"] != "Speak warmly and slowly." || request["seed"] != float64(4242) {
			t.Fatalf("unexpected request %#v", request)
		}
		w.Header().Set("Content-Type", "audio/wav")
		_, _ = w.Write([]byte("fake wav"))
	}))
	defer worker.Close()

	cfg := config.Config{
		DataDir: t.TempDir(),
		Engines: map[string]config.Engine{"speech": {Endpoint: worker.URL}},
		Speech:  config.Speech{CustomVoiceModel: "test-custom", DefaultLanguage: "Korean", DefaultSpeaker: "Sohee"},
	}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()

	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("text", "generated words")
	_ = form.WriteField("language", "Korean")
	_ = form.WriteField("speaker", "Sohee")
	_ = form.WriteField("instructions", "Speak warmly and slowly.")
	_ = form.WriteField("seed", "4242")
	_ = form.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/speech", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		list := store.List()
		if len(list) == 1 && list[0].Status == "completed" {
			got, err := os.ReadFile(store.OutputPath(list[0].ID + ".wav"))
			if err != nil || string(got) != "fake wav" {
				t.Fatalf("output=%q err=%v", got, err)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("job did not complete: %#v", store.List())
}

func TestRecognitionUsesOpenAICompatibleRequest(t *testing.T) {
	asrWorker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/audio/transcriptions" {
			http.NotFound(w, r)
			return
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		if r.FormValue("model") != "test-asr" || r.FormValue("language") != "Korean" || r.FormValue("prompt") != "SparkTalk" {
			t.Fatalf("unexpected fields: %#v", r.MultipartForm.Value)
		}
		file, _, err := r.FormFile("file")
		if err != nil {
			t.Fatal(err)
		}
		data, _ := io.ReadAll(file)
		_ = file.Close()
		if string(data) != "fake audio" {
			t.Fatalf("unexpected audio %q", data)
		}
		_ = json.NewEncoder(w).Encode(map[string]string{"text": "인식 결과", "language": "Korean"})
	}))
	defer asrWorker.Close()
	mediaWorker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/media/prepare" {
			http.NotFound(w, r)
			return
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Fatal(err)
		}
		archivePath := filepath.Join(t.TempDir(), "prepared.zip")
		archiveFile, err := os.Create(archivePath)
		if err != nil {
			t.Fatal(err)
		}
		archive := zip.NewWriter(archiveFile)
		manifest, _ := archive.Create("manifest.json")
		_, _ = manifest.Write([]byte(`{"source_name":"sample.mp4","asset":{"id":"0123456789abcdef0123456789abcdef","filename":"video.mp4","media_type":"video","content_type":"video/mp4","size":1024,"duration":1,"width":640,"height":360},"segments":[{"name":"segment-00000.wav","start":0,"end":1,"duration":1}]}`))
		segment, _ := archive.Create("segment-00000.wav")
		_, _ = segment.Write([]byte("fake audio"))
		_ = archive.Close()
		_ = archiveFile.Close()
		w.Header().Set("Content-Type", "application/zip")
		http.ServeFile(w, r, archivePath)
	}))
	defer mediaWorker.Close()

	cfg := config.Config{
		DataDir:     t.TempDir(),
		Engines:     map[string]config.Engine{"recognition": {Endpoint: asrWorker.URL}, "media": {Endpoint: mediaWorker.URL}},
		Recognition: config.Recognition{Model: "test-asr", DefaultLanguage: "Auto", MaxUploadMB: 1, SegmentSeconds: 30, DefaultOutputFormats: []string{"txt"}, DefaultTranslationMode: "none"},
	}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()

	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	_ = form.WriteField("language", "Korean")
	_ = form.WriteField("context", "SparkTalk")
	part, _ := form.CreateFormFile("audio", "sample.wav")
	_, _ = part.Write([]byte("fake audio"))
	_ = form.Close()
	req := httptest.NewRequest(http.MethodPost, "/api/jobs/recognition", &body)
	req.Header.Set("Content-Type", form.FormDataContentType())
	res := httptest.NewRecorder()
	handler.ServeHTTP(res, req)
	if res.Code != http.StatusAccepted {
		t.Fatalf("status=%d body=%s", res.Code, res.Body.String())
	}

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		list := store.List()
		if len(list) == 1 && list[0].Status == "completed" {
			if list[0].Params["text"] != "인식 결과" || list[0].Params["detected_language"] != "Korean" {
				t.Fatalf("unexpected result %#v", list[0])
			}
			if list[0].MediaAssetID != "0123456789abcdef0123456789abcdef" || list[0].MediaURL == "" || list[0].CaptionURL == "" {
				t.Fatalf("missing media result %#v", list[0])
			}
			media, ok := list[0].Params["media"].(map[string]any)
			if !ok || media["media_type"] != "video" || media["content_type"] != "video/mp4" {
				t.Fatalf("missing media metadata %#v", list[0].Params["media"])
			}
			got, err := os.ReadFile(store.OutputPath(list[0].ID + ".txt"))
			if err != nil || string(got) != "인식 결과\n" {
				t.Fatalf("output=%q err=%v", got, err)
			}
			caption, err := os.ReadFile(store.OutputPath(list[0].ID + ".player.vtt"))
			if err != nil || !bytes.HasPrefix(caption, []byte("WEBVTT\n")) {
				t.Fatalf("caption=%q err=%v", caption, err)
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("job did not complete: %#v", store.List())
}
