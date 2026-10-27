package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"sparktalk/internal/config"
	"sparktalk/internal/orchestrator"
)

func TestShortsCoverageRejectsMissingNarration(t *testing.T) {
	if got := speechCoverage("안녕하세요. 좋은 하루 보내세요.", "안녕하세요 좋은하루 보내세요"); got != 0 {
		t.Fatal(got)
	}
	if got := speechCoverage("첫 번째 내용을 설명합니다. 두 번째 내용도 설명합니다.", "첫 번째 내용을 설명합니다."); got <= .25 {
		t.Fatal("missing sentence passed", got)
	}
}

func TestShortsFilesRequireOwningConversationAndKeepVideoRanges(t *testing.T) {
	job := "abcdef0123456789abcdef0123456789"
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			io.WriteString(w, `{"status":"ok"}`)
		case "/v1/projects/" + job:
			io.WriteString(w, `{"session_id":"session"}`)
		case "/v1/projects/" + job + "/files/final.mp4":
			if r.Header.Get("Range") != "bytes=0-3" {
				t.Error("range not forwarded")
			}
			w.Header().Set("Content-Range", "bytes 0-3/100")
			w.Header().Set("Content-Type", "video/mp4")
			w.WriteHeader(206)
			io.WriteString(w, "test")
		default:
			http.NotFound(w, r)
		}
	}))
	defer worker.Close()
	s, _ := testImageServer(t)
	if _, err := s.db.CreateSession("other", "other", "model", "none"); err != nil {
		t.Fatal(err)
	}
	cat, _ := orchestrator.LoadCatalog()
	for i := range cat.Components {
		if cat.Components[i].ID == "extra-shorts" {
			cat.Components[i].Endpoint = worker.URL
			cat.Components[i].HealthURL = worker.URL + "/health"
			cat.Components[i].Controller = "external"
			cat.Components[i].AutoAddress = false
		}
	}
	runtime, err := orchestrator.NewControllerWithCatalog(cat)
	if err != nil {
		t.Fatal(err)
	}
	defer runtime.Close()
	s.runtime = runtime
	s.cfg.Runtime.Mode = "managed"
	s.cfg.Runtime.ActiveBundle = "qwen38fn_exl3"
	s.cfg.Runtime.Catalog = &cat
	for _, tc := range []struct {
		session string
		code    int
	}{{"other", 404}, {"session", 206}} {
		req := httptest.NewRequest("GET", "/api/shorts/"+job+"/files/final.mp4?session_id="+tc.session, nil)
		req.Header.Set("Range", "bytes=0-3")
		response := httptest.NewRecorder()
		s.shortsAPI(response, req)
		if response.Code != tc.code {
			t.Fatalf("%s: %d %s", tc.session, response.Code, response.Body)
		}
		if response.Code == 206 && response.Header().Get("Content-Range") == "" {
			t.Fatal("range response missing")
		}
	}
}

func TestShortsAssetDoesNotReadImagesFromAnotherConversation(t *testing.T) {
	s, attachment := testImageServer(t)
	_, err := s.shortsAsset(context.Background(), shortsRequest{SessionID: "other", Scene: &shortsScene{Visual: "image", AttachmentID: attachment.ID}})
	if err == nil || !strings.Contains(err.Error(), "이미지") {
		t.Fatal("foreign image accepted", err)
	}
}

func TestShortsPublicationIsIdempotentAndRequiresCompletedOutput(t *testing.T) {
	job := "abcdef0123456789abcdef0123456789"
	ready := false
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/final.mp4") {
			w.Header().Set("Content-Type", "video/mp4")
			_, _ = w.Write([]byte("\x00\x00\x00\x18ftypisom\x00\x00\x00\x00isommp42"))
			return
		}
		status := "running"
		if ready {
			status = "completed"
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"session_id": "session", "title": "시험 영상", "status": status, "output_ready": ready})
	}))
	defer worker.Close()
	s, _ := testImageServer(t)
	cat, _ := orchestrator.LoadCatalog()
	for i := range cat.Components {
		if cat.Components[i].ID == "extra-shorts" {
			cat.Components[i].Endpoint = worker.URL
		}
	}
	s.cfg.Runtime.Catalog = &cat
	a := shortsRequest{SessionID: "session", JobID: job}
	if _, err := s.shortsPublish(context.Background(), a); err == nil {
		t.Fatal("unfinished output published")
	}
	ready = true
	before, _ := s.db.Messages("session")
	for i := 0; i < 2; i++ {
		if _, err := s.shortsPublish(context.Background(), a); err != nil {
			t.Fatal(err)
		}
	}
	after, _ := s.db.Messages("session")
	if len(after) != len(before)+1 {
		t.Fatalf("duplicate delivery: %d -> %d", len(before), len(after))
	}
}

func TestShortsImageUsesConfiguredBackendSize(t *testing.T) {
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request map[string]any
		_ = json.NewDecoder(r.Body).Decode(&request)
		if request["size"] != "1024x1024" {
			http.Error(w, "joint backend requires 1024x1024", 400)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"seed": 42, "data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString(onePixelPNG)}}})
	}))
	defer worker.Close()
	s, _ := testImageServer(t)
	s.cfg.Image = config.ImageConfig{Endpoint: worker.URL, Model: "joint-qwim", Mode: "qwen-image21", DefaultSize: "1024x1024", Timeout: "2s"}
	if _, err := s.shortsAsset(context.Background(), shortsRequest{SessionID: "session", Scene: &shortsScene{Visual: "image", ImagePrompt: "A wooden desk with books"}}); err != nil {
		t.Fatal(err)
	}
}
