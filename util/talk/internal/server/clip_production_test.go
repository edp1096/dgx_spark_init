package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/orchestrator"
)

func TestClipCreationSnapshotsActualBranchImageAndStableRequestKey(t *testing.T) {
	var requests []map[string]any
	id := strings.Repeat("a", 32)
	s, _ := shortsToolTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/clips" && r.Method == "POST" {
			var value map[string]any
			_ = json.NewDecoder(r.Body).Decode(&value)
			requests = append(requests, value)
			_ = json.NewEncoder(w).Encode(clipJob{ID: id, SessionID: "session", Status: "queued"})
			return
		}
		w.Write([]byte(`{"status":"ok","busy":false}`))
	})
	history, _ := s.db.Messages("session")
	source := history[0].Attachments[0]
	args := clipCreate{SourceImageID: source.ID, Brief: "만화로 바꾸고 손 흔들게 해줘"}
	ctx := withRequestAttachments(context.Background(), "session", history)
	for i := 0; i < 2; i++ {
		job, err := s.createClip(ctx, "session", args)
		if err != nil || job.ID != id {
			t.Fatal(job, err)
		}
	}
	want := "data:image/png;base64," + base64.StdEncoding.EncodeToString(onePixelPNG)
	if requests[0]["source_image"] != want || requests[0]["request_key"] != requests[1]["request_key"] {
		t.Fatal("source bytes or idempotency key changed")
	}
	excluded := withRequestAttachments(context.Background(), "session", nil)
	if _, err := s.createClip(excluded, "session", args); err == nil || len(requests) != 2 {
		t.Fatal("nonselected branch reached the production worker")
	}
	generated, err := s.media.SaveReader(bytes.NewReader(onePixelPNG), "fresh.png", "image/png", 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	turn := context.WithValue(excluded, turnImageKey{}, &turnImages{sessionID: "session", items: map[string]db.Attachment{generated.ID: generated}})
	args.SourceImageID = generated.ID
	if _, err = s.createClip(turn, "session", args); err != nil {
		t.Fatal("same-turn edited image could not enter video production", err)
	}
	if requests[2]["source_image_id"] != generated.ID || requests[2]["source_image"] != want {
		t.Fatal("generated image lineage lost")
	}
}

func TestClipImageBridgeUsesNativeReferenceAndRejectsSwap(t *testing.T) {
	s, _ := testImageServer(t)
	var calls atomic.Int32
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var payload map[string]any
		_ = json.NewDecoder(r.Body).Decode(&payload)
		if payload["source_image"] == nil || payload["operation"] != "inpaint" || payload["mask_box"] == nil {
			t.Errorf("native edit fields lost: %+v", payload)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"seed": 42, "data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString(onePixelPNG)}}})
	}))
	defer backend.Close()
	s.cfg.Image = config.ImageConfig{Endpoint: backend.URL, Model: "qwen-image-2.1-uc-nvfp4", Mode: "reference", Timeout: "2s"}
	a := clipCallback{Operation: "inpaint", Prompt: "Separate fingers inside the mask", SourceImage: "data:image/png;base64," + base64.StdEncoding.EncodeToString(onePixelPNG), MaskBox: []int{0, 0, 1, 1}, Seed: 42}
	if _, err := s.clipProductionImage(context.Background(), a); err != nil {
		t.Fatal(err)
	}
	a.Operation = "head_swap"
	if _, err := s.clipProductionImage(context.Background(), a); err == nil || calls.Load() != 1 {
		t.Fatal("deferred face-swap operation was accepted")
	}
	a.Operation, a.SourceImage = "identity_edit", "https://example.com/untrusted.png"
	if _, err := s.clipProductionImage(context.Background(), a); err == nil || calls.Load() != 1 {
		t.Fatal("arbitrary source URL reached backend")
	}
}

func TestJointRuntimeStandaloneReferenceEditSendsExplicitOperation(t *testing.T) {
	s, source := testImageServer(t)
	var operation string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			w.Write([]byte(`{"events":[]}`))
			return
		}
		var payload map[string]any
		json.NewDecoder(r.Body).Decode(&payload)
		operation, _ = payload["operation"].(string)
		if payload["source_image"] == nil {
			t.Error("source image lost")
		}
		json.NewEncoder(w).Encode(map[string]any{"seed": 42, "data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString(onePixelPNG)}}})
	}))
	defer backend.Close()
	cat, _ := orchestrator.LoadCatalog()
	for i := range cat.Components {
		if cat.Components[i].ID == "qwim-mmh3" {
			cat.Components[i].Endpoint = backend.URL
			cat.Components[i].HealthURL = backend.URL + "/health"
		}
	}
	var err error
	cat, err = orchestrator.ValidateCatalog(cat)
	if err != nil {
		t.Fatal(err)
	}
	s.cfg.Runtime = config.RuntimeConfig{Mode: "external", Bundle: "qwen38fn_exl3", Catalog: &cat}
	if endpoint, _ := videoRuntime(s.cfg); endpoint != backend.URL {
		t.Fatal("test backend not isolated")
	}
	cfg := config.ImageConfig{Endpoint: backend.URL, Model: "qwen-image-2.1-uc-nvfp4", Mode: "reference", DefaultSize: "512x512", Timeout: "2s"}
	args, _ := json.Marshal(map[string]string{"operation": "identity_edit", "source_image_id": source.ID, "prompt": "Restyle as a cartoon"})
	_, err = s.executeImageGenerateTool(context.Background(), "session", cfg, llm.ToolCall{ID: "standalone-edit", Function: llm.FunctionCall{Arguments: string(args)}}, func(string, any) error { return nil })
	if err != nil || operation != "identity_edit" {
		t.Fatal("native edit would be rejected as text generation", operation, err)
	}
}

func TestClipVideoBridgeRejectsTextFallbackAndDoesNotPublishIntermediates(t *testing.T) {
	s, _ := testImageServer(t)
	var confirmed atomic.Bool
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]any
		_ = json.NewDecoder(r.Body).Decode(&payload)
		if payload["first_frame"] != "data:image/png;base64,"+base64.StdEncoding.EncodeToString(onePixelPNG) {
			t.Error("accepted image bytes were not sent to H3")
		}
		w.Header().Set("Content-Type", "video/mp4")
		w.Header().Set("X-Video-Width", "512")
		w.Header().Set("X-Video-Height", "768")
		if confirmed.Load() {
			w.Header().Set("X-Video-Input-Mode", "i2v")
		}
		w.Write(importTestVideo)
	}))
	defer backend.Close()
	cat, _ := orchestrator.LoadCatalog()
	for i := range cat.Components {
		if cat.Components[i].ID == "qwim-mmh3" {
			cat.Components[i].Endpoint = backend.URL
			cat.Components[i].HealthURL = backend.URL + "/health"
		}
	}
	var err error
	cat, err = orchestrator.ValidateCatalog(cat)
	if err != nil {
		t.Fatal(err)
	}
	s.cfg.Runtime = config.RuntimeConfig{Mode: "external", Bundle: "qwen38fn_exl3", Catalog: &cat}
	if endpoint, ok := videoRuntime(s.cfg); !ok || endpoint != backend.URL {
		t.Fatal("test video endpoint was not isolated", endpoint)
	}
	a := clipCallback{FirstFrame: "data:image/png;base64," + base64.StdEncoding.EncodeToString(onePixelPNG), Prompt: "Wave the hand", Resolution: "480p"}
	if _, err := s.clipProductionVideo(context.Background(), a); err == nil {
		t.Fatal("text-only fallback was accepted")
	}
	confirmed.Store(true)
	before, _ := s.db.Messages("session")
	if _, err := s.clipProductionVideo(context.Background(), a); err != nil {
		t.Fatal(err)
	}
	after, _ := s.db.Messages("session")
	if len(after) != len(before) {
		t.Fatal("unreviewed video was attached")
	}
}

func TestClipPublicationRequiresPassedCompletionAndDeduplicates(t *testing.T) {
	var downloads atomic.Int32
	id := strings.Repeat("b", 32)
	s, _ := shortsToolTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/files/final.mp4") {
			downloads.Add(1)
			w.Header().Set("Content-Type", "video/mp4")
			w.Write(importTestVideo)
			return
		}
		w.Write([]byte(`{"status":"ok"}`))
	})
	job := clipJob{ID: id, SessionID: "session", Status: "failed", Report: json.RawMessage(`{"passed":false}`)}
	if _, err := s.publishClip(context.Background(), job); err == nil || downloads.Load() != 0 {
		t.Fatal("failed production was published")
	}
	job.Status, job.OutputReady, job.Report = "completed", true, json.RawMessage(`{"passed":true,"plan":{"title":"만화 영상"}}`)
	first, err := s.publishClip(context.Background(), job)
	if err != nil {
		t.Fatal(err)
	}
	second, err := s.publishClip(context.Background(), job)
	if err != nil || downloads.Load() != 1 {
		t.Fatal("duplicate publication", downloads.Load(), err)
	}
	a, _ := json.Marshal(first)
	b, _ := json.Marshal(second)
	if !bytes.Equal(a, b) {
		t.Fatal("publication receipt changed")
	}
}

func TestClipLookupRejectsOtherConversation(t *testing.T) {
	id := strings.Repeat("c", 32)
	s, _ := shortsToolTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(clipJob{ID: id, SessionID: "another-session", Status: "running"})
	})
	if _, err := s.conversationClip(context.Background(), "session", id); err == nil {
		t.Fatal("another conversation's production was accessible")
	}
}

func TestCanceledClipCannotResumeFromTheSameInstructionOrAStatusQuestion(t *testing.T) {
	s, _ := testImageServer(t)
	var calls atomic.Int32
	model := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var body struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		text := body.Messages[len(body.Messages)-1].Content
		decision := `{"resume":false,"evidence":""}`
		if strings.Contains(text, "다시 이어서 해줘") {
			decision = `{"resume":true,"evidence":"다시 이어서 해줘"}`
		}
		if strings.Contains(text, "잘못된 근거") {
			decision = `{"resume":true,"evidence":"사용자가 실제로 하지 않은 말"}`
		}
		chunk, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]string{"content": decision}}}})
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte("data: " + string(chunk) + "\n\ndata: [DONE]\n\n"))
	}))
	defer model.Close()
	s.llm = llm.New(model.URL, "test-model", "", "qwen3.8")
	s.cfg.Model.DefaultModel = "test-model"
	turn := &activeTurn{session: "session", anchor: 1, base: "이 사진으로 영상 만들어줘"}
	ctx := context.WithValue(context.Background(), activeTurnKey{}, turn)
	job := clipJob{SessionID: "session", Status: "canceled"}
	job.Project.CancelTurnKey, _ = s.clipTurnIntent(ctx, "session")
	if allowed, err := s.clipResumeRequested(ctx, job); err != nil || allowed || calls.Load() != 0 {
		t.Fatal("same tool loop tried to undo cancellation", allowed, err)
	}
	turn.applied = []db.TurnInput{{Content: "지금 어디까지 됐어?"}}
	if allowed, err := s.clipResumeRequested(ctx, job); err != nil || allowed {
		t.Fatal("status question resumed a canceled job", allowed, err)
	}
	turn.applied = []db.TurnInput{{Content: "아까 멈춘 작업 다시 이어서 해줘"}}
	if allowed, err := s.clipResumeRequested(ctx, job); err != nil || !allowed {
		t.Fatal("explicit resume was rejected", allowed, err)
	}
	turn.applied = []db.TurnInput{{Content: "잘못된 근거 검사"}}
	if allowed, err := s.clipResumeRequested(ctx, job); err != nil || allowed {
		t.Fatal("fabricated evidence resumed the job", allowed, err)
	}
}
