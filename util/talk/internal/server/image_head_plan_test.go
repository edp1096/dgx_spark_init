package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image/color"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
)

func TestHeadPlanCompletesDespitePrematureModelFinal(t *testing.T) {
	for _, tc := range []struct {
		fail   bool
		rounds int
	}{{false, 6}, {true, 6}, {false, 1}} {
		fail := tc.fail
		t.Run(fmt.Sprintf("failure=%v/rounds=%d", fail, tc.rounds), func(t *testing.T) {
			s, first := testImageServer(t)
			second := addReferenceFixture(t, s, "user", color.RGBA{B: 255, A: 255}, 64, 64)
			heads := []imageHeadTarget{{first.ID, "left person"}, {second.ID, "right person"}}
			calls := 0
			worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls++
				if fail && calls == 2 {
					http.Error(w, "BFS failed", 500)
					return
				}
				json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString(onePixelPNG)}}})
			}))
			defer worker.Close()
			s.cfg.Image = config.ImageConfig{Enabled: true, Mode: "paint", Endpoint: worker.URL, Model: "test", DefaultSize: "512x512", Timeout: "2s"}
			n := 0
			model := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				n++
				w.Header().Set("Content-Type", "text/event-stream")
				delta := map[string]any{"content": "prematurely finished"}
				if n == 1 {
					a, _ := json.Marshal(imageGenerationArgs{Operation: "reference_generate", Prompt: "left person slaps right person", ReferenceImages: []imageReference{{first.ID, "left identity"}, {second.ID, "right identity"}}, HeadTargets: &heads})
					delta = map[string]any{"tool_calls": []map[string]any{{"index": 0, "id": "scene", "type": "function", "function": map[string]string{"name": "image_generate", "arguments": string(a)}}}}
				}
				if n == 2 && !fail {
					delta = map[string]any{"tool_calls": []map[string]any{{"index": 0, "id": "wrong-pair", "type": "function", "function": map[string]string{"name": "image_generate", "arguments": `{"operation":"head_swap","source_image_id":"old-scene","reference_images":[{"image_id":"wrong-person","description":"wrong"}],"prompt":"wrong person"}`}}}}
				}
				b, _ := json.Marshal(map[string]any{"choices": []map[string]any{{"delta": delta}}})
				fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", b)
			}))
			defer model.Close()
			var streamed strings.Builder
			r, err := runCompletionLoopForSessionWithMedia(s, "session", context.Background(), llm.New(model.URL, "test", ""), []llm.Message{{Role: "user", Content: "draw these people"}}, "test", "none", "", config.ToolsConfig{Enabled: true, MaxRounds: tc.rounds}, true, func(kind string, p any) error {
				if kind == "delta" {
					if v, ok := p.(map[string]string); ok {
						streamed.WriteString(v["delta"])
					}
				}
				return nil
			}, func(db.Attachment) error { return nil })
			if tc.rounds == 1 {
				if err == nil || calls != 1 || len(r.ToolTrace) != 1 || !strings.Contains(r.Content, "중간 결과") || strings.Contains(streamed.String(), "prematurely") {
					t.Fatalf("round limit concealed incomplete plan: calls=%d result=%+v err=%v", calls, r, err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if calls != 3 || len(r.ToolTrace) != 3 {
				t.Fatalf("plan skipped or repeated: calls=%d trace=%+v", calls, r.ToolTrace)
			}
			var firstResult struct {
				Attachments []db.Attachment `json:"attachments"`
			}
			json.Unmarshal([]byte(r.ToolTrace[0].Result), &firstResult)
			base := firstResult.Attachments[0].ID
			for i, trace := range r.ToolTrace[1:] {
				var a imageGenerationArgs
				json.Unmarshal([]byte(trace.Arguments), &a)
				if a.Operation != "head_swap" || a.SourceImageID != base || a.ReferenceImages[0].ImageID != heads[i].ReferenceImageID || !strings.Contains(a.Prompt, "slaps") {
					t.Fatalf("wrong chain: %+v", a)
				}
				if trace.Error == "" {
					var out struct {
						Attachments []db.Attachment `json:"attachments"`
					}
					json.Unmarshal([]byte(trace.Result), &out)
					base = out.Attachments[0].ID
				}
			}
			if fail {
				if !strings.Contains(r.Content, "실패") || strings.Contains(streamed.String(), "prematurely") {
					t.Fatalf("failure hidden: %q / %q", r.Content, streamed.String())
				}
			} else if strings.Count(streamed.String(), "prematurely") != 1 {
				t.Fatalf("premature completion leaked: %q", streamed.String())
			}
		})
	}
}

func TestHeadPlanRetryUsesNewSceneAndRejectsDroppingTargets(t *testing.T) {
	turn := &turnImages{}
	ctx := context.WithValue(context.Background(), turnImageKey{}, turn)
	targets := []imageHeadTarget{{"original", "left person"}}
	a := imageGenerationArgs{PaintMode: true, Operation: "reference_generate", ReferenceImages: []imageReference{{"original", "portrait"}}, HeadTargets: &targets, Prompt: "requested action"}
	if err := validateHeadPlan(ctx, a); err != nil {
		t.Fatal(err)
	}
	recordHeadPlan(ctx, a, "scene1")
	recordHeadPlan(ctx, a, "scene2")
	call := pendingHeadCall(ctx)
	var next imageGenerationArgs
	json.Unmarshal([]byte(call.Function.Arguments), &next)
	if next.SourceImageID != "scene2" {
		t.Fatal(next)
	}
	empty := []imageHeadTarget{}
	a.HeadTargets = &empty
	if validateHeadPlan(ctx, a) == nil {
		t.Fatal("dropped target accepted")
	}
	next.SourceImageID = "scene1"
	if validateHeadPlan(ctx, next) == nil {
		t.Fatal("stale scene accepted")
	}
	next.SourceImageID = "scene2"
	recordHeadPlan(ctx, next, "corrected")
	if pendingHeadCall(ctx) != nil {
		t.Fatal("completed head repeated")
	}
	for _, bad := range []imageGenerationArgs{{PaintMode: true, Operation: "reference_generate"}, {Operation: "generate", HeadTargets: &targets}} {
		if validateHeadPlan(context.Background(), bad) == nil {
			t.Fatal("invalid plan accepted")
		}
	}
	a.HeadTargets = &empty
	if validateHeadPlan(context.Background(), a) != nil {
		t.Fatal("object/style references rejected")
	}
}
