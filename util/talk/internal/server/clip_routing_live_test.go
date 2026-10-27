package server

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"sparktalk/internal/db"
	"sparktalk/internal/llm"
)

// The real local conversation model chooses a tool; no media tool executes.
func TestClipNaturalLanguageRoutingLive(t *testing.T) {
	endpoint := os.Getenv("TALK_CLIP_MODEL_ENDPOINT")
	if endpoint == "" {
		t.Skip("set TALK_CLIP_MODEL_ENDPOINT for read-only model routing checks")
	}
	s, reg := shortsToolTestServer(t, http.NotFound)
	s.registerVideoTool(context.Background(), &reg, s.cfg, func(db.Attachment) error { return nil })
	s.registerClipProduction(context.Background(), &reg)
	reg.register(imageGenerateToolDefinition("reference"), func(context.Context, llm.ToolCall, []llm.Message, eventEmitter) (registeredToolResult, error) {
		t.Fatal("read-only routing test must never execute an image tool")
		return registeredToolResult{}, nil
	})
	history, _ := s.db.Messages("session")
	imageID := history[0].Attachments[0].ID
	client := llm.New(endpoint, "qwen38fn_exl3", "", "qwen3.8-tensorfold")
	for _, tc := range []struct{ prompt, context, tool, action string }{
		{"이 사진 만화처럼 바꾸고 웃으면서 손 흔들게 해줘.", "The attached photo has one person with a visible right hand. Source image ID: " + imageID, "clip_produce", ""},
		{"이 사진 자연스럽게 움직이게 해줘. 720p로.", "Source image ID: " + imageID, "clip_produce", ""},
		{"그 사진 영상 작업 어디까지 됐어?", "The latest photo-to-video production is running.", "clip_manage", "status"},
		{"아까 사진 영상 만들던 거 중지해.", "The latest photo-to-video production is running.", "clip_manage", "cancel"},
		{"만화 사진만 만들어줘. 영상은 만들지 마.", "Source image ID: " + imageID, "image_generate", ""},
	} {
		t.Run(tc.prompt, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
			defer cancel()
			result, err := client.Stream(llm.WithOutputLimit(ctx, 1000), []llm.Message{{Role: "system", Content: strings.Join(reg.prompts, "\n") + "\n" + tc.context}, {Role: "user", Content: tc.prompt}}, "qwen38fn_exl3", "none", reg.definitions, func(string, string) error { return nil })
			if err != nil {
				t.Fatal(err)
			}
			if len(result.ToolCalls) != 1 || result.ToolCalls[0].Function.Name != tc.tool {
				t.Fatalf("wrong routing: %+v / %s", result.ToolCalls, result.Content)
			}
			var args map[string]any
			_ = json.Unmarshal([]byte(result.ToolCalls[0].Function.Arguments), &args)
			if tc.action != "" && args["action"] != tc.action {
				t.Fatal("wrong management action", args)
			}
			if tc.tool == "clip_produce" && args["source_image_id"] != imageID {
				t.Fatal("source image ID changed", args)
			}
			if strings.Contains(tc.prompt, "720p") && args["resolution"] != "720p" {
				t.Fatal("requested resolution lost", args)
			}
			t.Log(result.ToolCalls[0].Function.Name, result.ToolCalls[0].Function.Arguments)
		})
	}
}

func TestClipIntentRouterLive(t *testing.T) {
	endpoint := os.Getenv("TALK_CLIP_MODEL_ENDPOINT")
	if endpoint == "" {
		t.Skip("set TALK_CLIP_MODEL_ENDPOINT for intent routing checks")
	}
	var sourceID string
	s, _ := shortsToolTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/clips" {
			json.NewEncoder(w).Encode([]any{map[string]any{"id": strings.Repeat("a", 32), "status": "failed", "project": map[string]string{"source_image_id": sourceID}}})
			return
		}
		w.Write([]byte(`{"status":"ok"}`))
	})
	history, _ := s.db.Messages("session")
	sourceID = history[0].Attachments[0].ID
	client := llm.New(endpoint, "qwen38fn_exl3", "", "qwen3.8-tensorfold")
	for _, tc := range []struct{ request, tool string }{
		{"같은 원본 사진으로 새 버전을 다시 만들어줘. 차가 조금 굴러가는 영상으로.", "clip_produce"},
		{"이 사진 만화로 바꾸고 자연스럽게 움직이는 영상으로 만들어줘.", "clip_produce"},
		{"사진만 만화로 바꿔줘. 영상은 만들지 마.", ""},
		{"이 사진을 영상으로 만들 수 있는지 방식만 설명해줘.", ""},
	} {
		t.Run(tc.request, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
			defer cancel()
			call, err := s.routeClipIntent(ctx, "session", []llm.Message{{Role: "assistant", Content: "이전 영상은 검수 실패로 끝났습니다."}, {Role: "user", Content: tc.request}}, client, "qwen38fn_exl3")
			if err != nil {
				t.Fatal(err)
			}
			if tc.tool == "" {
				if call != nil {
					t.Fatal("unrequested production", call)
				}
				return
			}
			if call == nil || call.Function.Name != tc.tool {
				t.Fatal("production request not routed", call)
			}
			t.Log(call.Function.Name, call.Function.Arguments)
		})
	}
}
