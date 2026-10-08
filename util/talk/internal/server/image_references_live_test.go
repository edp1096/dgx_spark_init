package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/media"
)

// Opt-in vision/tool-selection test against the installed Qwen. No production
// conversation is modified and no diffusion request is made by this test.
func TestLiveQwenOrderedImageReferences(t *testing.T) {
	if os.Getenv("TALK_LIVE_IMAGE_REFERENCES") != "1" {
		t.Skip("explicit live Qwen image-reference test")
	}
	cfg, _, err := config.Load("../../dist/sparktalk.yaml")
	if err != nil {
		t.Fatal(err)
	}
	s, _ := testImageServer(t)
	const session = "live-reference"
	if _, err = s.db.CreateSession(session, "live reference test", cfg.Model.DefaultModel, "none"); err != nil {
		t.Fatal(err)
	}
	ids := []string{"cfbacb8cc17291a6a66afc2ed42e7462", "301ee4494d6136b300cb8775737ec312"}
	photos := []db.Attachment{}
	for _, id := range ids {
		f, err := os.Open(filepath.Join("../../dist/sparktalk.db.media", id))
		if err != nil {
			t.Fatal(err)
		}
		a, err := s.media.SaveReader(f, "images.jpg", "image/jpeg", media.MaxImageBytes)
		f.Close()
		if err != nil {
			t.Fatal(err)
		}
		photos = append(photos, a)
	}
	if _, err = s.db.AddMessage(session, "user", "원본 사진 두 장", "", nil, photos); err != nil {
		t.Fatal(err)
	}
	client := llm.New(cfg.Model.Endpoint, cfg.Model.DefaultModel, cfg.Model.APIKey, cfg.Model.ModelType)
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString(onePixelPNG)}}})
	}))
	defer worker.Close()
	cases := []struct {
		request, operation string
		selected           []string
	}{
		{"첨부한 원본 사진 두 장의 서로 다른 인물을 한 장의 만화 장면에 그려라. 첫 번째 사진의 손 흔드는 인물은 왼쪽, 두 번째 사진의 정면 초상 인물은 오른쪽에 둬라. 원본 두 장을 실제 참조로 전달하고 얼굴과 복장을 반영해라. image_generate 한 번만 호출해라.", "reference_generate", []string{photos[0].ID, photos[1].ID}},
		{"첨부한 첫 번째 사진(손 흔드는 인물)만 리터칭해라. 사진 속 얼굴·손동작·복장·구도를 유지하고 선명도만 개선해라. 첫 번째 사진 ID를 source_image_id로 지정하고 실제 사진을 확인한 설명을 source_image_description에 써라. image_generate 한 번만 호출해라.", "identity_edit", []string{photos[0].ID}},
		{"첨부한 두 번째 사진(정면 초상 인물)만 리터칭해라. 사진 속 얼굴·자세·복장·구도를 유지하고 선명도만 개선해라. 두 번째 사진 ID를 source_image_id로 지정하고 실제 사진을 확인한 설명을 source_image_description에 써라. image_generate 한 번만 호출해라.", "identity_edit", []string{photos[1].ID}},
	}
	var scene db.Attachment
	if path := os.Getenv("TALK_LIVE_REFERENCE_SCENE"); path != "" {
		f, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		scene, err = s.media.SaveReader(f, "generated-scene.png", "image/png", media.MaxImageBytes)
		f.Close()
		if err != nil {
			t.Fatal(err)
		}
		if _, err = s.db.AddMessage(session, "assistant", "중간 생성 장면", "", nil, []db.Attachment{scene}); err != nil {
			t.Fatal(err)
		}
		cases = append(cases, struct {
			request, operation string
			selected           []string
		}{"직전 생성한 만화 장면의 얼굴을 원본 사진 두 장을 기준으로 다시 수정해라. 생성 장면은 source_image_id이고 원본 두 장은 reference_images이다. 왼쪽 손 흔드는 인물은 첫 번째 원본, 오른쪽 정면 인물은 두 번째 원본을 따라라. 배경과 복장과 좌우 배치를 유지해라. image_generate 한 번만 호출해라.", "identity_edit", []string{scene.ID}})
	}
	for _, tc := range cases {
		messages, err := s.llmMessages(context.Background(), []db.Message{{Role: "user", Content: tc.request, Attachments: photos}}, config.Config{})
		if err != nil {
			t.Fatal(err)
		}
		if scene.ID != "" && tc.selected[0] == scene.ID {
			messages, err = s.llmMessages(context.Background(), []db.Message{{Role: "user", Content: "원본 사진 두 장", Attachments: photos}, {Role: "assistant", Content: "직전 생성한 중간 장면", Attachments: []db.Attachment{scene}}, {Role: "user", Content: tc.request}}, config.Config{})
			if err != nil {
				t.Fatal(err)
			}
		}
		messages = append([]llm.Message{{Role: "system", Content: imageToolSystemPrompt("paint") + "\n" + imageAttachmentCatalog(s, session)}}, messages...)
		for attempt := 0; attempt < 3; attempt++ {
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
			result, err := client.Stream(ctx, messages, cfg.Model.DefaultModel, "none", []llm.Tool{imageGenerateToolDefinition("paint")}, func(string, string) error { return nil })
			cancel()
			if err != nil {
				t.Fatal(err)
			}
			if len(result.ToolCalls) != 1 || result.ToolCalls[0].Function.Name != "image_generate" {
				t.Fatalf("expected one image tool call, got %v", result.ToolCalls)
			}
			call := result.ToolCalls[0]
			var args imageGenerationArgs
			if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
				t.Fatal(err)
			}
			t.Logf("Qwen %s: %s", tc.operation, call.Function.Arguments)
			if args.Operation != tc.operation {
				t.Fatalf("wrong operation: %s", args.Operation)
			}
			if tc.operation == "reference_generate" {
				if len(args.ReferenceImages) != 2 || args.ReferenceImages[0].ImageID != tc.selected[0] || args.ReferenceImages[1].ImageID != tc.selected[1] {
					t.Fatalf("Qwen dropped/swapped original references: %+v", args.ReferenceImages)
				}
			} else if args.SourceImageID != tc.selected[0] {
				t.Fatalf("Qwen swapped the retouch source: %s", args.SourceImageID)
			}
			if scene.ID != "" && tc.selected[0] == scene.ID {
				if len(args.ReferenceImages) != 2 || args.ReferenceImages[0].ImageID != photos[0].ID || args.ReferenceImages[1].ImageID != photos[1].ID {
					t.Fatalf("multi-stage edit lost original references: %+v", args)
				}
			}
			items, _ := s.sessionImageAttachments(session)
			if _, err := s.validateImageReferences(context.Background(), session, "paint", args, items); err != nil {
				t.Fatal(err)
			}
			if _, err := s.executeImageGenerateTool(context.Background(), session, config.ImageConfig{Endpoint: worker.URL, Model: "test", Mode: "paint", DefaultSize: "1024x1024", Timeout: "2s"}, call, func(string, any) error { return nil }); err != nil {
				if attempt == 2 {
					t.Fatalf("Qwen did not correct rejected arguments: %v", err)
				}
				t.Logf("Rejected attempt %d: %v", attempt+1, err)
				messages = append(messages, llm.Message{Role: "assistant", Content: result.Content, ToolCalls: result.ToolCalls}, llm.Message{Role: "tool", ToolCallID: call.ID, Content: "Tool rejected the arguments before generation: " + err.Error() + ". Correct the arguments using only the tool schema. Do not invent options; omit size unless necessary."})
				continue
			}
			break
		}
	}
}
