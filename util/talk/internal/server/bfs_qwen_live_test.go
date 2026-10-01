package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
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
	"sparktalk/internal/orchestrator"
)

func TestLiveQwenBFSHeadSelection(t *testing.T) {
	if os.Getenv("TALK_LIVE_BFS_QWEN") != "1" {
		t.Skip("explicit Qwen BFS tool-selection test")
	}
	cfg, _, err := config.Load("../../dist/sparktalk.yaml")
	if err != nil {
		t.Fatal(err)
	}
	s, _ := testImageServer(t)
	const session = "live-bfs-qwen"
	if _, err = s.db.CreateSession(session, "BFS routing", cfg.Model.DefaultModel, "none"); err != nil {
		t.Fatal(err)
	}
	photos := []db.Attachment{}
	for i, id := range []string{"cfbacb8cc17291a6a66afc2ed42e7462", "301ee4494d6136b300cb8775737ec312", "ad3ad70e69762efb8af36552bf2364cb"} {
		f, err := os.Open(filepath.Join("../../dist/sparktalk.db.media", id))
		if err != nil {
			t.Fatal(err)
		}
		mime, role := "image/jpeg", "user"
		if i == 2 {
			mime, role = "image/png", "assistant"
		}
		a, err := s.media.SaveReader(f, id, mime, media.MaxImageBytes)
		f.Close()
		if err != nil {
			t.Fatal(err)
		}
		if _, err = s.db.AddMessage(session, role, "image input", "", nil, []db.Attachment{a}); err != nil {
			t.Fatal(err)
		}
		photos = append(photos, a)
	}
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString(onePixelPNG)}}})
	}))
	defer worker.Close()
	imageConfig := config.ImageConfig{Endpoint: worker.URL, Model: "test", Mode: "paint", DefaultSize: "1024x1024", Timeout: "2s"}
	report := os.Getenv("TALK_LIVE_BFS_QWEN_REPORT")
	if report != "" {
		if err := os.MkdirAll(report, 0700); err != nil {
			t.Fatal(err)
		}
		s.cfg = cfg
		s.runtime, err = orchestrator.NewControllerWithCatalog(*cfg.Runtime.Catalog)
		if err != nil {
			t.Fatal(err)
		}
		s.runtime.ConfigurePaths(cfg.Runtime.DataDir, cfg.Runtime.ModelCache)
		imageConfig = cfg.Image
	}
	client := llm.New(cfg.Model.Endpoint, cfg.Model.DefaultModel, cfg.Model.APIKey, cfg.Model.ModelType)
	for i, request := range []string{
		"직전 만화의 왼쪽 인물 머리만 첫 번째 원본 사진의 머리로 교체해라. 오른쪽 인물, 옷, 몸과 배경은 그대로 둬라. BFS head_swap을 한 번 호출해라. 좌표는 제공하지 않았다.",
		"직전 만화의 오른쪽 인물 머리만 두 번째 원본 사진의 머리로 교체해라. 왼쪽 인물, 옷, 몸과 배경은 그대로 둬라. BFS head_swap을 한 번 호출해라. 좌표는 제공하지 않았다.",
	} {
		history := []db.Message{{Role: "user", Content: "첫 번째와 두 번째 원본 사진", Attachments: photos[:2]}, {Role: "assistant", Content: "직전 생성한 만화", Attachments: photos[2:]}, {Role: "user", Content: request}}
		messages, err := s.llmMessages(context.Background(), history, config.Config{})
		if err != nil {
			t.Fatal(err)
		}
		messages = append([]llm.Message{{Role: "system", Content: imageToolSystemPrompt("paint") + "\n" + imageAttachmentCatalog(s, session)}}, messages...)

		for attempt := 0; attempt < 3; attempt++ {
			ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
			result, err := client.Stream(ctx, messages, cfg.Model.DefaultModel, "none", []llm.Tool{imageGenerateToolDefinition("paint")}, func(string, string) error { return nil })
			cancel()
			if err != nil {
				t.Fatal(err)
			}
			if len(result.ToolCalls) != 1 {
				t.Fatalf("expected one head tool, got %v", result.ToolCalls)
			}
			call := result.ToolCalls[0]
			var args imageGenerationArgs
			if err = json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
				t.Fatal(err)
			}
			t.Logf("Qwen head %d: %s", i+1, call.Function.Arguments)
			if args.Operation != "head_swap" || args.SourceImageID != photos[2].ID || len(args.ReferenceImages) != 1 || args.ReferenceImages[0].ImageID != photos[i].ID {
				t.Fatalf("wrong base/reference assignment: %+v", args)
			}
			if len(args.MaskBox) > 0 || len(args.ReferenceCropBox) > 0 {
				t.Fatal("Qwen invented a crop region without supplied coordinates")
			}
			imageResult, imageErr := s.executeImageGenerateTool(context.Background(), session, imageConfig, call, func(string, any) error { return nil })
			err = imageErr
			if err == nil {
				if report != "" {
					f, e := s.media.Open(imageResult.Attachments[0])
					if e != nil {
						t.Fatal(e)
					}
					data, e := os.ReadFile(f.Name())
					f.Close()
					if e != nil {
						t.Fatal(e)
					}
					stem := filepath.Join(report, fmt.Sprintf("qwen-head-%d", i+1))
					if e = os.WriteFile(stem+".png", data, 0600); e != nil {
						t.Fatal(e)
					}
					if e = os.WriteFile(stem+".json", []byte(imageResult.Result), 0600); e != nil {
						t.Fatal(e)
					}
				}
				break
			}
			if attempt == 2 {
				t.Fatalf("Qwen did not correct invalid crop/arguments: %v", err)
			}
			messages = append(messages, llm.Message{Role: "assistant", Content: result.Content, ToolCalls: result.ToolCalls}, llm.Message{Role: "tool", ToolCallID: call.ID, Content: "Rejected before generation: " + err.Error() + ". Correct only the invalid fields using the schema."})
		}
	}
}
