package server

import (
	"context"
	"encoding/json"
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

// Explicit integration test: use normal memory admission and the running
// Qwen bundle. Save only to an isolated test DB and a requested report directory.
func TestLiveKleinReferenceGeneration(t *testing.T) {
	if os.Getenv("TALK_LIVE_KLEIN_REFERENCES") != "1" && os.Getenv("TALK_LIVE_KLEIN_PAINT") != "1" {
		t.Skip("explicit live Klein integration")
	}
	cfg, _, err := config.Load("../../dist/sparktalk.yaml")
	if err != nil {
		t.Fatal(err)
	}
	s, _ := testImageServer(t)
	s.cfg = cfg
	if cfg.Runtime.Catalog == nil {
		t.Fatal("managed catalogue required")
	}
	s.runtime, err = orchestrator.NewControllerWithCatalog(*cfg.Runtime.Catalog)
	if err != nil {
		t.Fatal(err)
	}
	s.runtime.ConfigurePaths(cfg.Runtime.DataDir, cfg.Runtime.ModelCache)
	const session = "live-generation"
	if _, err = s.db.CreateSession(session, "reference integration", cfg.Model.DefaultModel, "none"); err != nil {
		t.Fatal(err)
	}
	photos := []db.Attachment{}
	for _, id := range []string{"cfbacb8cc17291a6a66afc2ed42e7462", "301ee4494d6136b300cb8775737ec312"} {
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
	if _, err = s.db.AddMessage(session, "user", "원본 사진 두 장을 참조", "", nil, photos); err != nil {
		t.Fatal(err)
	}
	headTargets := []imageHeadTarget{{photos[0].ID, "left person"}, {photos[1].ID, "right person"}}
	seed := int64(77341)
	cases := []imageGenerationArgs{
		{Operation: "reference_generate", HeadTargets: &headTargets, Prompt: "A clear cartoon illustration of the two distinct men from the references standing side by side. Left: the man with glasses in the tan jacket from image 1, waving. Right: the front-facing man in the dark jacket from image 2. Keep their distinct faces, hairstyles and outfits from the actual references. Plain light blue background, clean outlines.", ReferenceImages: []imageReference{{photos[0].ID, "left subject: man in glasses and tan jacket waving"}, {photos[1].ID, "right subject: front-facing man in dark jacket"}}, Size: "1024x768", Seed: &seed},
		{Operation: "identity_edit", SourceImageID: photos[0].ID, SourceImageDescription: "original waving man wearing glasses and a tan jacket", Prompt: "Improve sharpness and clarity slightly. Preserve the original face, glasses, raised hand, clothing and composition. Do not replace the person or add objects.", Size: "768x1024", Seed: &seed},
		{Operation: "identity_edit", SourceImageID: photos[1].ID, SourceImageDescription: "original front-facing man in a dark jacket, bookshelves and flag behind", Prompt: "Improve sharpness and clarity slightly. Preserve the original face, hairstyle, dark jacket and front-facing pose. Do not add a raised hand, glasses, or a new person.", Size: "1024x1024", Seed: &seed},
	}
	report := os.Getenv("TALK_LIVE_REFERENCE_REPORT")
	if report == "" {
		t.Fatal("TALK_LIVE_REFERENCE_REPORT required")
	}
	if err = os.MkdirAll(report, 0700); err != nil {
		t.Fatal(err)
	}
	var intermediate db.Attachment
	// Fourth case edits the generated scene with both ORIGINAL references,
	// exercising the requested multi-stage composition path.
	cases = append(cases, imageGenerationArgs{Operation: "identity_edit", SourceImageDescription: "generated cartoon scene to refine without changing subject placement", ReferenceImages: []imageReference{{photos[0].ID, "original identity for the left waving man"}, {photos[1].ID, "original identity for the right front-facing man"}}, Prompt: "Refine only the faces of the two men in image 1 using the distinct original people in images 2 and 3. Keep the cartoon style, background, placement, clothing, and poses of image 1. Left face follows image 2; right face follows image 3.", Size: "1024x768", Seed: &seed})
	paint := os.Getenv("TALK_LIVE_KLEIN_PAINT") == "1"
	if paint {
		cases = []imageGenerationArgs{
			{Operation: "generate", Prompt: "A small turtle on a mossy log in a sunlit forest, detailed nature photograph.", Size: "512x512", Seed: &seed},
			{Operation: "inpaint", SourceImageID: photos[0].ID, Prompt: "Repair the small empty corner of the stone wall with matching grey marble texture.", MaskBox: []int{0, 0, 64, 64}, Seed: &seed},
			{Operation: "outpaint", SourceImageID: photos[1].ID, Prompt: "Continue the existing bookshelf and room background naturally around this portrait. Preserve the central portrait.", OutpaintLeft: 32, OutpaintRight: 32, OutpaintTop: 32, OutpaintBottom: 32, PreserveSource: true, Seed: &seed},
			{Operation: "generate", Prompt: "A small turtle on a mossy log in a sunlit forest, detailed nature photograph.", Size: "512x512", Seed: &seed},
		}
	}
	for i, args := range cases {
		if i == 3 && !paint {
			args.SourceImageID = intermediate.ID
		}
		b, _ := json.Marshal(args)
		ctx, cancel := context.WithTimeout(context.Background(), 6*time.Minute)
		start := time.Now()
		result, err := s.executeImageGenerateTool(ctx, session, cfg.Image, llm.ToolCall{Function: llm.FunctionCall{Name: "image_generate", Arguments: string(b)}}, func(event string, value any) error {
			if event == "tool_output" {
				t.Logf("progress: %v", value)
			}
			return nil
		})
		cancel()
		if err != nil {
			t.Fatal(err)
		}
		if len(result.Attachments) != 1 {
			t.Fatalf("missing output: %s", result.Result)
		}
		if i == 0 {
			intermediate = result.Attachments[0]
			if _, err = s.db.AddMessage(session, "assistant", "intermediate scene", "", nil, result.Attachments); err != nil {
				t.Fatal(err)
			}
		}
		f, err := s.media.Open(result.Attachments[0])
		if err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(f.Name())
		f.Close()
		if err != nil {
			t.Fatal(err)
		}
		stem := filepath.Join(report, args.Operation+"-"+string(rune('1'+i)))
		if err = os.WriteFile(stem+".png", data, 0600); err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(stem+".json", []byte(result.Result), 0600); err != nil {
			t.Fatal(err)
		}
		t.Logf("%s completed in %s, output=%s", args.Operation, time.Since(start).Round(time.Millisecond), stem+".png")
	}
}
