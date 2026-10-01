package server

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/media"
	"sparktalk/internal/orchestrator"
)

// Opt-in integration uses the normal workload admission, installed Qwen bundle,
// and a separate test DB. Production conversations are never modified.
func TestLiveBFSHeadSwap(t *testing.T) {
	if os.Getenv("TALK_LIVE_BFS") != "1" {
		t.Skip("explicit BFS GPU integration")
	}
	cfg, _, err := config.Load("../../dist/sparktalk.yaml")
	if err != nil {
		t.Fatal(err)
	}
	s, _ := testImageServer(t)
	s.cfg = cfg
	s.runtime, err = orchestrator.NewControllerWithCatalog(*cfg.Runtime.Catalog)
	if err != nil {
		t.Fatal(err)
	}
	s.runtime.ConfigurePaths(cfg.Runtime.DataDir, cfg.Runtime.ModelCache)
	const session = "live-bfs"
	if _, err = s.db.CreateSession(session, "BFS isolated verification", cfg.Model.DefaultModel, "none"); err != nil {
		t.Fatal(err)
	}
	images := []db.Attachment{}
	for i, id := range []string{"cfbacb8cc17291a6a66afc2ed42e7462", "301ee4494d6136b300cb8775737ec312", "ad3ad70e69762efb8af36552bf2364cb"} {
		f, err := os.Open(filepath.Join("../../dist/sparktalk.db.media", id))
		if err != nil {
			t.Fatal(err)
		}
		mime, role, name := "image/jpeg", "user", id+".jpg"
		if i == 2 {
			mime, role, name = "image/png", "assistant", id+".png"
		}
		a, err := s.media.SaveReader(f, name, mime, media.MaxImageBytes)
		f.Close()
		if err != nil {
			t.Fatal(err)
		}
		if _, err = s.db.AddMessage(session, role, "original test input", "", nil, []db.Attachment{a}); err != nil {
			t.Fatal(err)
		}
		images = append(images, a)
	}
	report := os.Getenv("TALK_LIVE_BFS_REPORT")
	if report == "" {
		t.Fatal("report path required")
	}
	if err = os.MkdirAll(report, 0700); err != nil {
		t.Fatal(err)
	}
	seed := int64(77341)
	cases := []imageGenerationArgs{
		{Operation: "generate", Prompt: "A small turtle on a mossy log in a sunlit forest, detailed nature photograph.", Size: "512x512", Seed: &seed},
		{Operation: "head_swap", SourceImageID: images[0].ID, SourceImageDescription: "original waving man in tan jacket; preserve raised hand and jacket", ReferenceImages: []imageReference{{images[1].ID, "original head with round face and flat-top dark hair"}}, MaskBox: []int{105, 25, 355, 285}, ReferenceCropBox: []int{55, 25, 365, 365}, Prompt: "Use the head from Picture 2. Preserve the photographic style, head angle and raised hand of Picture 1. Keep the hair, face shape, eyes, nose and mouth from Picture 2.", Seed: &seed},
		{Operation: "head_swap", SourceImageID: images[2].ID, SourceImageDescription: "cartoon scene; only replace the left older head, retain clothing and newspaper", ReferenceImages: []imageReference{{images[0].ID, "original older head wearing large transparent glasses"}}, MaskBox: []int{85, 35, 375, 340}, ReferenceCropBox: []int{110, 25, 345, 275}, Prompt: "Transfer the actual facial proportions and hairstyle of Picture 2 into the left head of Picture 1. Preserve the existing cartoon line style and scolding expression. Do not invent a younger face, sunglasses, gray hair, or exaggerated chibi facial proportions.", Seed: &seed},
		{Operation: "head_swap", SourceImageDescription: "preceding cartoon with corrected left head; replace only the right head", ReferenceImages: []imageReference{{images[1].ID, "original round face and flat-top dark hair"}}, MaskBox: []int{360, 325, 730, 780}, ReferenceCropBox: []int{55, 25, 365, 365}, Prompt: "Use the facial proportions, nose, mouth and hair of Picture 2 for the right head in Picture 1. Preserve the existing cartoon style, crying expression and both hands touching the head. Do not turn the face into a child or exaggerate its proportions.", Seed: &seed},
		{Operation: "generate", Prompt: "A small turtle on a mossy log in a sunlit forest, detailed nature photograph.", Size: "512x512", Seed: &seed},
	}
	if os.Getenv("TALK_LIVE_BFS_REFINED") == "1" {
		cases[1].MaskBox = []int{85, 10, 375, 320}
		cases[1].Prompt = "Completely replace the original head, hair and eyewear with the adult head from Picture 2. The new head has no eyewear. Keep the target's head angle and photographic rendering."
		cases[2].MaskBox = []int{65, 15, 375, 400}
		cases[2].Prompt = "Completely replace the old head with the adult head in Picture 2. Copy the reference's broad forehead, thick cheeks, natural eye size, broad nose, mouth proportions, hairstyle and large transparent eyeglasses. Render those same adult features in the cartoon line style of Picture 1, with its scolding expression. Keep the neckline and clothes aligned."
		cases[3].MaskBox = []int{385, 350, 710, 790}
		cases[3].Prompt = "Completely replace the old head with the adult head in Picture 2. Copy the reference's broad rounded face, small natural eyes, broad nose, full lips and high dark haircut. Preserve the cartoon line style and crying expression of Picture 1. Keep the hands in front of the head and the neckline aligned."
	}
	if os.Getenv("TALK_LIVE_BFS_CARTOON_FULL") == "1" {
		cases[2].MaskBox = nil
		cases[3].MaskBox = nil
	}
	if value := os.Getenv("TALK_LIVE_BFS_STRENGTH"); value != "" {
		strength, err := strconv.ParseFloat(value, 64)
		if err != nil {
			t.Fatal(err)
		}
		for i := range cases {
			if cases[i].Operation == "head_swap" {
				cases[i].HeadSwapStrength = &strength
			}
		}
	}
	if os.Getenv("TALK_LIVE_BFS_FULL") == "1" {
		f, err := os.Open("/tmp/sparktalk-reference-live/identity_edit-3.png")
		if err != nil {
			t.Fatal(err)
		}
		a, err := s.media.SaveReader(f, "full-size-portrait.png", "image/png", media.MaxImageBytes)
		f.Close()
		if err != nil {
			t.Fatal(err)
		}
		if _, err = s.db.AddMessage(session, "assistant", "generated 1024 portrait", "", nil, []db.Attachment{a}); err != nil {
			t.Fatal(err)
		}
		cases = []imageGenerationArgs{{Operation: "head_swap", SourceImageID: a.ID, SourceImageDescription: "single-person portrait at 1024 square", ReferenceImages: []imageReference{{images[0].ID, "original head with transparent glasses"}}, ReferenceCropBox: []int{110, 25, 345, 275}, Prompt: "Completely replace the head and hairstyle using Picture 2, including its transparent glasses. Keep the photographic rendering, portrait angle, clothing and background.", Seed: &seed}}
	}
	var intermediate db.Attachment
	for i, args := range cases {
		if i == 3 {
			args.SourceImageID = intermediate.ID
		}
		b, _ := json.Marshal(args)
		ctx, cancel := context.WithTimeout(context.Background(), 6*time.Minute)
		start := time.Now()
		result, err := s.executeImageGenerateTool(ctx, session, cfg.Image, llm.ToolCall{Function: llm.FunctionCall{Name: "image_generate", Arguments: string(b)}}, func(string, any) error { return nil })
		cancel()
		if err != nil {
			t.Fatal(err)
		}
		if len(result.Attachments) != 1 {
			t.Fatalf("missing image: %s", result.Result)
		}
		if i == 2 {
			intermediate = result.Attachments[0]
			if _, err = s.db.AddMessage(session, "assistant", "BFS intermediate", "", nil, result.Attachments); err != nil {
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
		t.Logf("%s completed in %s: %s", args.Operation, time.Since(start).Round(time.Millisecond), stem+".png")
	}
}
