package server

import (
	"context"
	"encoding/json"
	"image"
	_ "image/png"
	"os"
	"path/filepath"
	"testing"
	"time"

	"sparktalk/internal/config"
	"sparktalk/internal/llm"
	"sparktalk/internal/orchestrator"
)

// Explicit live test. It uses normal Talk admission and the configured CUDA
// services, while keeping messages and generated attachments in a temporary DB.
func TestLiveQAD640KImage(t *testing.T) {
	if os.Getenv("TALK_LIVE_QAD640_IMAGE") != "1" {
		t.Skip("explicit QAD640K/image integration")
	}
	report := os.Getenv("TALK_LIVE_QAD640_REPORT")
	if report == "" {
		t.Fatal("TALK_LIVE_QAD640_REPORT required")
	}
	cfg, _, err := config.Load("../../dist/sparktalk.yaml")
	if err != nil {
		t.Fatal(err)
	}
	bundle, ok := cfg.Runtime.Catalog.Bundle("flash-next")
	if !ok || bundle.ContextTokens != 655360 || cfg.Context.WindowTokens != 655360 {
		t.Fatal("640K QAD profile required")
	}
	s, _ := testImageServer(t)
	s.cfg = cfg
	s.runtime, err = orchestrator.NewControllerWithCatalog(*cfg.Runtime.Catalog)
	if err != nil {
		t.Fatal(err)
	}
	s.runtime.ConfigurePaths(cfg.Runtime.DataDir, cfg.Runtime.ModelCache)
	t.Cleanup(s.runtime.Close)
	if err := os.MkdirAll(report, 0700); err != nil {
		t.Fatal(err)
	}
	seed := int64(1096)
	args := imageGenerationArgs{
		Operation: "generate", Size: "1024x1024", Seed: &seed,
		Prompt: "An adorable baby penguin standing on sparkling Antarctic snow, fluffy grey feathers, tiny orange feet, shiny black eyes, soft rosy cheeks, a pale blue scarf fluttering in the breeze, pastel blue sky, warm gentle sunrise lighting, charming storybook illustration, clean composition, no text.",
	}
	body, _ := json.Marshal(args)
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Minute)
	defer cancel()
	start := time.Now()
	result, err := s.executeImageGenerateTool(ctx, "session", cfg.Image, llm.ToolCall{Function: llm.FunctionCall{Name: "image_generate", Arguments: string(body)}}, func(string, any) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Attachments) != 1 {
		t.Fatalf("expected one output: %s", result.Result)
	}
	file, err := s.media.Open(result.Attachments[0])
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(file.Name())
	file.Close()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(report, "penguin.png"), data, 0600); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(filepath.Join(report, "penguin.png"))
	if err != nil {
		t.Fatal(err)
	}
	size, _, err := image.DecodeConfig(f)
	f.Close()
	if err != nil || size.Width != 1024 || size.Height != 1024 {
		t.Fatalf("wrong output size: %+v %v", size, err)
	}
	if err := os.WriteFile(filepath.Join(report, "result.json"), []byte(result.Result), 0600); err != nil {
		t.Fatal(err)
	}
	t.Logf("QAD640K + Talk image tool: 1024x1024 penguin generated in %s", time.Since(start).Round(time.Millisecond))
}
