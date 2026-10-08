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
	"sparktalk/internal/llm"
)

func TestBFSHeadSwapSendsBaseAndOriginalSeparatelyAndRecordsCropCoordinates(t *testing.T) {
	s, _ := testImageServer(t)
	base := addReferenceFixture(t, s, "assistant", color.RGBA{R: 255, A: 255}, 128, 160)
	head := addReferenceFixture(t, s, "user", color.RGBA{B: 255, A: 255}, 160, 128)
	var payload map[string]any
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/images/generations" {
			t.Errorf("wrong endpoint: %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Error(err)
		}
		json.NewEncoder(w).Encode(map[string]any{"seed": 42, "data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString(onePixelPNG)}}})
	}))
	defer worker.Close()
	strength := 1.3
	args := imageGenerationArgs{Operation: "head_swap", SourceImageID: base.ID, ReferenceImages: []imageReference{{head.ID, "original head"}}, Prompt: "Replace only the selected head.", HeadBoxCoordinates: "normalized_1000", MaskBox: []int{125, 100, 750, 800}, ReferenceCropBox: []int{200, 0, 800, 750}, HeadSwapStrength: &strength}
	b, _ := json.Marshal(args)
	result, err := s.executeImageGenerateTool(context.Background(), "session", config.ImageConfig{Endpoint: worker.URL, Mode: "paint", Model: "test", DefaultSize: "1024x1024", Timeout: "2s"}, llm.ToolCall{Function: llm.FunctionCall{Arguments: string(b)}}, func(string, any) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	if payload["operation"] != "head_swap" || payload["head_image"] == payload["source_image"] || payload["head_swap_strength"] != 1.3 || payload["anypaint_image"] != nil {
		t.Fatalf("wrong BFS transport: %v", payload)
	}
	if fmt.Sprint(payload["mask_box"]) != "[16 16 96 128]" || fmt.Sprint(payload["reference_crop_box"]) != "[32 0 128 96]" || payload["head_box_coordinates"] != nil {
		t.Fatalf("wrong coordinate conversion: %v", payload)
	}
	var report struct {
		Inputs []imageInputEvidence `json:"input_images"`
	}
	if err = json.Unmarshal([]byte(result.Result), &report); err != nil {
		t.Fatal(err)
	}
	if len(report.Inputs) != 2 || report.Inputs[0].ID != base.ID || report.Inputs[1].ID != head.ID || len(report.Inputs[0].CropBox) != 4 || report.Inputs[0].Origin != "assistant" || report.Inputs[1].Origin != "user" {
		t.Fatalf("wrong BFS provenance: %s", result.Result)
	}
}

func TestBFSRejectsBadReferenceCountRegionsAndStrengthBeforeWorker(t *testing.T) {
	s, _ := testImageServer(t)
	base := addReferenceFixture(t, s, "assistant", color.RGBA{R: 255, A: 255}, 128, 160)
	head := addReferenceFixture(t, s, "user", color.RGBA{B: 255, A: 255}, 160, 128)
	badStrength := 2.0
	cases := []imageGenerationArgs{
		{Operation: "head_swap", SourceImageID: base.ID},
		{Operation: "head_swap", SourceImageID: base.ID, ReferenceImages: []imageReference{{head.ID, "head"}}, MaskBox: []int{0, 0, 999, 999}},
		{Operation: "head_swap", SourceImageID: base.ID, ReferenceImages: []imageReference{{head.ID, "head"}}, ReferenceCropBox: []int{0, 0, 20, 20}},
		{Operation: "head_swap", SourceImageID: base.ID, ReferenceImages: []imageReference{{head.ID, "head"}}, HeadSwapStrength: &badStrength},
		{Operation: "head_swap", SourceImageID: base.ID, ReferenceImages: []imageReference{{base.ID, "wrong base as reference"}}},
		{Operation: "head_swap", SourceImageID: base.ID, ReferenceImages: []imageReference{{head.ID, "head"}}, HeadBoxCoordinates: "normalized_1000", MaskBox: []int{0, 0, 1001, 1000}},
		{Operation: "head_swap", SourceImageID: base.ID, ReferenceImages: []imageReference{{head.ID, "head"}}, HeadBoxCoordinates: "unknown"},
	}
	for _, args := range cases {
		args.Prompt = "replace the head"
		b, _ := json.Marshal(args)
		_, err := s.executeImageGenerateTool(context.Background(), "session", config.ImageConfig{Endpoint: "http://127.0.0.1:1", Mode: "paint", Model: "test", DefaultSize: "1024x1024"}, llm.ToolCall{Function: llm.FunctionCall{Arguments: string(b)}}, func(string, any) error { return nil })
		if err == nil || strings.Contains(err.Error(), "connection refused") {
			t.Fatalf("bad request reached worker: %+v error=%v", args, err)
		}
	}
}
