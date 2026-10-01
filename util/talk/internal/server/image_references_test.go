package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/media"
)

func addReferenceFixture(t *testing.T, s *Server, role string, c color.RGBA, w, h int) db.Attachment {
	t.Helper()
	im := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			im.SetRGBA(x, y, c)
		}
	}
	var b bytes.Buffer
	if err := png.Encode(&b, im); err != nil {
		t.Fatal(err)
	}
	a, err := s.media.SaveReader(bytes.NewReader(b.Bytes()), "images.png", "image/png", media.MaxImageBytes)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = s.db.AddMessage("session", role, "reference", "", nil, []db.Attachment{a}); err != nil {
		t.Fatal(err)
	}
	return a
}

func TestImageOrderedReferencesReachKleinAndReportActualInputs(t *testing.T) {
	s, _ := testImageServer(t)
	first := addReferenceFixture(t, s, "user", color.RGBA{R: 255, A: 255}, 12, 20)
	second := addReferenceFixture(t, s, "user", color.RGBA{B: 255, A: 255}, 24, 10)
	calls := 0
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.URL.Path != "/v1/images/edits" {
			t.Errorf("references sent to wrong endpoint: %s", r.URL.Path)
			http.Error(w, "wrong endpoint", 400)
			return
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Error(err)
			return
		}
		defer r.MultipartForm.RemoveAll()
		files := r.MultipartForm.File["image"]
		if len(files) != 2 {
			t.Errorf("expected two references, got %d", len(files))
			return
		}
		for i, f := range files {
			input, err := f.Open()
			if err != nil {
				t.Error(err)
				return
			}
			im, err := png.Decode(input)
			input.Close()
			if err != nil {
				t.Error(err)
				return
			}
			rgba := color.RGBAModel.Convert(im.At(0, 0)).(color.RGBA)
			if (i == 0 && rgba.R != 255) || (i == 1 && rgba.B != 255) {
				t.Errorf("reference order changed: %d %v", i, rgba)
			}
		}
		prompt := r.FormValue("prompt")
		if !strings.Contains(prompt, "Image 1: red character on the left") || !strings.Contains(prompt, "Image 2: blue character on the right") {
			t.Errorf("reference role mapping missing: %s", prompt)
		}
		if r.FormValue("seed") != "123" || r.FormValue("model") != "test" {
			t.Errorf("portable fields missing: %v", r.MultipartForm.Value)
		}
		json.NewEncoder(w).Encode(map[string]any{"seed": 123, "data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString(onePixelPNG)}}})
	}))
	defer worker.Close()
	args, _ := json.Marshal(map[string]any{"operation": "reference_generate", "prompt": "Both characters together in a park.", "head_targets": []imageHeadTarget{{first.ID, "left character"}, {second.ID, "right character"}}, "size": "512x512", "seed": 123, "source_image_id": first.ID, "reference_images": []imageReference{{first.ID, "red character on the left"}, {second.ID, "blue character on the right"}}})
	result, err := s.executeImageGenerateTool(context.Background(), "session", config.ImageConfig{Endpoint: worker.URL, Model: "test", Mode: "paint", DefaultSize: "512x512", Timeout: "2s"}, llm.ToolCall{Function: llm.FunctionCall{Arguments: string(args)}}, func(string, any) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	var evidence struct {
		Inputs   []imageInputEvidence `json:"input_images"`
		Verified bool                 `json:"identity_verified"`
	}
	if err := json.Unmarshal([]byte(result.Result), &evidence); err != nil {
		t.Fatal(err)
	}
	if calls != 1 || len(evidence.Inputs) != 2 || evidence.Inputs[0].ID != first.ID || evidence.Inputs[1].ID != second.ID || evidence.Verified {
		t.Fatalf("incorrect audit: %s", result.Result)
	}
	if evidence.Inputs[0].Width != 12 || evidence.Inputs[0].Height != 20 || evidence.Inputs[0].Origin != "user" {
		t.Fatalf("incorrect reference metadata: %+v", evidence.Inputs[0])
	}
	followup, _ := json.Marshal(result.Followups[0].Content)
	if !strings.Contains(string(followup), "not been verified") {
		t.Fatal("result instruction must not claim identity preservation")
	}
	if !strings.Contains(string(followup), `origin=\"assistant\"`) {
		t.Fatal("generated followup image mislabeled as an original upload")
	}
}

func TestImageReferencesRejectMoreThanFourActualInputs(t *testing.T) {
	s, base := testImageServer(t)
	refs := []imageReference{}
	for i := 0; i < 4; i++ {
		a := addReferenceFixture(t, s, "user", color.RGBA{R: uint8(i + 1), A: 255}, 8, 8)
		refs = append(refs, imageReference{a.ID, "distinct reference"})
	}
	items, _ := s.sessionImageAttachments("session")
	_, err := s.validateImageReferences(context.Background(), "session", "paint", imageGenerationArgs{Operation: "identity_edit", SourceImageID: base.ID, ReferenceImages: refs}, items)
	if err == nil || !strings.Contains(err.Error(), "at most four") {
		t.Fatalf("accepted five actual inputs: %v", err)
	}
}

func TestImageReferenceValidationRejectsMissingOriginalAndWrongIDs(t *testing.T) {
	s, original := testImageServer(t)
	generated := addReferenceFixture(t, s, "assistant", color.RGBA{G: 255, A: 255}, 8, 8)
	items, err := s.sessionImageAttachments("session")
	if err != nil {
		t.Fatal(err)
	}
	turn := &turnImages{sessionID: "session", items: map[string]db.Attachment{}, requiredOriginals: map[string]bool{original.ID: true}}
	ctx := context.WithValue(context.Background(), turnImageKey{}, turn)
	cases := []imageGenerationArgs{
		{Operation: "generate"},
		{Operation: "identity_edit", SourceImageID: generated.ID},
		{Operation: "reference_generate"},
		{Operation: "reference_generate", ReferenceImages: []imageReference{{"other-session-id", "character"}}},
		{Operation: "reference_generate", ReferenceImages: []imageReference{{original.ID, "character"}, {original.ID, "duplicate"}}},
		{Operation: "generate", ReferenceImages: []imageReference{{original.ID, "silently ignored reference"}}},
		{Operation: "reference_generate", ReferenceImages: []imageReference{{original.ID, ""}}},
	}
	for _, args := range cases {
		if _, err := s.validateImageReferences(ctx, "session", "paint", args, items); err == nil {
			t.Errorf("invalid reference request accepted: %+v", args)
		}
	}
	args := imageGenerationArgs{Operation: "identity_edit", SourceImageID: generated.ID, ReferenceImages: []imageReference{{original.ID, "original character identity"}}}
	inputs, err := s.validateImageReferences(ctx, "session", "paint", args, items)
	if err != nil || len(inputs) != 2 || inputs[0].Role != "source" || inputs[1].Role != "reference" {
		t.Fatalf("valid base plus original rejected: %+v %v", inputs, err)
	}
	turn.referencesSatisfied = true
	if _, err := s.validateImageReferences(ctx, "session", "paint", imageGenerationArgs{Operation: "identity_edit", SourceImageID: generated.ID}, items); err != nil {
		t.Fatalf("subsequent requested chain edit rejected: %v", err)
	}
}

func TestImageFailedReferenceEditNeverFallsBackToGeneration(t *testing.T) {
	s, source := testImageServer(t)
	calls := 0
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls++; http.Error(w, "reference API failed", 422) }))
	defer worker.Close()
	args, _ := json.Marshal(map[string]any{"operation": "reference_generate", "prompt": "same character", "reference_images": []imageReference{{source.ID, "original character"}}})
	_, err := s.executeImageGenerateTool(context.Background(), "session", config.ImageConfig{Endpoint: worker.URL, Model: "test", Mode: "reference", DefaultSize: "512x512", Timeout: "2s"}, llm.ToolCall{Function: llm.FunctionCall{Arguments: string(args)}}, func(string, any) error { return nil })
	if err == nil || calls != 1 {
		t.Fatalf("failed edit hidden or retried: calls=%d err=%v", calls, err)
	}
}

func TestImageAttachmentLabelsStayAdjacentAndPreserveUploadOrder(t *testing.T) {
	s, first := testImageServer(t)
	second := addReferenceFixture(t, s, "user", color.RGBA{B: 255, A: 255}, 12, 20)
	messages, err := s.llmMessages(context.Background(), []db.Message{{Role: "user", Content: "use both", Attachments: []db.Attachment{first, second}}}, config.Config{})
	if err != nil {
		t.Fatal(err)
	}
	parts := messages[0].Content.([]map[string]any)
	if len(parts) != 5 || !strings.Contains(parts[0]["text"].(string), first.ID) || parts[1]["type"] != "image_url" || !strings.Contains(parts[2]["text"].(string), second.ID) || parts[3]["type"] != "image_url" {
		t.Fatalf("image labels separated from their images: %+v", parts)
	}
	catalog := imageAttachmentCatalog(s, "session")
	if strings.Index(catalog, first.ID) > strings.Index(catalog, second.ID) || !strings.Contains(catalog, "origin=user") {
		t.Fatal("catalog lost original upload order/provenance")
	}
	for _, text := range []string{"올린 사진 두 장을 반영해라", "use these reference photos"} {
		if _, err := s.db.AddMessage("session", "user", text, "", nil, nil); err != nil {
			t.Fatal(err)
		}
		if !requestedOriginalImages(s, "session")[second.ID] {
			t.Fatalf("historical original missing for %q", text)
		}
	}
	if _, err := s.db.AddMessage("session", "user", "참조 없이 새로 그려라", "", nil, nil); err != nil {
		t.Fatal(err)
	}
	if len(requestedOriginalImages(s, "session")) != 0 {
		t.Fatal("explicit no-reference generation rejected")
	}
}

func TestImageReferenceEditPutsBaseBeforeAdditionalIdentities(t *testing.T) {
	s, base := testImageServer(t)
	ref := addReferenceFixture(t, s, "user", color.RGBA{B: 255, A: 255}, 12, 20)
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Error(err)
			return
		}
		defer r.MultipartForm.RemoveAll()
		for i, id := range []string{base.ID, ref.ID} {
			f, err := r.MultipartForm.File["image"][i].Open()
			if err != nil {
				t.Error(err)
				return
			}
			got, _ := io.ReadAll(f)
			f.Close()
			expected, _ := s.media.Open(map[string]db.Attachment{base.ID: base, ref.ID: ref}[id])
			want, _ := io.ReadAll(expected)
			expected.Close()
			if !bytes.Equal(got, want) {
				t.Errorf("base/reference image swapped at %d", i)
			}
		}
		json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString(onePixelPNG)}}})
	}))
	defer worker.Close()
	args, _ := json.Marshal(map[string]any{"operation": "identity_edit", "source_image_id": base.ID, "source_image_description": "base scene", "reference_images": []imageReference{{ref.ID, "identity for right subject"}}, "prompt": "Change right subject only."})
	_, err := s.executeImageGenerateTool(context.Background(), "session", config.ImageConfig{Endpoint: worker.URL, Model: "test", Mode: "paint", DefaultSize: "512x512", Timeout: "2s"}, llm.ToolCall{Function: llm.FunctionCall{Arguments: string(args)}}, func(string, any) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
}
