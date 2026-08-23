package server

import (
	"bytes"
	"image"
	"testing"
	"time"

	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
)

func TestPNGEXIFMetadataRoundTrip(t *testing.T) {
	createdAt := time.Date(2026, 8, 23, 16, 20, 0, 0, time.Local)
	job := jobs.Job{
		ID: "image-metadata", Kind: "image", Prompt: "붉은 소파 위의 고양이", CreatedAt: createdAt,
		Params: map[string]any{
			"model": "krea2-turbo-nvfp4", "mode": "create", "width": 32, "height": 24,
			"seed": int64(42), "styles": []any{map[string]any{"name": "film", "strength": 0.8}},
		},
	}
	original := testPNG(t, 32, 24)
	profile := config.ImageMetadata{Creator: "홍길동", Copyright: "© 2026 Hong", Website: "https://example.com", Note: "개인 작업"}
	embedded := embedImageEXIF(original, metadataForImageJob(job, "a red cat on a sofa", profile))
	if bytes.Equal(original, embedded) {
		t.Fatal("PNG was not changed")
	}
	if config, _, err := image.DecodeConfig(bytes.NewReader(embedded)); err != nil || config.Width != 32 || config.Height != 24 {
		t.Fatalf("embedded PNG is invalid: config=%#v err=%v", config, err)
	}
	metadata, ok := extractImageEXIF(embedded)
	if !ok {
		t.Fatal("embedded EXIF was not found")
	}
	if metadata.JobID != job.ID || metadata.Prompt != job.Prompt || metadata.EffectivePrompt != "a red cat on a sofa" {
		t.Fatalf("metadata=%#v", metadata)
	}
	if metadata.Model != "krea2-turbo-nvfp4" || metadata.Mode != "create" || metadata.Width != 32 || metadata.Height != 24 {
		t.Fatalf("generation fields=%#v", metadata)
	}
	if metadata.Creator != profile.Creator || metadata.Copyright != profile.Copyright || metadata.Website != profile.Website || metadata.Note != profile.Note {
		t.Fatalf("creator profile=%#v", metadata)
	}
}

func TestPNGEXIFLeavesNonPNGUnchanged(t *testing.T) {
	original := []byte("not a png")
	if got := embedImageEXIF(original, imageEXIFMetadata{Prompt: "test"}); !bytes.Equal(got, original) {
		t.Fatalf("non-PNG changed: %q", got)
	}
}
