package server

import (
	"encoding/base64"
	"mediaapp/internal/config"
	"os"
	"path/filepath"
	"testing"
)

func TestBuildKreaRequestEncodesSequenceReIDReference(t *testing.T) {
	path := filepath.Join(t.TempDir(), "character.png")
	data := []byte("character reference")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatal(err)
	}
	request, err := buildKreaRequest(
		config.ImageBackend{Model: "krea-test"}, "same character in a new scene", 1024, 1024, 42,
		imageGenerationOptions{checkpoint: "official", reidPath: path, steps: 8, filterMode: "balanced"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if request["reid_image"] != base64.StdEncoding.EncodeToString(data) {
		t.Fatalf("ReID reference was not encoded: %#v", request)
	}
	if request["source_image"] != nil {
		t.Fatalf("ReID must not be sent as Identity Edit source: %#v", request)
	}
	if request["prompt"] != "same character in a new scene" || request["steps"] != 8 {
		t.Fatalf("unexpected request: %#v", request)
	}
}
