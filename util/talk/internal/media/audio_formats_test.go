package media

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRealReferenceAudioFormats(t *testing.T) {
	dir := os.Getenv("SPARKTALK_VOICE_FORMAT_FIXTURES")
	if dir == "" {
		t.Skip("set SPARKTALK_VOICE_FORMAT_FIXTURES to real ffmpeg-generated audio fixtures")
	}
	files, err := filepath.Glob(filepath.Join(dir, "sample.*"))
	if err != nil || len(files) < 12 {
		t.Fatal("missing codec fixtures", err)
	}
	for _, file := range files {
		t.Run(filepath.Base(file), func(t *testing.T) {
			data, err := os.ReadFile(file)
			if err != nil {
				t.Fatal(err)
			}
			mime, err := classifyMedia(data, filepath.Base(file), "application/octet-stream", false)
			if err != nil || !strings.HasPrefix(mime, "audio/") {
				t.Fatalf("%s: %s %v", file, mime, err)
			}
			// Formats with unambiguous audio magic also work without a useful filename.
			if !strings.HasSuffix(file, ".wma") && !strings.HasSuffix(file, ".mka") && !strings.HasSuffix(file, ".m4a") {
				wrong, err := classifyMedia(data, "unknown.txt", "application/octet-stream", false)
				if err != nil || !strings.HasPrefix(wrong, "audio/") {
					t.Fatal("magic lost to extension", wrong, err)
				}
			}
		})
	}
}
