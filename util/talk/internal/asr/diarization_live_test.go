package asr

import (
	"context"
	"os"
	"sparktalk/internal/config"
	"testing"
	"time"
)

// Explicit opt-in: runs the real model and Extra Media, not part of offline CI.
func TestLiveDiarization(t *testing.T) {
	path := os.Getenv("TALK_DIAR_LIVE_AUDIO")
	if path == "" {
		t.Skip("set TALK_DIAR_LIVE_AUDIO and TALK_DIAR_LIVE_ENDPOINT")
	}
	endpoint := os.Getenv("TALK_DIAR_LIVE_ENDPOINT")
	if endpoint == "" {
		t.Fatal("endpoint required")
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	client := New(config.ASRConfig{Enabled: true, Diarization: true, Endpoint: endpoint, FFmpegEndpoint: "http://127.0.0.1:8690", MediaLanguage: "auto", Timeout: "15m"})
	start := time.Now()
	r, err := client.Transcribe(context.Background(), f, "fixture.wav", "audio/wav")
	if err != nil {
		t.Fatal(err)
	}
	speakers := map[int]bool{}
	for _, turn := range r.Turns {
		for _, id := range turn.Speakers {
			speakers[id] = true
		}
	}
	t.Logf("elapsed=%s text_chars=%d words=%d turns=%d speakers=%d status=%s warning=%s", time.Since(start), len([]rune(r.Text)), len(r.Words), len(r.Turns), len(speakers), r.DiarizationStatus, r.Warning)
	if r.DiarizationStatus != "completed" || len(r.Turns) == 0 || len(speakers) == 0 || r.Text == "" {
		t.Fatal("diarization did not complete")
	}
	if os.Getenv("TALK_DIAR_REQUIRE_MULTI") == "1" && len(speakers) < 2 {
		t.Fatal("multiple meeting speakers were not distinguished")
	}
}
