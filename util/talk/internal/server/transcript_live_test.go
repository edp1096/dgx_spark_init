package server

import (
	"context"
	"os"
	"path/filepath"
	"sparktalk/internal/asr"
	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/media"
	"sparktalk/internal/orchestrator"
	"testing"
	"time"
)

func TestAttachmentTranscriptLive(t *testing.T) {
	path, id := os.Getenv("TALK_TRANSCRIPT_LIVE_CONFIG"), os.Getenv("TALK_TRANSCRIPT_LIVE_ATTACHMENT")
	if path == "" || id == "" {
		t.Skip("real transcript validation not requested")
	}
	cfg, _, err := config.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	store, err := media.New(filepath.Join(filepath.Dir(path), "sparktalk.db"), cfg.Attachments)
	if err != nil {
		t.Fatal(err)
	}
	controller, err := orchestrator.NewControllerWithCatalog(*cfg.Runtime.Catalog)
	if err != nil {
		t.Fatal(err)
	}
	defer controller.Close()
	controller.ConfigurePaths(cfg.Runtime.DataDir, cfg.Runtime.ModelCache)
	s := &Server{cfg: cfg, media: store, asr: asr.New(cfg.ASR), runtime: controller}
	item := db.Attachment{ID: id, Name: "MiniMax-H3.mp4", MIME: "video/mp4"}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	result, err := s.transcribeAttachment(ctx, item, cfg.ASR)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok, err := store.LoadTranscript(id, transcriptFingerprint(cfg.ASR)); err != nil || !ok {
		t.Fatal("result not cached", err)
	}
	again, err := s.transcribeAttachment(ctx, item, cfg.ASR)
	if err != nil || result.Status != again.Status || result.Text != again.Text {
		t.Fatal("cache read differs", err)
	}
	t.Logf("status=%s recognized_characters=%d warning=%s; second call used cached result", result.Status, len([]rune(result.Text)), result.Warning)
}
