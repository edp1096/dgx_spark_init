package server

import (
	"context"
	"net/http/httptest"
	"sparktalk/internal/asr"
	"sparktalk/internal/config"
	"sparktalk/internal/media"
	"strings"
	"testing"
)

func TestSpeakerTranscriptCacheAndDisplay(t *testing.T) {
	store, _ := media.New(t.TempDir() + "/chat.db")
	item, err := store.SaveReader(strings.NewReader("ID3audio"), "voice.mp3", "audio/mpeg", media.MaxAttachmentBytes)
	if err != nil {
		t.Fatal(err)
	}
	cfg := config.ASRConfig{Diarization: true}
	cached := media.TranscriptCache{Text: "hello", Fingerprint: transcriptFingerprint(cfg), DiarizationStatus: "completed", Turns: []asr.Turn{{Start: 0, End: 1, Speakers: []int{1}, Text: "hello"}}}
	if err = store.SaveTranscript(item.ID, cached); err != nil {
		t.Fatal(err)
	}
	s := &Server{media: store}
	actual, err := s.transcribeAttachment(context.Background(), item, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(transcriptBlock(item, actual), "화자 1: hello") {
		t.Fatal("speaker not passed to model")
	}
	rec := httptest.NewRecorder()
	s.attachmentTranscript(rec, httptest.NewRequest("GET", "/api/media/transcript/"+item.ID, nil))
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"speakers":[1]`) {
		t.Fatalf("view %d %s", rec.Code, rec.Body.String())
	}
	cfg.Diarization = false
	if _, ok, _ := store.LoadTranscript(item.ID, transcriptFingerprint(cfg)); ok {
		t.Fatal("diarization toggle reused stale cache")
	}
}
