package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/media"
)

func TestImportedAudioProvenanceSurvivesConversationStorage(t *testing.T) {
	remote := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "audio/mpeg")
		w.Header().Set("Content-Disposition", `attachment; filename="original.mp3"`)
		w.Header().Set("X-Media-Audio-Language", "en-US")
		w.Header().Set("X-Media-Audio-Format", "140-20")
		w.Header().Set("X-Media-Audio-Original", "true")
		_, _ = w.Write([]byte("ID3sample audio"))
	}))
	defer remote.Close()
	s := titleTestServer(t)
	var err error
	s.media, err = media.New(t.TempDir() + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	s.cfg = config.Config{ASR: config.ASRConfig{FFmpegEndpoint: remote.URL, Timeout: "5s"}}
	item, err := s.importMediaSource(context.Background(), "https://example.com/watch")
	if err != nil {
		t.Fatal(err)
	}
	_, err = s.db.AddMessage("current", "user", "원본 음성만 분석해라.", "", nil, []db.Attachment{item})
	if err != nil {
		t.Fatal(err)
	}
	messages, err := s.db.Messages("current")
	if err != nil || len(messages) != 1 || len(messages[0].Attachments) != 1 {
		t.Fatal(messages, err)
	}
	saved := messages[0].Attachments[0]
	if !saved.SourceAudioOriginal || saved.SourceAudioLanguage != "en-US" || saved.SourceAudioFormat != "140-20" {
		t.Fatalf("lost audio provenance: %+v", saved)
	}
}
