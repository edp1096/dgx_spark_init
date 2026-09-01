package server

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"sparktalk/internal/asr"
	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/media"
)

func (s *Server) transcribeAttachment(ctx context.Context, item db.Attachment, cfg config.ASRConfig) (media.TranscriptCache, error) {
	fingerprint := transcriptFingerprint(cfg)
	if cached, ok, err := s.media.LoadTranscript(item.ID, fingerprint); err == nil && ok {
		return cached, nil
	}

	// The deployed ASR service handles one request at a time. Serializing
	// cache misses also prevents two chat rooms from transcribing the same file.
	s.asrMu.Lock()
	defer s.asrMu.Unlock()
	if cached, ok, err := s.media.LoadTranscript(item.ID, fingerprint); err == nil && ok {
		return cached, nil
	}
	file, err := s.media.Open(item)
	if err != nil {
		return media.TranscriptCache{}, fmt.Errorf("open %s for transcription: %w", item.Name, err)
	}
	defer file.Close()
	client := s.asrSnapshot()
	if client == nil {
		client = asr.New(cfg)
	}
	result, err := client.Transcribe(ctx, file, item.Name, item.MIME)
	if err != nil {
		return media.TranscriptCache{}, fmt.Errorf("transcribe %s: %w", item.Name, err)
	}
	cached := media.TranscriptCache{Fingerprint: fingerprint, Text: result.Text, Language: result.Language}
	if err := s.media.SaveTranscript(item.ID, cached); err != nil {
		return media.TranscriptCache{}, fmt.Errorf("cache transcript for %s: %w", item.Name, err)
	}
	return cached, nil
}

func transcriptFingerprint(cfg config.ASRConfig) string {
	data, _ := json.Marshal(struct {
		Version        int
		FFmpegEndpoint string
		Endpoint       string
		Model          string
		MediaLanguage  string
		Prompt         string
	}{3, cfg.FFmpegEndpoint, cfg.Endpoint, cfg.Model, cfg.MediaLanguage, cfg.Prompt})
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func transcriptBlock(item db.Attachment, cached media.TranscriptCache) string {
	language := strings.TrimSpace(cached.Language)
	if language == "" {
		language = "unknown"
	}
	return fmt.Sprintf("<media_transcript filename=%q language=%q>\n%s\n</media_transcript>", item.Name, language, cached.Text)
}

func isNoAudio(err error) bool { return errors.Is(err, asr.ErrNoAudio) }
