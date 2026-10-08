package server

import (
	"mediaapp/internal/config"
	"testing"
)

func TestSpeechJobParamsRoundTripAndLegacyDefaults(t *testing.T) {
	cfg := config.Speech{DefaultLanguage: "Korean", DefaultSpeaker: "Sohee"}
	params := speechJobParams{
		Language: "Japanese", Speaker: "Aoi", Instructions: "calm", Seed: 42,
		Stage: "queued", QueuedAt: "2026-08-27T00:00:00Z",
	}
	decoded := decodeSpeechJobParams(params.toMap(), cfg)
	if decoded.Language != "Japanese" || decoded.Speaker != "Aoi" || decoded.Seed != 42 {
		t.Fatalf("speech contract did not round-trip: %#v", decoded)
	}
	legacy := decodeSpeechJobParams(map[string]any{}, cfg)
	if legacy.Language != "Korean" || legacy.Speaker != "Sohee" || legacy.Seed != -1 {
		t.Fatalf("speech legacy defaults were not applied: %#v", legacy)
	}
}

func TestSubtitleJobParamsRoundTripAndLegacyDefaults(t *testing.T) {
	cfg := config.Recognition{
		DefaultLanguage: "Auto", DefaultOutputFormats: []string{"srt", "txt"},
		DefaultTranslationMode: "none", DefaultTranslationLanguage: "Korean",
	}
	params := subtitleJobParams{
		Language: "Japanese", Context: "period drama", Source: "video_job", SourceJobID: "video-1",
		OutputFormats: []string{"vtt", "txt"}, TranslationMode: "bilingual", TargetLanguage: "Korean",
		MediaPart: "1", MediaSource: "main", Stage: "queued", QueuedAt: "2026-08-27T00:00:00Z",
	}
	decoded := decodeSubtitleJobParams(params.toMap(), cfg)
	if decoded.Source != "video_job" || decoded.SourceJobID != "video-1" || decoded.Language != "Japanese" {
		t.Fatalf("subtitle source contract did not round-trip: %#v", decoded)
	}
	if decoded.TranslationMode != "bilingual" || len(decoded.OutputFormats) != 2 || decoded.MediaPart != "1" {
		t.Fatalf("subtitle output contract did not round-trip: %#v", decoded)
	}
	legacy := decodeSubtitleJobParams(map[string]any{}, cfg)
	if legacy.Language != "Auto" || legacy.Source != "file" || legacy.TranslationMode != "none" || len(legacy.OutputFormats) != 2 {
		t.Fatalf("subtitle legacy defaults were not applied: %#v", legacy)
	}
}
