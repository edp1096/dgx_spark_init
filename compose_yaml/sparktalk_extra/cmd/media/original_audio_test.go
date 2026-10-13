package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestOriginalAudioAppliesToMuxedAndAudioOnlyFormats(t *testing.T) {
	original := sourceFormat{ID: "en-original", VideoCodec: "none", AudioCodec: "mp4a", Language: "en-US", LanguagePref: 10, FileSize: 4 << 20}
	dub := sourceFormat{ID: "ko-default", VideoCodec: "none", AudioCodec: "mp4a", Language: "ko", LanguagePref: 5, FileSize: 5 << 20}
	video := sourceFormat{ID: "video", VideoCodec: "avc1", AudioCodec: "none", Height: 480, FileSize: 20 << 20}
	mux := sourceFormat{ID: "dubbed-mux", VideoCodec: "avc1", AudioCodec: "mp4a", Language: "ko", LanguagePref: 5, Height: 720, FileSize: 24 << 20}
	for _, tc := range []struct {
		name    string
		formats []sourceFormat
		want    string
	}{
		{"audio-only", []sourceFormat{dub, original}, "en-original"},
		{"generic-audio-unknown-codec", []sourceFormat{{ID: "podcast", Extension: "mp3"}}, "podcast"},
		{"muxed-dub-must-not-bypass", []sourceFormat{video, dub, original, mux}, "video+en-original"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, _ := selectDownloadFormat(sourceInfo{Language: "ko", Formats: tc.formats}, 64, 720)
			if got != tc.want {
				t.Fatalf("selected %q; want %q", got, tc.want)
			}
		})
	}
	// A marked muxed original is valid even when a higher-quality dub exists.
	mux.ID, mux.Language, mux.LanguagePref, mux.FormatNote = "original-mux", "en-US", 0, "English (original)"
	if got, _ := selectDownloadFormat(sourceInfo{Formats: []sourceFormat{mux, dub}}, 64, 720); got != "original-mux" {
		t.Fatal(got)
	}
}

func TestOriginalAudioDoesNotFallBackToDubWhenTooLarge(t *testing.T) {
	info := sourceInfo{Formats: []sourceFormat{
		{ID: "v", VideoCodec: "avc1", AudioCodec: "none", Height: 480, FileSize: 20 << 20},
		{ID: "original", VideoCodec: "none", AudioCodec: "mp4a", LanguagePref: 10, FileSize: 60 << 20},
		{ID: "dub", VideoCodec: "none", AudioCodec: "mp4a", LanguagePref: 5, FileSize: 1 << 20},
	}}
	if got, _ := selectDownloadFormat(info, 64, 720); got != "" {
		t.Fatal("substituted dub:", got)
	}
}

func TestAmbiguousOrDubbedAudioIsNotAssumedOriginal(t *testing.T) {
	info := sourceInfo{Formats: []sourceFormat{
		{ID: "a", VideoCodec: "none", AudioCodec: "mp4a", Language: "en", LanguagePref: 5},
		{ID: "b", VideoCodec: "none", AudioCodec: "mp4a", Language: "ko"},
	}}
	if got, _ := selectDownloadFormat(info, 64, 720); got != "" {
		t.Fatal("guessed original language:", got)
	}
	info.Language = "ko"
	if got, _ := selectDownloadFormat(info, 64, 720); got != "b" {
		t.Fatal(got)
	}
	info.Formats[1].FormatNote = "Korean auto-dubbed"
	if got, _ := selectDownloadFormat(info, 64, 720); got != "" {
		t.Fatal("accepted labelled dub:", got)
	}
}

func TestSourceMetadataProjectionWithLargeUnusedFields(t *testing.T) {
	// Uses the actual pinned yt-dlp binary, offline. Its projection must exclude
	// huge subtitle data before it reaches the 8 MiB bounded process capture.
	binary := os.Getenv("YTDLP_TEST_BINARY")
	if binary == "" {
		t.Skip("requires a local yt-dlp binary")
	}
	dir := t.TempDir()
	fixture := filepath.Join(dir, "info.json")
	data, _ := json.Marshal(map[string]any{
		"id": "large", "title": "large", "duration": 10, "extractor": "generic", "webpage_url": "https://example.com/video",
		"automatic_captions": map[string]any{"en": []map[string]any{{"ext": "vtt", "url": "https://example.com/captions.vtt", "name": strings.Repeat("x", 9<<20)}}},
		"formats":            []map[string]any{{"format_id": "audio", "url": "https://example.com/audio.m4a", "ext": "m4a", "vcodec": "none", "acodec": "mp4a", "language": "en", "language_preference": 10, "filesize_approx": 1024.5}},
	})
	if err := os.WriteFile(fixture, data, 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("YTDLP_TEST_FIXTURE", fixture)
	wrapper := filepath.Join(dir, "yt-dlp")
	if err := os.WriteFile(wrapper, []byte("#!/bin/sh\nexec \"$YTDLP_TEST_BINARY\" --load-info-json \"$YTDLP_TEST_FIXTURE\" \"$@\"\n"), 0700); err != nil {
		t.Fatal(err)
	}
	a := &api{cfg: config{YtDLPPath: wrapper, MaxDurationSec: 60}}
	info, err := a.sourceMetadata(context.Background(), "https://example.com/video")
	if err != nil {
		t.Fatal(err)
	}
	if info.ID != "large" || len(info.Formats) != 1 || info.Formats[0].FileSizeApprox != 1025 || !isOriginalAudio(info.Formats[0]) {
		t.Fatalf("lost metadata: %+v", info)
	}
}
