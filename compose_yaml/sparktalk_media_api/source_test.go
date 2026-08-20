package main

import (
	"context"
	"net"
	"os"
	"path/filepath"
	"testing"
)

func TestValidateSourceURL(t *testing.T) {
	lookup := func(_ context.Context, host string) ([]net.IPAddr, error) {
		if host == "media.example" {
			return []net.IPAddr{{IP: net.ParseIP("93.184.216.34")}}, nil
		}
		return []net.IPAddr{{IP: net.ParseIP("127.0.0.1")}}, nil
	}
	tests := []struct {
		name    string
		url     string
		wantErr bool
	}{
		{name: "public https", url: "https://media.example/watch/1"},
		{name: "loopback literal", url: "http://127.0.0.1/video", wantErr: true},
		{name: "private literal", url: "http://192.168.1.20/video", wantErr: true},
		{name: "private dns", url: "https://internal.example/video", wantErr: true},
		{name: "credentials", url: "https://user:pass@media.example/video", wantErr: true},
		{name: "file", url: "file:///etc/passwd", wantErr: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, err := validateSourceURL(context.Background(), test.url, lookup)
			if (err != nil) != test.wantErr {
				t.Fatalf("validateSourceURL() error = %v, wantErr %v", err, test.wantErr)
			}
		})
	}
}

func TestDownloadedSourcePath(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "source.mp4"), []byte("video"), 0o600); err != nil {
		t.Fatal(err)
	}
	path, err := downloadedSourcePath(dir)
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(path) != "source.mp4" {
		t.Fatalf("unexpected path %q", path)
	}
}

func TestBoundedLimit(t *testing.T) {
	if got := boundedLimit(64, 4096); got != 64 {
		t.Fatalf("boundedLimit(64, 4096) = %d", got)
	}
	for _, requested := range []int64{0, -1, 5000} {
		if got := boundedLimit(requested, 4096); got != 4096 {
			t.Fatalf("boundedLimit(%d, 4096) = %d", requested, got)
		}
	}
}

func TestSelectDownloadFormatFitsCombinedLimit(t *testing.T) {
	info := sourceInfo{Language: "ko", Formats: []sourceFormat{
		{ID: "720", Extension: "mp4", Height: 720, VideoCodec: "avc1", AudioCodec: "none", FileSize: 60 << 20},
		{ID: "480", Extension: "mp4", Height: 480, VideoCodec: "avc1", AudioCodec: "none", FileSize: 38 << 20},
		{ID: "low-en", Extension: "m4a", VideoCodec: "none", AudioCodec: "mp4a", Language: "en-US", LanguagePref: -1, FileSize: 9 << 20},
		{ID: "high-en", Extension: "m4a", VideoCodec: "none", AudioCodec: "mp4a", Language: "en-US", LanguagePref: -1, FileSize: 24 << 20},
		{ID: "low-ko", Extension: "m4a", VideoCodec: "none", AudioCodec: "mp4a", Language: "ko", LanguagePref: 10, FileSize: 9 << 20},
		{ID: "high-ko", Extension: "m4a", VideoCodec: "none", AudioCodec: "mp4a", Language: "ko", LanguagePref: 10, FileSize: 24 << 20},
	}}
	format, height := selectDownloadFormat(info, 64, 720)
	if format != "480+high-ko" || height != 480 {
		t.Fatalf("selected %q at %dp", format, height)
	}
}

func TestSelectDownloadFormatPrefersOriginalWithoutDRC(t *testing.T) {
	info := sourceInfo{Language: "ko", Formats: []sourceFormat{
		{ID: "video", Extension: "mp4", Height: 720, VideoCodec: "avc1", AudioCodec: "none", FileSize: 20 << 20},
		{ID: "audio-drc", Extension: "m4a", VideoCodec: "none", AudioCodec: "mp4a", Language: "ko", LanguagePref: 10, FormatNote: "Korean original, DRC", FileSize: 5 << 20},
		{ID: "audio", Extension: "m4a", VideoCodec: "none", AudioCodec: "mp4a", Language: "ko", LanguagePref: 10, FormatNote: "Korean original", FileSize: 5 << 20},
	}}
	format, _ := selectDownloadFormat(info, 64, 720)
	if format != "video+audio" {
		t.Fatalf("selected %q", format)
	}
}

func TestSelectDownloadFormatSupportsAudioOnly(t *testing.T) {
	format, height := selectDownloadFormat(sourceInfo{Formats: []sourceFormat{{ID: "audio", VideoCodec: "none", AudioCodec: "mp4a"}}}, 64, 720)
	if format != "bestaudio/best" || height != 0 {
		t.Fatalf("selected %q at %dp", format, height)
	}
}
