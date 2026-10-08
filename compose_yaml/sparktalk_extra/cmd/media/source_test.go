package main

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSelectHLSWithBitrateAndDuration(t *testing.T) {
	var info sourceInfo
	if err := json.Unmarshal([]byte(`{"duration":35,"formats":[
		{"format_id":"hls720","height":720,"vcodec":"avc1.4d401f","acodec":"none","tbr":5216},
		{"format_id":"hls360","height":360,"vcodec":"avc1.4d401e","acodec":"none","tbr":1149},
		{"format_id":"audio","vcodec":"none","acodec":"mp4a.40.2","tbr":128}
	]}`), &info); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		mb     int64
		format string
		height int
	}{{64, "hls720+audio", 720}, {8, "hls360+audio", 360}, {1, "", 0}} {
		if format, height := selectDownloadFormat(info, tc.mb, 720); format != tc.format || height != tc.height {
			t.Fatalf("budget %d selected %s/%d, want %s/%d", tc.mb, format, height, tc.format, tc.height)
		}
	}
	info.Duration = 0
	if format, _ := selectDownloadFormat(info, 64, 720); format != "" {
		t.Fatal("unknown duration must not be treated as a bounded download")
	}
}

func TestHLSSizeEstimateBounds(t *testing.T) {
	if got := sourceFormatSize(sourceFormat{Bitrate: 128}, 35); got != 616000 {
		t.Fatalf("estimate = %d", got)
	}
	if got := sourceFormatSize(sourceFormat{FileSize: 123, Bitrate: 128}, 35); got != 123 {
		t.Fatalf("declared size must take precedence: %d", got)
	}
	for _, duration := range []float64{0, -1, math.NaN(), math.Inf(1), math.MaxFloat64} {
		if got := sourceFormatSize(sourceFormat{Bitrate: 5216}, duration); got != 0 {
			t.Fatalf("invalid duration %v estimated %d", duration, got)
		}
	}
}

func TestSourceMetadataFloatingByteEstimates(t *testing.T) {
	var info sourceInfo
	if err := json.Unmarshal([]byte(`{"id":"archive","duration":4075,"formats":[
		{"format_id":"video","height":480,"vcodec":"avc1","acodec":"none","filesize_approx":12582912.25},
		{"format_id":"audio","vcodec":"none","acodec":"mp4a","filesize":1.5e6},
		{"format_id":"unknown","filesize":null,"filesize_approx":null}
	]}`), &info); err != nil {
		t.Fatal(err)
	}
	if info.Formats[0].FileSizeApprox != 12582913 || info.Formats[1].FileSize != 1500000 || info.Formats[2].FileSize != 0 {
		t.Fatalf("unexpected byte sizes: %+v", info.Formats)
	}
	if format, height := selectDownloadFormat(info, 64, 720); format != "video+audio" || height != 480 {
		t.Fatalf("selected %q at %dp", format, height)
	}
}

func TestSourceByteSizeBounds(t *testing.T) {
	for _, test := range []struct {
		raw  string
		want int64
	}{
		{"0", 0}, {"12.0", 12}, {"12.01", 13}, {"1.2e2", 120},
		{"9007199254740993", 9007199254740993},
		{"9223372036854775807", 9223372036854775807},
	} {
		got, err := sourceByteSize(json.RawMessage(test.raw))
		if err != nil || got != test.want {
			t.Fatalf("size(%s) = %d, %v; want %d", test.raw, got, err, test.want)
		}
	}
	for _, raw := range []string{"-1", "-0.5", "1e100", "9223372036854775808", `"123"`, "true", "{}"} {
		if _, err := sourceByteSize(json.RawMessage(raw)); err == nil {
			t.Fatalf("accepted invalid size %s", raw)
		}
	}
}

func TestSourceMetadataDecodeFailureReportsCause(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "yt-dlp")
	if err := os.WriteFile(path, []byte("#!/bin/sh\nprintf '%s' '{\"id\":\"broken\",\"formats\":['\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	a := &api{cfg: config{YtDLPPath: path}}
	_, err := a.sourceMetadata(context.Background(), "https://example.com/video")
	var apiErr *httpError
	if !errors.As(err, &apiErr) || !strings.Contains(apiErr.Message, "metadata decode failed") || !strings.Contains(apiErr.Message, "unexpected end of JSON input") {
		t.Fatalf("missing decoder cause: %v", err)
	}
}

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

func TestSelectDownloadFormatPrefersModelSafeH264AtLowerResolution(t *testing.T) {
	info := sourceInfo{Language: "ko", Formats: []sourceFormat{
		{ID: "av1-720", Extension: "mp4", Height: 720, FPS: 60, VideoCodec: "av01.0.08M.08", AudioCodec: "none", FileSize: 26 << 20},
		{ID: "h264-480", Extension: "mp4", Height: 480, FPS: 30, VideoCodec: "avc1.4d401f", AudioCodec: "none", FileSize: 14 << 20},
		{ID: "audio", Extension: "m4a", VideoCodec: "none", AudioCodec: "mp4a.40.2", Language: "ko", LanguagePref: 10, FileSize: 8 << 20},
	}}
	format, height := selectDownloadFormat(info, 64, 720)
	if format != "h264-480+audio" || height != 480 {
		t.Fatalf("selected %q at %dp", format, height)
	}
}

func TestSourceNeedsVideoNormalization(t *testing.T) {
	info := sourceInfo{Formats: []sourceFormat{
		{ID: "safe", VideoCodec: "avc1.4d401f", FPS: 30},
		{ID: "fast", VideoCodec: "avc1.4d4020", FPS: 60},
		{ID: "av1", VideoCodec: "av01.0.08M.08", FPS: 30},
	}}
	if sourceNeedsVideoNormalization(info, "safe+audio") {
		t.Fatal("ordinary H.264 30fps should only be remuxed")
	}
	if !sourceNeedsVideoNormalization(info, "fast+audio") || !sourceNeedsVideoNormalization(info, "av1+audio") {
		t.Fatal("high-fps H.264 and AV1 must be normalized")
	}
}

func TestLongVideoLimitDoesNotFallBackToUnavailableMuxedFormat(t *testing.T) {
	info := sourceInfo{Formats: []sourceFormat{
		{ID: "160", Height: 144, VideoCodec: "avc1", AudioCodec: "none", FileSize: 47 << 20},
		{ID: "135", Height: 480, VideoCodec: "avc1", AudioCodec: "none", FileSize: 446 << 20},
		{ID: "139", VideoCodec: "none", AudioCodec: "mp4a", FileSize: 28 << 20},
	}}
	if format, _ := selectDownloadFormat(info, 64, 720); format != "" {
		t.Fatalf("must report no fit, got %s", format)
	}
	if format, height := selectDownloadFormat(info, 512, 720); format != "135+139" || height != 480 {
		t.Fatalf("got %s height %d", format, height)
	}
}
