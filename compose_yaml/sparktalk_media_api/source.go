package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"mime"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const maxSourceRequestBytes = 64 * 1024

type sourceRequest struct {
	URL                string `json:"url"`
	MaxDownloadMB      int64  `json:"max_download_mb,omitempty"`
	MaxDurationSeconds int64  `json:"max_duration_seconds,omitempty"`
	MaxHeight          int    `json:"max_height,omitempty"`
}

type sourceInfo struct {
	ID           string         `json:"id,omitempty"`
	Title        string         `json:"title,omitempty"`
	Description  string         `json:"description,omitempty"`
	Duration     float64        `json:"duration,omitempty"`
	WebpageURL   string         `json:"webpage_url,omitempty"`
	Extractor    string         `json:"extractor,omitempty"`
	ExtractorKey string         `json:"extractor_key,omitempty"`
	Thumbnail    string         `json:"thumbnail,omitempty"`
	LiveStatus   string         `json:"live_status,omitempty"`
	Language     string         `json:"language,omitempty"`
	Formats      []sourceFormat `json:"formats,omitempty"`
}

type sourceFormat struct {
	ID             string  `json:"format_id,omitempty"`
	Extension      string  `json:"ext,omitempty"`
	Width          int     `json:"width,omitempty"`
	Height         int     `json:"height,omitempty"`
	FPS            float64 `json:"fps,omitempty"`
	AudioCodec     string  `json:"acodec,omitempty"`
	VideoCodec     string  `json:"vcodec,omitempty"`
	FileSize       int64   `json:"filesize,omitempty"`
	FileSizeApprox int64   `json:"filesize_approx,omitempty"`
	Language       string  `json:"language,omitempty"`
	LanguagePref   int     `json:"language_preference,omitempty"`
	FormatNote     string  `json:"format_note,omitempty"`
}

func ytDLPVersion(path string) (string, error) {
	out, err := execOutput(path, "--version")
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

func execOutput(path string, args ...string) ([]byte, error) {
	stdout, stderr, err := run(context.Background(), path, args...)
	if err != nil {
		return nil, fmt.Errorf("%w: %s", err, strings.TrimSpace(string(stderr)))
	}
	return stdout, nil
}

func (a *api) probeSource(w http.ResponseWriter, r *http.Request) {
	a.withSource(w, r, func(ctx context.Context, request sourceRequest, rawURL string, _ string) error {
		info, err := a.sourceMetadata(ctx, rawURL)
		if err != nil {
			return err
		}
		writeJSON(w, http.StatusOK, info)
		return nil
	})
}

func (a *api) downloadSource(w http.ResponseWriter, r *http.Request) {
	a.withSource(w, r, func(ctx context.Context, request sourceRequest, rawURL, dir string) error {
		maxDownloadMB := boundedLimit(request.MaxDownloadMB, a.cfg.MaxDownloadMB)
		maxDurationSec := boundedLimit(request.MaxDurationSeconds, a.cfg.MaxDurationSec)
		maxHeight := boundedIntLimit(request.MaxHeight, a.cfg.MaxVideoHeight)
		info, err := a.sourceMetadata(ctx, rawURL)
		if err != nil {
			return err
		}
		if info.LiveStatus == "is_live" {
			return &httpError{Status: http.StatusUnprocessableEntity, Message: "live streams are not supported"}
		}
		if info.Duration > 0 && info.Duration > float64(maxDurationSec) {
			return &httpError{Status: http.StatusRequestEntityTooLarge, Message: fmt.Sprintf("media duration exceeds %d seconds", maxDurationSec)}
		}

		format, selectedHeight := selectDownloadFormat(info, maxDownloadMB, maxHeight)
		log.Printf("yt-dlp selected format=%s height=%d source=%s", format, selectedHeight, info.ID)
		outputTemplate := filepath.Join(dir, "source.%(ext)s")
		maxSize := strconv.FormatInt(maxDownloadMB, 10) + "M"
		_, stderr, err := run(ctx, a.cfg.YtDLPPath,
			"--no-playlist", "--no-progress", "--no-warnings", "--no-part",
			"--max-filesize", maxSize,
			"--format", format,
			"--merge-output-format", "mp4",
			"--output", outputTemplate,
			"--", rawURL)
		if err != nil {
			return processError("yt-dlp", err, stderr)
		}
		path, err := downloadedSourcePath(dir)
		if err != nil {
			return err
		}
		if selectedHeight > 0 {
			path, err = a.remuxVideo(ctx, path)
			if err != nil {
				return err
			}
		}
		stat, err := os.Stat(path)
		if err != nil {
			return err
		}
		if stat.Size() > maxDownloadMB*1024*1024 {
			return &httpError{Status: http.StatusRequestEntityTooLarge, Message: "download exceeds configured limit"}
		}
		contentType := mime.TypeByExtension(strings.ToLower(filepath.Ext(path)))
		if contentType == "" {
			file, err := os.Open(path)
			if err != nil {
				return err
			}
			var header [512]byte
			n, _ := file.Read(header[:])
			_ = file.Close()
			contentType = http.DetectContentType(header[:n])
		}
		w.Header().Set("X-Media-Title", url.QueryEscape(info.Title))
		w.Header().Set("X-Media-Source-ID", info.ID)
		return serveFile(w, path, contentType, filepath.Base(path))
	})
}

func (a *api) remuxVideo(ctx context.Context, input string) (string, error) {
	output := filepath.Join(filepath.Dir(input), "ready.mp4")
	_, stderr, err := run(ctx, a.cfg.FFmpegPath,
		"-nostdin", "-hide_banner", "-loglevel", "error", "-y",
		"-i", input, "-map", "0:v:0", "-map", "0:a:0?",
		"-c", "copy", "-shortest", "-avoid_negative_ts", "make_zero",
		"-movflags", "+faststart", output)
	if err != nil {
		return "", processError("ffmpeg remux", err, stderr)
	}
	return output, nil
}

func (a *api) withSource(w http.ResponseWriter, r *http.Request, fn func(context.Context, sourceRequest, string, string) error) {
	var request sourceRequest
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxSourceRequestBytes))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&request); err != nil {
		writeError(w, &httpError{Status: http.StatusBadRequest, Message: "body must be JSON containing a url"})
		return
	}
	rawURL, err := validateSourceURL(r.Context(), request.URL, net.DefaultResolver.LookupIPAddr)
	if err != nil {
		writeError(w, &httpError{Status: http.StatusBadRequest, Message: err.Error()})
		return
	}

	select {
	case a.sem <- struct{}{}:
		defer func() { <-a.sem }()
	case <-r.Context().Done():
		writeError(w, &httpError{Status: 499, Message: "request canceled"})
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), a.cfg.ProcessTimeout)
	defer cancel()
	dir, err := os.MkdirTemp(a.cfg.TempDir, "source-")
	if err != nil {
		writeError(w, fmt.Errorf("create source directory: %w", err))
		return
	}
	defer os.RemoveAll(dir)
	if err := fn(ctx, request, rawURL, dir); err != nil {
		writeError(w, err)
	}
}

func boundedLimit(requested, maximum int64) int64 {
	if requested <= 0 || requested > maximum {
		return maximum
	}
	return requested
}

func boundedIntLimit(requested, maximum int) int {
	if requested <= 0 || requested > maximum {
		return maximum
	}
	return requested
}

// selectDownloadFormat chooses the highest resolution whose video and audio
// estimates fit together. yt-dlp's --max-filesize is per stream and cannot
// prevent two individually small streams from exceeding the final limit.
func selectDownloadFormat(info sourceInfo, maxDownloadMB int64, maxHeight int) (string, int) {
	limit := maxDownloadMB * 1024 * 1024 * 98 / 100
	hasVideo := false
	for _, format := range info.Formats {
		if format.Height > 0 && format.VideoCodec != "" && format.VideoCodec != "none" {
			hasVideo = true
			break
		}
	}
	if !hasVideo {
		return "bestaudio/best", 0
	}
	preferredLanguage, preferredLanguageValue, hasLanguagePreference := sourceAudioPreference(info)
	bestFormat, bestHeight, bestScore := "", 0, int64(-1)
	for _, video := range info.Formats {
		videoSize := knownFormatSize(video)
		if video.Height < 1 || video.Height > maxHeight || videoSize < 1 || video.VideoCodec == "none" {
			continue
		}
		if video.AudioCodec != "" && video.AudioCodec != "none" {
			if videoSize <= limit {
				score := formatScore(video, sourceFormat{})
				if score > bestScore {
					bestFormat, bestHeight, bestScore = video.ID, video.Height, score
				}
			}
			continue
		}
		for _, audio := range info.Formats {
			audioSize := knownFormatSize(audio)
			if audioSize < 1 || audio.VideoCodec != "none" || audio.AudioCodec == "" || audio.AudioCodec == "none" || videoSize+audioSize > limit {
				continue
			}
			if hasLanguagePreference && audio.LanguagePref != preferredLanguageValue {
				continue
			}
			if !hasLanguagePreference && preferredLanguage != "" && audio.Language != "" && audio.Language != preferredLanguage {
				continue
			}
			score := formatScore(video, audio)
			if score > bestScore {
				bestFormat, bestHeight, bestScore = video.ID+"+"+audio.ID, video.Height, score
			}
		}
	}
	if bestFormat != "" {
		return bestFormat, bestHeight
	}
	return fmt.Sprintf("best[height<=%d][filesize<%dM]/best[height<=%d]", maxHeight, maxDownloadMB, maxHeight), maxHeight
}

func sourceAudioPreference(info sourceInfo) (string, int, bool) {
	preferredLanguage := strings.TrimSpace(info.Language)
	maxPreference := 0
	hasPreference := false
	for _, format := range info.Formats {
		if format.VideoCodec != "none" || format.AudioCodec == "" || format.AudioCodec == "none" {
			continue
		}
		if !hasPreference || format.LanguagePref > maxPreference {
			maxPreference = format.LanguagePref
			hasPreference = true
		}
	}
	// A list where every format has the zero default carries no useful
	// preference. In that case the video's declared primary language is used.
	if hasPreference && maxPreference == 0 {
		hasPreference = false
	}
	return preferredLanguage, maxPreference, hasPreference
}

func knownFormatSize(format sourceFormat) int64 {
	if format.FileSize > 0 {
		return format.FileSize
	}
	return format.FileSizeApprox
}

func formatScore(video, audio sourceFormat) int64 {
	score := int64(video.Height) * 1_000_000_000
	codec := strings.ToLower(video.VideoCodec)
	if video.Extension == "mp4" {
		score += 100_000_000
	}
	if strings.HasPrefix(codec, "avc1") || strings.HasPrefix(codec, "h264") {
		score += 50_000_000
	}
	if audio.Extension == "m4a" || strings.HasPrefix(strings.ToLower(audio.AudioCodec), "mp4a") {
		score += 10_000_000
	}
	if strings.Contains(strings.ToLower(audio.FormatNote), "drc") || strings.Contains(strings.ToLower(audio.ID), "drc") {
		score -= 5_000_000
	}
	score += knownFormatSize(audio) / 1024
	return score
}

func (a *api) sourceMetadata(ctx context.Context, rawURL string) (sourceInfo, error) {
	stdout, stderr, err := run(ctx, a.cfg.YtDLPPath,
		"--dump-single-json", "--skip-download", "--no-playlist", "--no-warnings", "--", rawURL)
	if err != nil {
		return sourceInfo{}, processError("yt-dlp", err, stderr)
	}
	var info sourceInfo
	if err := json.Unmarshal(stdout, &info); err != nil {
		return sourceInfo{}, &httpError{Status: http.StatusUnprocessableEntity, Message: "yt-dlp returned invalid metadata"}
	}
	if info.ID == "" && info.Title == "" {
		return sourceInfo{}, &httpError{Status: http.StatusUnprocessableEntity, Message: "yt-dlp found no media"}
	}
	if info.Duration > 0 && info.Duration > float64(a.cfg.MaxDurationSec) {
		return sourceInfo{}, &httpError{Status: http.StatusRequestEntityTooLarge, Message: fmt.Sprintf("media duration exceeds %d seconds", a.cfg.MaxDurationSec)}
	}
	return info, nil
}

func downloadedSourcePath(dir string) (string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", err
	}
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasPrefix(entry.Name(), "source.") || strings.HasSuffix(entry.Name(), ".part") || strings.HasSuffix(entry.Name(), ".ytdl") {
			continue
		}
		return filepath.Join(dir, entry.Name()), nil
	}
	return "", &httpError{Status: http.StatusUnprocessableEntity, Message: "yt-dlp completed without a media file"}
}

type ipLookup func(context.Context, string) ([]net.IPAddr, error)

func validateSourceURL(ctx context.Context, raw string, lookup ipLookup) (string, error) {
	raw = strings.TrimSpace(raw)
	parsed, err := url.Parse(raw)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Hostname() == "" {
		return "", errors.New("url must use http or https")
	}
	if parsed.User != nil {
		return "", errors.New("url credentials are not allowed")
	}
	if ip := net.ParseIP(parsed.Hostname()); ip != nil {
		if !isPublicIP(ip) {
			return "", errors.New("local and private network urls are not allowed")
		}
		return parsed.String(), nil
	}
	addresses, err := lookup(ctx, parsed.Hostname())
	if err != nil || len(addresses) == 0 {
		return "", errors.New("url host could not be resolved")
	}
	for _, address := range addresses {
		if !isPublicIP(address.IP) {
			return "", errors.New("local and private network urls are not allowed")
		}
	}
	return parsed.String(), nil
}

func isPublicIP(ip net.IP) bool {
	return ip != nil && !ip.IsLoopback() && !ip.IsPrivate() && !ip.IsLinkLocalUnicast() &&
		!ip.IsLinkLocalMulticast() && !ip.IsUnspecified() && !ip.IsMulticast()
}
