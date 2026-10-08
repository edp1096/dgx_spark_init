package main

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
)

// videoFrames produces one bounded contact sheet for image-only VLMs.
func (a *api) videoFrames(w http.ResponseWriter, r *http.Request) {
	a.withInput(w, r, func(ctx context.Context, input string) error {
		out, stderr, err := run(ctx, a.cfg.FFprobePath, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", input)
		if err != nil {
			return processError("ffprobe", err, stderr)
		}
		duration, err := strconv.ParseFloat(strings.TrimSpace(string(out)), 64)
		if err != nil || math.IsNaN(duration) || math.IsInf(duration, 0) || duration <= 0 {
			return fmt.Errorf("video duration unavailable")
		}
		output := filepath.Join(filepath.Dir(input), "frames.jpg")
		if duration > 120 {
			// Seek to each sample instead of decoding hours of intervening frames.
			for i := 0; i < 8; i++ {
				frame := filepath.Join(filepath.Dir(input), fmt.Sprintf("sample-%02d.jpg", i))
				_, stderr, err = run(ctx, a.cfg.FFmpegPath, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-threads", "2",
					"-ss", strconv.FormatFloat(float64(i)*duration/8, 'f', 3, 64), "-i", input, "-map", "0:v:0", "-an", "-sn", "-dn",
					"-vf", "scale=512:512:force_original_aspect_ratio=decrease,pad=512:512:(ow-iw)/2:(oh-ih)/2", "-filter_threads", "1", "-frames:v", "1", "-q:v", "3", frame)
				if err != nil {
					return processError("ffmpeg sample", err, stderr)
				}
			}
			_, stderr, err = run(ctx, a.cfg.FFmpegPath, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-threads", "2",
				"-i", filepath.Join(filepath.Dir(input), "sample-%02d.jpg"), "-vf", "tile=4x2:nb_frames=8", "-filter_threads", "1", "-frames:v", "1", "-q:v", "3", output)
			if err != nil {
				return processError("ffmpeg sheet", err, stderr)
			}
			w.Header().Set("X-Video-Duration", strconv.FormatFloat(duration, 'f', 3, 64))
			return serveFile(w, output, "image/jpeg", "frames.jpg")
		}
		filter := fmt.Sprintf("fps=fps=%.12f:start_time=0:round=up,scale=512:512:force_original_aspect_ratio=decrease,pad=512:512:(ow-iw)/2:(oh-ih)/2,tile=4x2:nb_frames=8", 8/duration)
		_, stderr, err = run(ctx, a.cfg.FFmpegPath, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-threads", "2", "-i", input, "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", filter, "-filter_threads", "1", "-frames:v", "1", "-q:v", "3", output)
		if err != nil {
			return processError("ffmpeg frames", err, stderr)
		}
		w.Header().Set("X-Video-Duration", strconv.FormatFloat(duration, 'f', 3, 64))
		return serveFile(w, output, "image/jpeg", "frames.jpg")
	})
}
