package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"mime"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

type api struct {
	cfg       config
	sem       chan struct{}
	ffmpegVer string
	ytDLPVer  string
}

func newAPI(cfg config) (*api, error) {
	if err := os.MkdirAll(cfg.TempDir, 0o700); err != nil {
		return nil, fmt.Errorf("create temp directory: %w", err)
	}
	ffmpegVer, err := commandVersion(cfg.FFmpegPath)
	if err != nil {
		return nil, fmt.Errorf("ffmpeg unavailable: %w", err)
	}
	if _, err := commandVersion(cfg.FFprobePath); err != nil {
		return nil, fmt.Errorf("ffprobe unavailable: %w", err)
	}
	ytDLPVer, err := ytDLPVersion(cfg.YtDLPPath)
	if err != nil {
		return nil, fmt.Errorf("yt-dlp unavailable: %w", err)
	}
	return &api{
		cfg: cfg, sem: make(chan struct{}, cfg.MaxConcurrency),
		ffmpegVer: ffmpegVer, ytDLPVer: ytDLPVer,
	}, nil
}

func commandVersion(path string) (string, error) {
	out, err := exec.Command(path, "-version").Output()
	if err != nil {
		return "", err
	}
	line := strings.SplitN(string(out), "\n", 2)[0]
	return strings.TrimSpace(line), nil
}

func (a *api) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", a.health)
	mux.HandleFunc("POST /v1/probe", a.probe)
	mux.HandleFunc("POST /v1/audio/extract", a.extractAudio)
	mux.HandleFunc("POST /v1/video/normalize", a.normalizeVideo)
	mux.HandleFunc("POST /v1/video/frames", a.videoFrames)
	mux.HandleFunc("POST /v1/source/probe", a.probeSource)
	mux.HandleFunc("POST /v1/source/download", a.downloadSource)
	return requestLog(mux)
}

func (a *api) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"status":          "ok",
		"ffmpeg":          a.ffmpegVer,
		"yt_dlp":          a.ytDLPVer,
		"active":          len(a.sem),
		"max_concurrency": cap(a.sem),
	})
}

func (a *api) probe(w http.ResponseWriter, r *http.Request) {
	a.withInput(w, r, func(ctx context.Context, input string) error {
		stdout, stderr, err := run(ctx, a.cfg.FFprobePath,
			"-v", "error", "-show_format", "-show_streams", "-of", "json", input)
		if err != nil {
			return processError("ffprobe", err, stderr)
		}
		w.Header().Set("Content-Type", "application/json")
		_, err = w.Write(stdout)
		return err
	})
}

func (a *api) extractAudio(w http.ResponseWriter, r *http.Request) {
	a.withInput(w, r, func(ctx context.Context, input string) error {
		sampleRate, err := boundedQueryInt(r, "sample_rate", 16000, 8000, 192000)
		if err != nil {
			return &httpError{Status: http.StatusBadRequest, Message: err.Error()}
		}
		channels, err := boundedQueryInt(r, "channels", 1, 1, 8)
		if err != nil {
			return &httpError{Status: http.StatusBadRequest, Message: err.Error()}
		}
		output := filepath.Join(filepath.Dir(input), "audio.wav")
		_, stderr, err := run(ctx, a.cfg.FFmpegPath,
			"-nostdin", "-hide_banner", "-loglevel", "error", "-y",
			"-i", input, "-map", "0:a:0", "-vn", "-sn", "-dn",
			"-ac", strconv.Itoa(channels), "-ar", strconv.Itoa(sampleRate),
			"-c:a", "pcm_s16le", output)
		if err != nil {
			return processError("ffmpeg", err, stderr)
		}
		return serveFile(w, output, "audio/wav", "audio.wav")
	})
}

func (a *api) normalizeVideo(w http.ResponseWriter, r *http.Request) {
	a.withInput(w, r, func(ctx context.Context, input string) error {
		output := filepath.Join(filepath.Dir(input), "video.mp4")
		_, stderr, err := run(ctx, a.cfg.FFmpegPath,
			"-nostdin", "-hide_banner", "-loglevel", "error", "-y",
			"-i", input, "-map", "0:v:0", "-map", "0:a:0?",
			"-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
			"-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
			"-movflags", "+faststart", output)
		if err != nil {
			return processError("ffmpeg", err, stderr)
		}
		return serveFile(w, output, "video/mp4", "video.mp4")
	})
}

func (a *api) withInput(w http.ResponseWriter, r *http.Request, fn func(context.Context, string) error) {
	select {
	case a.sem <- struct{}{}:
		defer func() { <-a.sem }()
	case <-r.Context().Done():
		writeError(w, &httpError{Status: 499, Message: "request canceled"})
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), a.cfg.ProcessTimeout)
	defer cancel()
	dir, err := os.MkdirTemp(a.cfg.TempDir, "request-")
	if err != nil {
		writeError(w, fmt.Errorf("create request directory: %w", err))
		return
	}
	defer os.RemoveAll(dir)

	input, err := saveUpload(w, r, dir, a.cfg.MaxUploadBytes)
	if err != nil {
		writeError(w, err)
		return
	}
	if err := fn(ctx, input); err != nil {
		writeError(w, err)
	}
}

func saveUpload(w http.ResponseWriter, r *http.Request, dir string, maxBytes int64) (string, error) {
	r.Body = http.MaxBytesReader(w, r.Body, maxBytes)
	reader := io.Reader(r.Body)
	ext := extensionFromContentType(r.Header.Get("Content-Type"))
	if strings.HasPrefix(r.Header.Get("Content-Type"), "multipart/form-data") {
		multipartReader, err := r.MultipartReader()
		if err != nil {
			return "", &httpError{Status: http.StatusBadRequest, Message: "invalid multipart request"}
		}
		for {
			part, err := multipartReader.NextPart()
			if errors.Is(err, io.EOF) {
				return "", &httpError{Status: http.StatusBadRequest, Message: "multipart request has no file"}
			}
			if err != nil {
				return "", &httpError{Status: http.StatusBadRequest, Message: "invalid multipart request"}
			}
			if part.FileName() == "" {
				part.Close()
				continue
			}
			reader = part
			ext = safeExtension(part.FileName())
			defer part.Close()
			break
		}
	}
	input := filepath.Join(dir, "input"+ext)
	file, err := os.OpenFile(input, os.O_CREATE|os.O_WRONLY|os.O_EXCL, 0o600)
	if err != nil {
		return "", fmt.Errorf("create input file: %w", err)
	}
	_, copyErr := io.Copy(file, reader)
	closeErr := file.Close()
	if copyErr != nil {
		var maxErr *http.MaxBytesError
		if errors.As(copyErr, &maxErr) {
			return "", &httpError{Status: http.StatusRequestEntityTooLarge, Message: "upload exceeds configured limit"}
		}
		return "", fmt.Errorf("save input: %w", copyErr)
	}
	if closeErr != nil {
		return "", fmt.Errorf("close input: %w", closeErr)
	}
	info, err := os.Stat(input)
	if err != nil || info.Size() == 0 {
		return "", &httpError{Status: http.StatusBadRequest, Message: "empty upload"}
	}
	return input, nil
}

func extensionFromContentType(contentType string) string {
	mediaType, _, err := mime.ParseMediaType(contentType)
	if err != nil {
		return ".media"
	}
	known := map[string]string{
		"audio/mpeg":  ".mp3",
		"audio/ogg":   ".ogg",
		"audio/wav":   ".wav",
		"audio/x-wav": ".wav",
		"video/mp4":   ".mp4",
		"video/ogg":   ".ogv",
		"video/webm":  ".webm",
	}
	if ext, ok := known[mediaType]; ok {
		return ext
	}
	exts, _ := mime.ExtensionsByType(mediaType)
	if len(exts) > 0 {
		return safeExtension(exts[0])
	}
	return ".media"
}

func safeExtension(name string) string {
	ext := strings.ToLower(filepath.Ext(name))
	if len(ext) < 2 || len(ext) > 10 {
		return ".media"
	}
	for _, char := range ext[1:] {
		if (char < 'a' || char > 'z') && (char < '0' || char > '9') {
			return ".media"
		}
	}
	return ext
}

func boundedQueryInt(r *http.Request, name string, fallback, min, max int) (int, error) {
	value := r.URL.Query().Get(name)
	if value == "" {
		return fallback, nil
	}
	parsed, err := strconv.Atoi(value)
	if err != nil || parsed < min || parsed > max {
		return 0, fmt.Errorf("%s must be between %d and %d", name, min, max)
	}
	return parsed, nil
}

func run(ctx context.Context, executable string, args ...string) ([]byte, []byte, error) {
	cmd := exec.CommandContext(ctx, executable, args...)
	var stdout, stderr limitedBuffer
	stdout.Limit = 8 * 1024 * 1024
	stderr.Limit = 64 * 1024
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	return stdout.Bytes(), stderr.Bytes(), err
}

type limitedBuffer struct {
	Data  []byte
	Limit int
}

func (b *limitedBuffer) Write(p []byte) (int, error) {
	available := b.Limit - len(b.Data)
	if available > 0 {
		if len(p) < available {
			available = len(p)
		}
		b.Data = append(b.Data, p[:available]...)
	}
	return len(p), nil
}

func (b *limitedBuffer) Bytes() []byte { return b.Data }

func processError(tool string, err error, stderr []byte) error {
	message := strings.TrimSpace(string(stderr))
	if message == "" {
		message = err.Error()
	}
	return &httpError{Status: http.StatusUnprocessableEntity, Message: tool + ": " + message}
}

func serveFile(w http.ResponseWriter, path, contentType, filename string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return err
	}
	w.Header().Set("Content-Type", contentType)
	w.Header().Set("Content-Disposition", fmt.Sprintf(`attachment; filename=%q`, filename))
	w.Header().Set("Content-Length", strconv.FormatInt(info.Size(), 10))
	_, err = io.Copy(w, file)
	return err
}

type httpError struct {
	Status  int
	Message string
}

func (e *httpError) Error() string { return e.Message }

func writeError(w http.ResponseWriter, err error) {
	status := http.StatusInternalServerError
	message := "internal server error"
	var apiErr *httpError
	if errors.As(err, &apiErr) {
		status, message = apiErr.Status, apiErr.Message
	} else {
		log.Printf("request error: %v", err)
	}
	writeJSON(w, status, map[string]string{"error": message})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func requestLog(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		started := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(started).Round(time.Millisecond))
	})
}
