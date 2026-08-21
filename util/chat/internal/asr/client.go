package asr

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"net/url"
	"path/filepath"
	"strings"
	"time"

	"sparktalk/internal/config"
)

var ErrNoAudio = errors.New("media has no audio stream")

type Result struct {
	Text       string      `json:"text"`
	Language   string      `json:"language"`
	Timestamps []Timestamp `json:"timestamps,omitempty"`
}

type Timestamp struct {
	Text  string  `json:"text"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}

type ServiceStatus struct {
	Status string `json:"status"`
	Model  string `json:"model,omitempty"`
	Error  string `json:"error,omitempty"`
}

type Status struct {
	Enabled bool          `json:"enabled"`
	FFmpeg  ServiceStatus `json:"ffmpeg"`
	ASR     ServiceStatus `json:"asr"`
}

type Client struct {
	cfg  config.ASRConfig
	http *http.Client
}

func New(cfg config.ASRConfig) *Client {
	timeout, _ := time.ParseDuration(cfg.Timeout)
	if timeout <= 0 {
		timeout = 30 * time.Minute
	}
	return &Client{cfg: cfg, http: &http.Client{Timeout: timeout}}
}

func (c *Client) Health(ctx context.Context) Status {
	status := Status{Enabled: c.cfg.Enabled}
	if !c.cfg.Enabled {
		status.FFmpeg.Status = "disabled"
		status.ASR.Status = "disabled"
		return status
	}
	status.FFmpeg = c.health(ctx, c.cfg.FFmpegEndpoint+"/health", false)
	status.ASR = c.health(ctx, c.cfg.Endpoint+"/health", true)
	return status
}

func (c *Client) health(ctx context.Context, endpoint string, modelResponse bool) ServiceStatus {
	checkCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(checkCtx, http.MethodGet, endpoint, nil)
	if err != nil {
		return ServiceStatus{Status: "offline", Error: err.Error()}
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return ServiceStatus{Status: "offline", Error: err.Error()}
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return ServiceStatus{Status: "offline", Error: resp.Status}
	}
	status := ServiceStatus{Status: "ok"}
	if modelResponse {
		var payload struct {
			Model string `json:"model"`
		}
		_ = json.NewDecoder(io.LimitReader(resp.Body, 64<<10)).Decode(&payload)
		status.Model = payload.Model
	}
	return status
}

func (c *Client) Transcribe(ctx context.Context, source io.Reader, filename, mimeType string) (Result, error) {
	if !c.cfg.Enabled {
		return Result{}, errors.New("ASR is disabled")
	}
	query := url.Values{"sample_rate": {"16000"}, "channels": {"1"}}
	ffmpegURL := c.cfg.FFmpegEndpoint + "/v1/audio/extract?" + query.Encode()
	ffmpegReq, err := http.NewRequestWithContext(ctx, http.MethodPost, ffmpegURL, source)
	if err != nil {
		return Result{}, err
	}
	ffmpegReq.Header.Set("Content-Type", mimeType)
	ffmpegResp, err := c.http.Do(ffmpegReq)
	if err != nil {
		return Result{}, fmt.Errorf("SparkTalk Extra Media: %w", err)
	}
	defer ffmpegResp.Body.Close()
	if ffmpegResp.StatusCode != http.StatusOK {
		detail, _ := io.ReadAll(io.LimitReader(ffmpegResp.Body, 64<<10))
		message := strings.TrimSpace(string(detail))
		if strings.Contains(strings.ToLower(message), "matches no streams") || strings.Contains(strings.ToLower(message), "no audio") {
			return Result{}, ErrNoAudio
		}
		return Result{}, fmt.Errorf("SparkTalk Extra Media HTTP %d: %s", ffmpegResp.StatusCode, message)
	}

	pipeReader, pipeWriter := io.Pipe()
	multipartWriter := multipart.NewWriter(pipeWriter)
	writeDone := make(chan error, 1)
	go func() {
		err := writeASRMultipart(multipartWriter, ffmpegResp.Body, filename, c.cfg)
		if closeErr := multipartWriter.Close(); err == nil {
			err = closeErr
		}
		_ = pipeWriter.CloseWithError(err)
		writeDone <- err
	}()

	asrReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.cfg.Endpoint+"/v1/audio/transcriptions", pipeReader)
	if err != nil {
		_ = pipeReader.CloseWithError(err)
		return Result{}, err
	}
	asrReq.Header.Set("Content-Type", multipartWriter.FormDataContentType())
	asrResp, err := c.http.Do(asrReq)
	if err != nil {
		_ = pipeReader.CloseWithError(err)
		<-writeDone
		return Result{}, fmt.Errorf("ASR API: %w", err)
	}
	defer asrResp.Body.Close()
	writeErr := <-writeDone
	if writeErr != nil {
		return Result{}, fmt.Errorf("stream audio to ASR: %w", writeErr)
	}
	if asrResp.StatusCode != http.StatusOK {
		detail, _ := io.ReadAll(io.LimitReader(asrResp.Body, 64<<10))
		return Result{}, fmt.Errorf("ASR API HTTP %d: %s", asrResp.StatusCode, strings.TrimSpace(string(detail)))
	}
	var result Result
	if err := json.NewDecoder(io.LimitReader(asrResp.Body, 8<<20)).Decode(&result); err != nil {
		return Result{}, fmt.Errorf("decode ASR response: %w", err)
	}
	result.Text = strings.TrimSpace(result.Text)
	if result.Text == "" {
		return Result{}, errors.New("ASR returned empty text")
	}
	return result, nil
}

func writeASRMultipart(writer *multipart.Writer, audio io.Reader, filename string, cfg config.ASRConfig) error {
	for name, value := range map[string]string{
		"model": cfg.Model, "language": cfg.Language, "prompt": cfg.Prompt,
	} {
		if strings.TrimSpace(value) != "" {
			if err := writer.WriteField(name, value); err != nil {
				return err
			}
		}
	}
	header := make(textproto.MIMEHeader)
	header.Set("Content-Disposition", fmt.Sprintf(`form-data; name="file"; filename=%q`, transcriptFilename(filename)))
	header.Set("Content-Type", "audio/wav")
	part, err := writer.CreatePart(header)
	if err != nil {
		return err
	}
	_, err = io.Copy(part, audio)
	return err
}

func transcriptFilename(filename string) string {
	base := strings.TrimSuffix(filepath.Base(filename), filepath.Ext(filename))
	if base == "" || base == "." {
		base = "audio"
	}
	return base + ".wav"
}
