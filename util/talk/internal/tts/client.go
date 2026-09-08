package tts

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"sparktalk/internal/config"
)

type Status struct {
	Enabled bool   `json:"enabled"`
	Status  string `json:"status"`
	Model   string `json:"model,omitempty"`
	Error   string `json:"error,omitempty"`
}

type Client struct {
	cfg  config.TTSConfig
	http *http.Client
}

func New(cfg config.TTSConfig) *Client {
	timeout, _ := time.ParseDuration(cfg.Timeout)
	if timeout <= 0 {
		timeout = 10 * time.Minute
	}
	return &Client{cfg: cfg, http: &http.Client{Timeout: timeout}}
}

func (c *Client) Health(ctx context.Context) Status {
	status := Status{Enabled: c.cfg.Enabled, Model: c.cfg.Model}
	if !c.cfg.Enabled {
		status.Status = "disabled"
		return status
	}
	checkCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(checkCtx, http.MethodGet, c.cfg.Endpoint+"/health", nil)
	if err == nil {
		var resp *http.Response
		resp, err = c.http.Do(req)
		if err == nil {
			defer resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				status.Status = "ok"
				return status
			}
			err = fmt.Errorf("HTTP %d", resp.StatusCode)
		}
	}
	status.Status = "offline"
	if err != nil {
		status.Error = err.Error()
	}
	return status
}

func (c *Client) Speech(ctx context.Context, text string) ([]byte, string, error) {
	stream, err := c.SpeechStream(ctx, text)
	if err != nil {
		return nil, "", err
	}
	defer stream.Body.Close()
	audio, err := io.ReadAll(stream.Body)
	if err != nil {
		return nil, "", err
	}
	if len(audio) == 0 {
		return nil, "", errors.New("TTS API returned empty audio")
	}
	return audio, stream.ContentType, nil
}

type SpeechStream struct {
	Body        io.ReadCloser
	ContentType string
	SampleRate  int
}

func (c *Client) SpeechStream(ctx context.Context, text string) (*SpeechStream, error) {
	return c.SpeechStreamLanguage(ctx, text, "")
}

func (c *Client) SpeechStreamLanguage(ctx context.Context, text, language string) (*SpeechStream, error) {
	if !c.cfg.Enabled {
		return nil, errors.New("TTS is disabled")
	}
	text = strings.TrimSpace(text)
	if text == "" {
		return nil, errors.New("text is required")
	}
	if len(text) > 64<<10 {
		return nil, errors.New("text is too large")
	}
	if strings.TrimSpace(language) == "" {
		language = c.cfg.Language
	}
	payload := map[string]any{
		"model": c.cfg.Model, "input": text, "language": language,
		"voice": c.cfg.Voice, "response_format": "pcm", "stream": true,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.cfg.Endpoint+"/v1/audio/speech", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("TTS API: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		detail, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
		return nil, fmt.Errorf("TTS API HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(detail)))
	}
	contentType := strings.TrimSpace(resp.Header.Get("Content-Type"))
	if contentType == "" || contentType == "application/octet-stream" {
		contentType = "audio/pcm"
	}
	return &SpeechStream{Body: resp.Body, ContentType: contentType, SampleRate: c.cfg.SampleRate}, nil
}
