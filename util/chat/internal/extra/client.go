package extra

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"strings"
	"time"
)

type Client struct {
	endpoint string
	http     *http.Client
}

type Target struct {
	Host  string `json:"host"`
	Port  int    `json:"port"`
	User  string `json:"user"`
	KeyID string `json:"key_id"`
}

type ExecRequest struct {
	Target
	Command        string `json:"command"`
	TimeoutSeconds int    `json:"timeout_seconds,omitempty"`
}

type Event struct {
	Type       string `json:"type"`
	Data       string `json:"data,omitempty"`
	ExitCode   *int   `json:"exit_code,omitempty"`
	DurationMS int64  `json:"duration_ms,omitempty"`
	Truncated  bool   `json:"truncated,omitempty"`
	Error      string `json:"error,omitempty"`
}

type Result struct {
	Stdout     string `json:"stdout"`
	Stderr     string `json:"stderr"`
	ExitCode   int    `json:"exit_code"`
	DurationMS int64  `json:"duration_ms"`
	Truncated  bool   `json:"truncated,omitempty"`
	Error      string `json:"error,omitempty"`
}

type HostKey struct {
	Fingerprint string `json:"fingerprint"`
	PublicKey   string `json:"public_key"`
}

type SSHKey struct {
	ID          string `json:"id"`
	Type        string `json:"type"`
	Fingerprint string `json:"fingerprint"`
	PublicKey   string `json:"public_key"`
}

type HTTPError struct {
	Status  int
	Message string
	HostKey *HostKey
}

func (e *HTTPError) Error() string { return e.Message }

func New(endpoint string) *Client {
	return &Client{endpoint: strings.TrimRight(strings.TrimSpace(endpoint), "/"), http: &http.Client{Timeout: 15 * time.Second}}
}

func (c *Client) Health(ctx context.Context) map[string]any {
	if c.endpoint == "" {
		return map[string]any{"status": "disabled"}
	}
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, c.endpoint+"/health", nil)
	resp, err := c.http.Do(req)
	if err != nil {
		return map[string]any{"status": "offline", "error": err.Error()}
	}
	defer resp.Body.Close()
	var result map[string]any
	if resp.StatusCode != http.StatusOK || json.NewDecoder(resp.Body).Decode(&result) != nil {
		return map[string]any{"status": "offline", "error": resp.Status}
	}
	return result
}

func (c *Client) Check(ctx context.Context, target Target) error {
	return c.jsonRequest(ctx, "/v1/ssh/check", target, nil)
}

func (c *Client) Keys(ctx context.Context) ([]SSHKey, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.endpoint+"/v1/ssh/keys", nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, decodeHTTPError(resp)
	}
	var keys []SSHKey
	if err := json.NewDecoder(resp.Body).Decode(&keys); err != nil {
		return nil, err
	}
	return keys, nil
}

func (c *Client) GenerateKey(ctx context.Context, keyID string) (SSHKey, error) {
	var key SSHKey
	err := c.jsonRequest(ctx, "/v1/ssh/keys/generate", map[string]string{"key_id": keyID}, &key)
	return key, err
}

func (c *Client) ImportKey(ctx context.Context, keyID string, privateKey []byte) (SSHKey, error) {
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	if err := writer.WriteField("key_id", keyID); err != nil {
		return SSHKey{}, err
	}
	part, err := writer.CreateFormFile("key", "private-key")
	if err != nil {
		return SSHKey{}, err
	}
	if _, err := part.Write(privateKey); err != nil {
		return SSHKey{}, err
	}
	if err := writer.Close(); err != nil {
		return SSHKey{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint+"/v1/ssh/keys/import", &body)
	if err != nil {
		return SSHKey{}, err
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())
	resp, err := c.http.Do(req)
	if err != nil {
		return SSHKey{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusCreated {
		return SSHKey{}, decodeHTTPError(resp)
	}
	var key SSHKey
	err = json.NewDecoder(resp.Body).Decode(&key)
	return key, err
}

func (c *Client) DeleteKey(ctx context.Context, keyID string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, c.endpoint+"/v1/ssh/keys/"+url.PathEscape(keyID), nil)
	if err != nil {
		return err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNoContent {
		return decodeHTTPError(resp)
	}
	return nil
}

func (c *Client) Trust(ctx context.Context, host string, port int, publicKey string) (map[string]any, error) {
	var result map[string]any
	err := c.jsonRequest(ctx, "/v1/ssh/trust", map[string]any{"host": host, "port": port, "public_key": publicKey}, &result)
	return result, err
}

func (c *Client) Execute(ctx context.Context, request ExecRequest, onEvent func(Event) error) (Result, error) {
	data, err := json.Marshal(request)
	if err != nil {
		return Result{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint+"/v1/ssh/exec", bytes.NewReader(data))
	if err != nil {
		return Result{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return Result{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return Result{}, decodeHTTPError(resp)
	}
	var result Result
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	for scanner.Scan() {
		var event Event
		if err := json.Unmarshal(scanner.Bytes(), &event); err != nil {
			return result, fmt.Errorf("decode Extra SSH stream: %w", err)
		}
		switch event.Type {
		case "stdout":
			result.Stdout += event.Data
		case "stderr":
			result.Stderr += event.Data
		case "exit":
			if event.ExitCode != nil {
				result.ExitCode = *event.ExitCode
			}
			result.DurationMS, result.Truncated, result.Error = event.DurationMS, event.Truncated, event.Error
		}
		if onEvent != nil {
			if err := onEvent(event); err != nil {
				return result, err
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return result, err
	}
	return result, nil
}

func (c *Client) jsonRequest(ctx context.Context, path string, input, output any) error {
	data, err := json.Marshal(input)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint+path, bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return decodeHTTPError(resp)
	}
	if output != nil {
		return json.NewDecoder(resp.Body).Decode(output)
	}
	return nil
}

func decodeHTTPError(resp *http.Response) error {
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 64*1024))
	var payload struct {
		Error   string  `json:"error"`
		HostKey HostKey `json:"host_key"`
	}
	_ = json.Unmarshal(data, &payload)
	message := strings.TrimSpace(payload.Error)
	if message == "" {
		message = strings.TrimSpace(string(data))
	}
	if message == "" {
		message = resp.Status
	}
	result := &HTTPError{Status: resp.StatusCode, Message: message}
	if payload.HostKey.Fingerprint != "" {
		result.HostKey = &payload.HostKey
	}
	return result
}
