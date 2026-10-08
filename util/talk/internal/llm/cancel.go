package llm

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Detect the native SGLang cancellation API without changing generic OpenAI
// requests. Probe once per application turn, not once per tool/model round.
func (c *Client) WithBackendCancellation(ctx context.Context) *Client {
	probe, cancel := context.WithTimeout(ctx, time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(probe, http.MethodGet, strings.TrimSuffix(c.endpoint, "/v1")+"/server_info", nil)
	if err != nil {
		return c
	}
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return c
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return c
	}
	var info map[string]json.RawMessage
	if json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&info) != nil {
		return c
	}
	if info["max_total_num_tokens"] == nil || info["mamba_radix_cache_strategy"] == nil {
		return c
	}
	copy := *c
	copy.nativeAbort = true
	return &copy
}

func (c *Client) abortNativeRequest(id string) {
	if id == "" {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	body, _ := json.Marshal(map[string]any{"rid": id, "abort_all": false})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimSuffix(c.endpoint, "/v1")+"/abort_request", bytes.NewReader(body))
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	resp, err := c.http.Do(req)
	if err == nil {
		io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
		resp.Body.Close()
	}
}

// SGLang can drop its per-request state before disconnect cleanup reaches the
// scheduler. Send the exact native abort while the stream is still connected.
func (c *Client) streamContext(ctx context.Context, payload map[string]any) (context.Context, func()) {
	if !c.nativeAbort {
		return ctx, func() {}
	}
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return ctx, func() {}
	}
	id := "talk-" + hex.EncodeToString(b)
	payload["rid"] = id
	transport, cancel := context.WithCancel(context.WithoutCancel(ctx))
	done := make(chan struct{})
	var once sync.Once
	var abortOnce sync.Once
	abort := func() { abortOnce.Do(func() { c.abortNativeRequest(id) }) }
	go func() {
		select {
		case <-ctx.Done():
			abort()
			cancel()
		case <-done:
		}
	}()
	return transport, func() {
		if ctx.Err() != nil {
			abort()
		}
		once.Do(func() { close(done) })
		cancel()
	}
}
