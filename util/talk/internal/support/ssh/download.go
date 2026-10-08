package ssh

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"hash"
	"io"
	"mime"
	"net/http"
	"path"
	"sparktalk/internal/attachment"
	"time"
)

type DownloadRequest struct {
	Target
	Path           string `json:"path"`
	MaxBytes       int64  `json:"max_bytes"`
	TimeoutSeconds int    `json:"timeout_seconds,omitempty"`
}

type DownloadResult struct {
	Body   io.ReadCloser
	Name   string
	SHA256 string
	Size   int64
}

// The verifying stream reports integrity failures before the caller can commit
// a staged attachment. Closing the stream cancels the transfer context.
type downloadStream struct {
	io.ReadCloser
	hash         hash.Hash
	expectedHash string
	size, read   int64
	cancel       context.CancelFunc
}

func (s *downloadStream) Read(p []byte) (int, error) {
	n, err := s.ReadCloser.Read(p)
	if n > 0 {
		_, _ = s.hash.Write(p[:n])
		s.read += int64(n)
	}
	if s.read > s.size {
		return n, fmt.Errorf("SSH download size mismatch")
	}
	if err == io.EOF && (s.read != s.size || fmt.Sprintf("%x", s.hash.Sum(nil)) != s.expectedHash) {
		return n, fmt.Errorf("SSH download size or checksum mismatch")
	}
	return n, err
}
func (s *downloadStream) Close() error { s.cancel(); return s.ReadCloser.Close() }

func (c *Client) Download(ctx context.Context, input DownloadRequest) (DownloadResult, error) {
	if input.MaxBytes == 0 {
		input.MaxBytes = (attachment.Limits{}).MaxBytes()
	}
	ctx, cancel := context.WithTimeout(ctx, 5*time.Minute)
	success := false
	defer func() {
		if !success {
			cancel()
		}
	}()
	body, err := json.Marshal(input)
	if err != nil {
		return DownloadResult{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint+"/v1/ssh/download", bytes.NewReader(body))
	if err != nil {
		return DownloadResult{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := (&http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}).Do(req)
	if err != nil {
		return DownloadResult{}, err
	}
	defer func() {
		if !success {
			resp.Body.Close()
		}
	}()
	if resp.StatusCode != http.StatusOK {
		return DownloadResult{}, decodeHTTPError(resp)
	}
	if resp.ContentLength < 1 || resp.ContentLength > input.MaxBytes {
		return DownloadResult{}, fmt.Errorf("SSH download exceeds %d MiB or has invalid size", input.MaxBytes>>20)
	}
	_, params, err := mime.ParseMediaType(resp.Header.Get("Content-Disposition"))
	if err != nil || params["filename"] != path.Base(input.Path) {
		return DownloadResult{}, fmt.Errorf("SSH download filename mismatch")
	}
	sum := resp.Header.Get("X-Content-SHA256")
	if len(sum) != 64 {
		return DownloadResult{}, fmt.Errorf("missing SSH download checksum")
	}
	success = true
	return DownloadResult{Body: &downloadStream{ReadCloser: resp.Body, hash: sha256.New(), expectedHash: sum, size: resp.ContentLength, cancel: cancel}, Name: params["filename"], SHA256: sum, Size: resp.ContentLength}, nil
}
