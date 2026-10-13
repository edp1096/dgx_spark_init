package webtools

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"

	"sparktalk/internal/sourcebook"
)

// Resource uses the same DNS-pinned client and redirect checks as web_fetch.
// A limit is an error, never a silently truncated successful source.
func (r *Runner) Resource(ctx context.Context, target string, limit int) ([]byte, string, string, error) {
	u, e := url.Parse(target)
	if e != nil {
		return nil, "", "", e
	}
	if e = validatePublicURL(ctx, u); e != nil {
		return nil, "", "", e
	}
	req, e := http.NewRequestWithContext(ctx, "GET", u.String(), nil)
	if e != nil {
		return nil, "", "", e
	}
	req.Header.Set("User-Agent", "SparkTalk/1.0")
	resp, e := r.client.Do(req)
	if e != nil {
		return nil, "", "", e
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, "", "", fmt.Errorf("source HTTP %d", resp.StatusCode)
	}
	data, e := io.ReadAll(io.LimitReader(resp.Body, int64(limit)+1))
	if e != nil {
		return nil, "", "", e
	}
	if len(data) > limit {
		return nil, "", "", fmt.Errorf("source exceeds %d bytes; full source not acquired", limit)
	}
	return data, resp.Header.Get("Content-Type"), resp.Request.URL.String(), nil
}
func (r *Runner) Source(ctx context.Context, target string) (*sourcebook.Book, []byte, error) {
	raw, typ, final, e := r.Resource(ctx, target, sourcebook.MaxBytes)
	if e != nil {
		return nil, nil, e
	}
	b, e := sourcebook.Parse(raw, typ, final, func(target string) ([]byte, string, error) {
		data, m, _, e := r.Resource(ctx, target, sourcebook.MaxAssetBytes)
		return data, m, e
	})
	return b, raw, e
}
