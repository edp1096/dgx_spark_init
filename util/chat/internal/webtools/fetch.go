package webtools

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
)

func (r *Runner) fetch(ctx context.Context, rawURL string) (string, string, error) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return "", "", err
	}
	if err := validatePublicURL(ctx, u); err != nil {
		return "", "", err
	}
	body, contentType, finalURL, err := r.fetchPage(ctx, u, "")
	if err != nil {
		return "", "", err
	}
	if strings.EqualFold(finalURL.Hostname(), "blog.naver.com") && strings.Contains(strings.ToLower(contentType), "html") {
		if frameURL := naverFrameURL(finalURL, string(body)); frameURL != nil {
			if err := validatePublicURL(ctx, frameURL); err != nil {
				return "", "", err
			}
			body, contentType, finalURL, err = r.fetchPage(ctx, frameURL, finalURL.String())
			if err != nil {
				return "", "", err
			}
		}
	}
	text := string(body)
	if strings.Contains(contentType, "html") || strings.Contains(strings.ToLower(text[:min(len(text), 256)]), "<html") {
		if strings.EqualFold(finalURL.Hostname(), "blog.naver.com") {
			if article := extractNaverArticle(text); article != "" {
				text = article
			} else {
				text = extractReadableHTML(text)
			}
		} else {
			text = extractReadableHTML(text)
		}
	}
	if len(text) > maxResultBytes {
		text = text[:maxResultBytes] + "\n[truncated]"
	}
	return text, finalURL.String(), nil
}

func (r *Runner) fetchPage(ctx context.Context, u *url.URL, referer string) ([]byte, string, *url.URL, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, "", nil, err
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; SparkTalk/1.0)")
	if referer != "" {
		req.Header.Set("Referer", referer)
	}
	resp, err := r.client.Do(req)
	if err != nil {
		return nil, "", nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, "", nil, fmt.Errorf("web page returned HTTP %d", resp.StatusCode)
	}
	contentType := strings.ToLower(resp.Header.Get("Content-Type"))
	if contentType != "" && !strings.Contains(contentType, "text/") && !strings.Contains(contentType, "json") && !strings.Contains(contentType, "xml") {
		return nil, "", nil, fmt.Errorf("unsupported content type: %s", contentType)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxFetchBytes))
	if err != nil {
		return nil, "", nil, err
	}
	return body, contentType, resp.Request.URL, nil
}
