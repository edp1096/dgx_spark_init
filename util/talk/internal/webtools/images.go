package webtools

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path"
	"strings"

	"golang.org/x/net/html"
)

type ImageSource struct {
	URL       string `json:"url"`
	SourceURL string `json:"source_url"`
	Alt       string `json:"alt,omitempty"`
}

// Extract addresses from HTML attributes, never from page prose or scripts.
func extractImages(base *url.URL, page string) []ImageSource {
	out := []ImageSource{}
	seen := map[string]bool{}
	add := func(raw, alt string) {
		u, err := url.Parse(strings.TrimSpace(raw))
		if err != nil || raw == "" || len(raw) > 4096 {
			return
		}
		u = base.ResolveReference(u)
		if (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" || u.User != nil || seen[u.String()] || len(out) >= 40 {
			return
		}
		seen[u.String()] = true
		if len(alt) > 300 {
			alt = string([]rune(alt)[:min(100, len([]rune(alt)))])
		}
		out = append(out, ImageSource{URL: u.String(), SourceURL: base.String(), Alt: alt})
	}
	z := html.NewTokenizer(strings.NewReader(page))
	for {
		tt := z.Next()
		if tt == html.ErrorToken {
			break
		}
		if tt != html.StartTagToken && tt != html.SelfClosingTagToken {
			continue
		}
		token := z.Token()
		attrs := map[string]string{}
		for _, a := range token.Attr {
			attrs[a.Key] = a.Val
		}
		switch token.Data {
		case "img":
			add(attrs["src"], attrs["alt"])
			add(attrs["data-src"], attrs["alt"])
		case "meta":
			if attrs["property"] == "og:image" || attrs["name"] == "twitter:image" {
				add(attrs["content"], "")
			}
		case "a":
			u, _ := url.Parse(attrs["href"])
			if u == nil {
				continue
			}
			switch strings.ToLower(path.Ext(u.Path)) {
			case ".jpg", ".jpeg", ".png", ".webp":
				add(attrs["href"], attrs["title"])
			}
		}
	}
	return out
}

// Uses the same DNS-pinned transport and redirect validation as web_fetch.
func (r *Runner) DownloadImage(ctx context.Context, rawURL string, limit int64) ([]byte, string, error) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return nil, "", err
	}
	if err = validatePublicURL(ctx, u); err != nil {
		return nil, "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, "", err
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; SparkTalk/1.0)")
	resp, err := r.client.Do(req)
	if err != nil {
		return nil, "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, "", fmt.Errorf("image download HTTP %d", resp.StatusCode)
	}
	if resp.ContentLength > limit {
		return nil, "", fmt.Errorf("image exceeds %d MB", limit>>20)
	}
	data, err := io.ReadAll(io.LimitReader(resp.Body, limit+1))
	if err != nil {
		return nil, "", err
	}
	if int64(len(data)) > limit {
		return nil, "", fmt.Errorf("image exceeds %d MB", limit>>20)
	}
	kind := http.DetectContentType(data)
	if kind != "image/png" && kind != "image/jpeg" && kind != "image/webp" {
		return nil, "", fmt.Errorf("URL is not a PNG/JPEG/WebP image; use web_fetch on the page and import an images[].url")
	}
	return data, resp.Request.URL.String(), nil
}
