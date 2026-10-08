package webtools

import (
	"context"
	"fmt"
	"html"
	"io"
	"net/http"
	"net/url"
	"strings"
)

func (r *Runner) search(ctx context.Context, query string) ([]SearchResult, error) {
	searchURL := "https://html.duckduckgo.com/html/?q=" + url.QueryEscape(query)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, searchURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; SparkTalk/1.0)")
	resp, err := r.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("DuckDuckGo returned HTTP %d", resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxFetchBytes))
	if err != nil {
		return nil, err
	}
	results := parseDuckDuckGo(string(body), r.results)
	if len(results) == 0 {
		return nil, fmt.Errorf("no search results")
	}
	return results, nil
}

func parseDuckDuckGo(page string, limit int) []SearchResult {
	parts := strings.Split(page, `class="result__a"`)
	results := make([]SearchResult, 0, limit)
	for _, part := range parts[1:] {
		if len(results) >= limit {
			break
		}
		href := attribute(part, "href")
		href = html.UnescapeString(href)
		if parsed, err := url.Parse(href); err == nil {
			if actual := parsed.Query().Get("uddg"); actual != "" {
				href = actual
			}
		}
		title := between(part, ">", "</a>")
		snippet := ""
		if index := strings.Index(part, `class="result__snippet"`); index >= 0 {
			snippet = between(part[index:], ">", "</a>")
		}
		title = cleanHTML(title)
		snippet = cleanHTML(snippet)
		if href != "" && title != "" {
			results = append(results, SearchResult{Title: title, URL: href, Snippet: snippet})
		}
	}
	return results
}
