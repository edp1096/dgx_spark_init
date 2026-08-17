package webtools

import (
	"context"
	"encoding/json"
	"fmt"
	"html"
	"io"
	"net"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"

	"sparktalk/internal/llm"
)

const (
	maxFetchBytes  = 512 * 1024
	maxResultBytes = 32 * 1024
)

var (
	tagPattern         = regexp.MustCompile(`<[^>]*>`)
	spacePattern       = regexp.MustCompile(`\s+`)
	punctuationPattern = regexp.MustCompile(`\s+([.,!?;:])`)
)

type Runner struct {
	results int
	timeout time.Duration
	client  *http.Client
}

type SearchResult struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Snippet string `json:"snippet,omitempty"`
}

func New(searchResults int, timeout time.Duration) *Runner {
	if searchResults < 1 {
		searchResults = 5
	}
	if timeout <= 0 {
		timeout = 15 * time.Second
	}
	r := &Runner{results: searchResults, timeout: timeout}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	transport.Proxy = nil
	transport.DialContext = dialPublic
	r.client = &http.Client{
		Timeout:   timeout,
		Transport: transport,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 5 {
				return fmt.Errorf("too many redirects")
			}
			return validatePublicURL(req.Context(), req.URL)
		},
	}
	return r
}

func Definitions() []llm.Tool {
	return []llm.Tool{
		{Type: "function", Function: llm.ToolFunction{
			Name:        "web_search",
			Description: "Search the public web. Returns titles, URLs, and snippets. Use returned URLs as citations in the final answer.",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"query":{"type":"string","description":"Search query"}},"required":["query"],"additionalProperties":false}`),
		}},
		{Type: "function", Function: llm.ToolFunction{
			Name:        "web_fetch",
			Description: "Fetch readable text from a public HTTP(S) URL. Local and private network addresses are blocked. Cite the URL in the final answer.",
			Parameters:  json.RawMessage(`{"type":"object","properties":{"url":{"type":"string","description":"Public HTTP(S) URL"}},"required":["url"],"additionalProperties":false}`),
		}},
	}
}

func (r *Runner) Execute(ctx context.Context, name, arguments string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()
	switch name {
	case "web_search":
		var args struct {
			Query string `json:"query"`
		}
		if err := json.Unmarshal([]byte(arguments), &args); err != nil || strings.TrimSpace(args.Query) == "" {
			return "", fmt.Errorf("web_search requires a query")
		}
		results, err := r.search(ctx, strings.TrimSpace(args.Query))
		if err != nil {
			return "", err
		}
		data, _ := json.Marshal(map[string]any{"query": args.Query, "results": results})
		return string(data), nil
	case "web_fetch":
		var args struct {
			URL string `json:"url"`
		}
		if err := json.Unmarshal([]byte(arguments), &args); err != nil || strings.TrimSpace(args.URL) == "" {
			return "", fmt.Errorf("web_fetch requires a URL")
		}
		text, finalURL, err := r.fetch(ctx, strings.TrimSpace(args.URL))
		if err != nil {
			return "", err
		}
		data, _ := json.Marshal(map[string]string{"url": finalURL, "content": text})
		return string(data), nil
	default:
		return "", fmt.Errorf("unknown tool: %s", name)
	}
}

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

func (r *Runner) fetch(ctx context.Context, rawURL string) (string, string, error) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return "", "", err
	}
	if err := validatePublicURL(ctx, u); err != nil {
		return "", "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return "", "", err
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; SparkTalk/1.0)")
	resp, err := r.client.Do(req)
	if err != nil {
		return "", "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", "", fmt.Errorf("web page returned HTTP %d", resp.StatusCode)
	}
	contentType := strings.ToLower(resp.Header.Get("Content-Type"))
	if contentType != "" && !strings.Contains(contentType, "text/") && !strings.Contains(contentType, "json") && !strings.Contains(contentType, "xml") {
		return "", "", fmt.Errorf("unsupported content type: %s", contentType)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxFetchBytes))
	if err != nil {
		return "", "", err
	}
	text := string(body)
	if strings.Contains(contentType, "html") || strings.Contains(strings.ToLower(text[:min(len(text), 256)]), "<html") {
		text = cleanHTML(removeNonContent(text))
	}
	if len(text) > maxResultBytes {
		text = text[:maxResultBytes] + "\n[truncated]"
	}
	return text, resp.Request.URL.String(), nil
}

func validatePublicURL(ctx context.Context, u *url.URL) error {
	if u.Scheme != "http" && u.Scheme != "https" {
		return fmt.Errorf("only http and https URLs are allowed")
	}
	host := u.Hostname()
	if host == "" {
		return fmt.Errorf("URL has no host")
	}
	_, err := publicAddresses(ctx, host)
	return err
}

func publicAddresses(ctx context.Context, host string) ([]net.IPAddr, error) {
	addresses, err := net.DefaultResolver.LookupIPAddr(ctx, host)
	if err != nil {
		return nil, fmt.Errorf("resolve host: %w", err)
	}
	if len(addresses) == 0 {
		return nil, fmt.Errorf("host has no addresses")
	}
	for _, address := range addresses {
		ip := address.IP
		if ip.IsPrivate() || ip.IsLoopback() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() || ip.IsUnspecified() || ip.IsMulticast() {
			return nil, fmt.Errorf("private or local network URLs are blocked")
		}
	}
	return addresses, nil
}

func dialPublic(ctx context.Context, network, address string) (net.Conn, error) {
	host, port, err := net.SplitHostPort(address)
	if err != nil {
		return nil, err
	}
	addresses, err := publicAddresses(ctx, host)
	if err != nil {
		return nil, err
	}
	dialer := &net.Dialer{}
	var lastErr error
	for _, resolved := range addresses {
		conn, dialErr := dialer.DialContext(ctx, network, net.JoinHostPort(resolved.IP.String(), port))
		if dialErr == nil {
			return conn, nil
		}
		lastErr = dialErr
	}
	return nil, lastErr
}

func attribute(part, name string) string {
	needle := name + `="`
	start := strings.Index(part, needle)
	if start < 0 {
		return ""
	}
	start += len(needle)
	end := strings.Index(part[start:], `"`)
	if end < 0 {
		return ""
	}
	return part[start : start+end]
}

func between(value, startToken, endToken string) string {
	start := strings.Index(value, startToken)
	if start < 0 {
		return ""
	}
	start += len(startToken)
	end := strings.Index(value[start:], endToken)
	if end < 0 {
		return ""
	}
	return value[start : start+end]
}

func cleanHTML(value string) string {
	value = tagPattern.ReplaceAllString(value, " ")
	value = html.UnescapeString(value)
	value = strings.TrimSpace(spacePattern.ReplaceAllString(value, " "))
	return punctuationPattern.ReplaceAllString(value, "$1")
}

func removeNonContent(value string) string {
	lower := strings.ToLower(value)
	for _, tag := range []string{"script", "style", "nav", "footer", "aside", "iframe", "noscript"} {
		for {
			start := strings.Index(lower, "<"+tag)
			if start < 0 {
				break
			}
			end := strings.Index(lower[start:], "</"+tag+">")
			if end < 0 {
				value = value[:start]
				lower = lower[:start]
				break
			}
			end += start + len(tag) + 3
			value = value[:start] + " " + value[end:]
			lower = strings.ToLower(value)
		}
	}
	return value
}
