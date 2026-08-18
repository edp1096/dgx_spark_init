package webtools

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"sparktalk/internal/llm"
)

const (
	maxFetchBytes  = 2 * 1024 * 1024
	maxResultBytes = 32 * 1024
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
