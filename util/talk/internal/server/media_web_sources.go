package server

import (
	"encoding/json"
	"net/url"
	"path"
	"sparktalk/internal/llm"
	"strings"
)

type discoveredMedia struct {
	SourcePage string
	Image      bool
}

func imageSourceURL(raw string) bool {
	u, err := url.Parse(raw)
	if err != nil {
		return false
	}
	switch strings.ToLower(path.Ext(u.Path)) {
	case ".png", ".jpg", ".jpeg", ".webp":
		return true
	}
	return false
}

// Tool messages are server-produced. Assistant prose, snippets, and page body
// text are deliberately excluded from this provenance check.
func discoveredMediaSource(conversation []llm.Message, target string) string {
	return discoveredMediaReference(conversation, target).SourcePage
}

func discoveredMediaReference(conversation []llm.Message, target string) discoveredMedia {
	calls := map[string]llm.FunctionCall{}
	for _, m := range conversation {
		if m.Role == "assistant" {
			for _, c := range m.ToolCalls {
				if c.ID != "" {
					calls[c.ID] = c.Function
				}
			}
			continue
		}
		if m.Role != "tool" {
			continue
		}
		call := calls[m.ToolCallID]
		name := call.Name
		if name != "web_search" && name != "web_fetch" && name != "web_collect" {
			continue
		}
		raw, ok := m.Content.(string)
		if !ok {
			continue
		}
		var result struct {
			URL     string `json:"url"`
			Error   any    `json:"error"`
			Results []struct {
				URL string `json:"url"`
			} `json:"results"`
			Images []struct {
				URL       string `json:"url"`
				SourceURL string `json:"source_url"`
			} `json:"images"`
			Links []struct {
				URL  string `json:"url"`
				Kind string `json:"kind"`
			} `json:"links"`
		}
		if json.Unmarshal([]byte(raw), &result) != nil || result.Error != nil {
			continue
		}
		for _, r := range result.Images {
			if r.URL == target {
				if result.URL != "" {
					return discoveredMedia{result.URL, true}
				}
				if r.SourceURL != "" {
					return discoveredMedia{r.SourceURL, true}
				}
				return discoveredMedia{target, true}
			}
		}
		if result.URL == target {
			return discoveredMedia{target, imageSourceURL(target)}
		}
		// A successful fetch can redirect (e.g. www.cnn.com -> edition.cnn.com).
		// Both the exact requested URL and the returned URL have been observed.
		if name == "web_fetch" || name == "web_collect" {
			var args struct {
				URL string `json:"url"`
			}
			if result.URL != "" && json.Unmarshal([]byte(call.Arguments), &args) == nil && args.URL == target {
				return discoveredMedia{result.URL, imageSourceURL(target)}
			}
		}
		for _, r := range result.Results {
			if r.URL == target {
				return discoveredMedia{target, imageSourceURL(target)}
			}
		}
		for _, r := range result.Links {
			if r.URL == target {
				image := r.Kind == "image" || imageSourceURL(target)
				if result.URL != "" {
					return discoveredMedia{result.URL, image}
				}
				return discoveredMedia{target, image}
			}
		}
	}
	return discoveredMedia{}
}
