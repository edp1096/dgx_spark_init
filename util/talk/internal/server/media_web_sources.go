package server

import (
	"encoding/json"
	"sparktalk/internal/llm"
)

// Tool messages are server-produced. Assistant prose, snippets, and page body
// text are deliberately excluded from this provenance check.
func discoveredMediaSource(conversation []llm.Message, target string) string {
	calls := map[string]string{}
	for _, m := range conversation {
		if m.Role == "assistant" {
			for _, c := range m.ToolCalls {
				if c.ID != "" {
					calls[c.ID] = c.Function.Name
				}
			}
			continue
		}
		if m.Role != "tool" {
			continue
		}
		name := calls[m.ToolCallID]
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
				URL string `json:"url"`
			} `json:"links"`
		}
		if json.Unmarshal([]byte(raw), &result) != nil || result.Error != nil {
			continue
		}
		if result.URL == target {
			return target
		}
		for _, r := range result.Results {
			if r.URL == target {
				return target
			}
		}
		for _, r := range result.Images {
			if r.URL == target {
				if result.URL != "" {
					return result.URL
				}
				return target
			}
		}
		for _, r := range result.Links {
			if r.URL == target {
				if result.URL != "" {
					return result.URL
				}
				return target
			}
		}
	}
	return ""
}
