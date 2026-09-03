package server

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"sparktalk/internal/knowledge"
	"sparktalk/internal/llm"
)

const maxWebCollectTextRunes = 20000

func webCollectToolDefinition() llm.Tool {
	parameters := json.RawMessage(`{
		"type":"object",
		"properties":{
			"url":{"type":"string","description":"Public HTTP(S) page URL to inspect"},
			"mode":{"type":"string","enum":["auto","direct","browser"],"description":"Use browser for JavaScript-rendered viewers; otherwise prefer auto"}
		},
		"required":["url"],
		"additionalProperties":false
	}`)
	return llm.Tool{Type: "function", Function: llm.ToolFunction{
		Name: "web_collect", Description: "Inspect a dynamic web page with SparkTalk Collector when ordinary page reading misses rendered content, viewer resources, tables, or document links.", Parameters: parameters,
	}}
}

func (s *Server) executeWebCollect(ctx context.Context, call llm.ToolCall) (string, error) {
	var arguments struct {
		URL  string `json:"url"`
		Mode string `json:"mode"`
	}
	if err := json.Unmarshal([]byte(call.Function.Arguments), &arguments); err != nil {
		return "", fmt.Errorf("web_collect received invalid arguments")
	}
	arguments.URL = strings.TrimSpace(arguments.URL)
	arguments.Mode = strings.ToLower(strings.TrimSpace(arguments.Mode))
	if arguments.Mode == "" {
		arguments.Mode = "auto"
	}
	if arguments.URL == "" || arguments.Mode != "auto" && arguments.Mode != "direct" && arguments.Mode != "browser" {
		return "", fmt.Errorf("web_collect requires a URL and a valid mode")
	}
	collected, err := s.collectorSnapshot().Inspect(ctx, arguments.URL, arguments.Mode)
	if err != nil {
		return "", err
	}
	type response struct {
		Title       string                          `json:"title"`
		URL         string                          `json:"url"`
		Method      string                          `json:"method"`
		ContentType string                          `json:"content_type"`
		Content     string                          `json:"content,omitempty"`
		Truncated   bool                            `json:"truncated,omitempty"`
		Links       []knowledge.CollectedLink       `json:"links,omitempty"`
		Publication *knowledge.CollectedPublication `json:"publication,omitempty"`
	}
	content, truncated := truncateCollectedText(collected.Text, maxWebCollectTextRunes)
	links := collected.Links
	if len(links) > 40 {
		links = links[:40]
	}
	publication := compactCollectedPublication(collected.Publication)
	data, err := json.Marshal(response{
		Title: collected.Manifest.Title, URL: collected.Manifest.FinalURL, Method: collected.Manifest.Method,
		ContentType: collected.Manifest.ContentType, Content: content, Truncated: truncated, Links: links, Publication: publication,
	})
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func truncateCollectedText(value string, limit int) (string, bool) {
	runes := []rune(strings.TrimSpace(value))
	if len(runes) <= limit {
		return string(runes), false
	}
	return strings.TrimSpace(string(runes[:limit])), true
}

func compactCollectedPublication(source *knowledge.CollectedPublication) *knowledge.CollectedPublication {
	if source == nil {
		return nil
	}
	result := *source
	if len(source.Pages) <= 8 {
		result.Pages = append([]knowledge.CollectedPublicationPage(nil), source.Pages...)
		return &result
	}
	result.Pages = append([]knowledge.CollectedPublicationPage(nil), source.Pages[:6]...)
	result.Pages = append(result.Pages, source.Pages[len(source.Pages)-2:]...)
	return &result
}
