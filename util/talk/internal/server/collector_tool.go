package server

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"sparktalk/internal/knowledge"
	"sparktalk/internal/llm"
	"sparktalk/internal/sourcebook"
)

const maxWebCollectTextRunes = 20000

func webCollectToolDefinition() llm.Tool {
	parameters := json.RawMessage(`{
		"type":"object",
		"properties":{
			"url":{"type":"string","description":"Public HTTP(S) page URL to inspect"},
                "source_id":{"type":"string"},"offset":{"type":"integer","minimum":0},
			"mode":{"type":"string","enum":["auto","direct","browser"],"description":"Use browser for JavaScript-rendered viewers; otherwise prefer auto"}
		},

		"additionalProperties":false
	}`)
	return llm.Tool{Type: "function", Function: llm.ToolFunction{
		Name: "web_collect", Description: "Inspect a dynamic web page with SparkTalk Collector when ordinary page reading misses rendered content, shadow DOM, viewer resources, tables, or document links. Browser mode waits for loading indicators to clear within a bounded window. If the requested content is still missing, report that limitation; do not claim scrolling or a longer wait occurred, and do not repeatedly offer or make identical retries.", Parameters: parameters,
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
	collected, err := s.collectSource(ctx, arguments.URL, arguments.Mode, false)
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

func (s *Server) executeWebCollectSource(ctx context.Context, session string, call llm.ToolCall) (string, error) {
	if s.db == nil || session == "" {
		return s.executeWebCollect(ctx, call)
	}
	var args struct {
		URL    string `json:"url"`
		Mode   string `json:"mode"`
		Source string `json:"source_id"`
		Offset int    `json:"offset"`
	}
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
		return "", err
	}
	if args.Source != "" {
		book, err := s.db.Source(session, args.Source, false)
		if err != nil {
			return "", err
		}
		return sourcePage(args.Source, book, args.Offset, 12000)
	}
	if args.URL == "" || args.Offset != 0 {
		return "", fmt.Errorf("provide URL once, then source_id and offset")
	}
	if args.Mode == "" {
		args.Mode = "auto"
	}
	if args.Mode != "auto" && args.Mode != "direct" && args.Mode != "browser" {
		return "", fmt.Errorf("invalid collector mode")
	}
	collected, err := s.collectSource(ctx, args.URL, args.Mode, false)
	if err != nil {
		return "", err
	}
	// Collector normalized text preserves rendered content. Its DOM/image reconstruction
	// is deliberately not claimed: direct HTML snapshots provide structured export.
	book, err := sourcebook.Parse([]byte(collected.Text), "text/plain", collected.Manifest.FinalURL, nil)
	if err != nil {
		return "", err
	}
	book.Title = collected.Manifest.Title
	book.Metadata = map[string]any{"method": collected.Manifest.Method, "content_type": collected.Manifest.ContentType, "links": collected.Links, "publication": compactCollectedPublication(collected.Publication)}
	if collected.Publication != nil {
		book.Incomplete = true
		book.Warnings = append(book.Warnings, "viewer preview only; publication pages have not been collected")
	}
	book.Warnings = append(book.Warnings, "Collector rendered-text source; original layout/images are not reconstructed")
	id, err := s.storeSource(session, book, []byte(collected.Text))
	if err != nil {
		return "", err
	}
	return sourcePage(id, book, 0, 12000)
}
