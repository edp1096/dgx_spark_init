package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"path"
	"regexp"
	"strconv"
	"strings"
)

// PublicationAdapter isolates viewer-specific discovery from the generic HTTP
// and browser collector. New viewer families register here without changing
// the API, archive format, or SparkTalk UI.
type publicationAdapter interface {
	Name() string
	Detect(collected) bool
	Plan(context.Context, config, collected) (*publicationPlan, error)
}

type publicationPage struct {
	Number   int    `json:"number"`
	URL      string `json:"url"`
	MIMEType string `json:"mime_type"`
}

type publicationPlan struct {
	Adapter   string            `json:"adapter"`
	Title     string            `json:"title"`
	PageCount int               `json:"page_count"`
	Pages     []publicationPage `json:"pages"`
}

var publicationAdapters = []publicationAdapter{
	cbsIBookAdapter{},
}

func planPublication(ctx context.Context, cfg config, item collected) *publicationPlan {
	for _, adapter := range publicationAdapters {
		if !adapter.Detect(item) {
			continue
		}
		plan, err := adapter.Plan(ctx, cfg, item)
		if err == nil && plan != nil && plan.PageCount > 0 {
			plan.Adapter = adapter.Name()
			return plan
		}
	}
	return nil
}

// cbsIBookAdapter detects the public config schema used by the viewer engine,
// not a publisher hostname. Other deployments of the same engine therefore
// reuse this adapter automatically.
type cbsIBookAdapter struct{}

func (cbsIBookAdapter) Name() string { return "cbs-ibook" }

func (cbsIBookAdapter) Detect(item collected) bool {
	for _, resource := range item.Resources {
		parsed, err := url.Parse(resource.URL)
		if err == nil && strings.HasSuffix(strings.ToLower(parsed.Path), "/config/config.js") {
			return true
		}
	}
	return false
}

var (
	cbsPagesPattern = regexp.MustCompile(`(?s)\b(?:var|let|const)\s+e_arrPageName\s*=\s*(\[[^;]+\])\s*;`)
	cbsCountPattern = regexp.MustCompile(`\b(?:var|let|const)\s+e_totalPage\s*=\s*(\d+)\s*;`)
	cbsTypePattern  = regexp.MustCompile(`\b(?:var|let|const)\s+e_pageTypes\s*=\s*["']([^"']+)["']\s*;`)
)

func (cbsIBookAdapter) Plan(ctx context.Context, cfg config, item collected) (*publicationPlan, error) {
	for _, resource := range item.Resources {
		parsed, err := url.Parse(resource.URL)
		if err != nil || !strings.HasSuffix(strings.ToLower(parsed.Path), "/config/config.js") {
			continue
		}
		plan, err := readCBSIBookConfig(ctx, cfg, resource.URL, item.Manifest.Title)
		if err == nil && plan != nil {
			return plan, nil
		}
	}
	return nil, nil
}

func readCBSIBookConfig(ctx context.Context, cfg config, configURL, title string) (*publicationPlan, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, configURL, nil)
	if err != nil {
		return nil, err
	}
	request.Header.Set("User-Agent", collectorUserAgent)
	response, err := safeHTTPClient(cfg.Timeout).Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, nil
	}
	data, err := io.ReadAll(io.LimitReader(response.Body, (4<<20)+1))
	if err != nil {
		return nil, err
	}
	if len(data) > 4<<20 {
		return nil, nil
	}
	return parseCBSIBookConfig(data, configURL, title)
}

func parseCBSIBookConfig(data []byte, configURL, title string) (*publicationPlan, error) {
	pageMatch := cbsPagesPattern.FindSubmatch(data)
	countMatch := cbsCountPattern.FindSubmatch(data)
	typeMatch := cbsTypePattern.FindSubmatch(data)
	if len(pageMatch) != 2 || len(countMatch) != 2 || len(typeMatch) != 2 || !strings.EqualFold(string(typeMatch[1]), "pdf") {
		return nil, nil
	}
	var names []string
	if err := json.Unmarshal(pageMatch[1], &names); err != nil {
		return nil, err
	}
	declared, _ := strconv.Atoi(string(countMatch[1]))
	if len(names) == 0 || len(names) > 2000 || declared != len(names) {
		return nil, nil
	}
	parsed, err := url.Parse(configURL)
	if err != nil {
		return nil, err
	}
	parsed.RawQuery, parsed.Fragment = "", ""
	dataPath := path.Join(path.Dir(path.Dir(parsed.Path)), "data")
	pages := make([]publicationPage, 0, len(names))
	for index, name := range names {
		name = strings.TrimSpace(name)
		if name == "" || strings.ContainsAny(name, "/\\") {
			return nil, nil
		}
		pageURL := *parsed
		pageURL.Path = path.Join(dataPath, name+".pdf")
		pages = append(pages, publicationPage{Number: index + 1, URL: pageURL.String(), MIMEType: "application/pdf"})
	}
	return &publicationPlan{Title: strings.TrimSpace(title), PageCount: len(pages), Pages: pages}, nil
}
