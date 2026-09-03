package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/chromedp/cdproto/fetch"
	cdpnetwork "github.com/chromedp/cdproto/network"
	"github.com/chromedp/chromedp"
)

const collectorUserAgent = "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"

func collectURL(ctx context.Context, cfg config, rawURL, mode string, maxBytes int64) (collected, error) {
	if _, err := validatePublicURL(ctx, rawURL); err != nil {
		return collected{}, err
	}
	if mode != "browser" {
		direct, err := collectDirect(ctx, cfg, rawURL, maxBytes)
		if err == nil && (mode == "direct" || direct.Manifest.ContentType != "text/html" || len([]rune(direct.Text)) >= 300) {
			return direct, nil
		}
		if mode == "direct" {
			return collected{}, err
		}
	}
	return collectBrowser(ctx, cfg, rawURL, maxBytes)
}

func collectDirect(ctx context.Context, cfg config, rawURL string, maxBytes int64) (collected, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return collected{}, err
	}
	request.Header.Set("User-Agent", collectorUserAgent)
	request.Header.Set("Accept", "text/html,application/xhtml+xml,application/pdf,application/json,text/plain,text/csv,application/xml,image/*;q=0.9,*/*;q=0.5")
	response, err := safeHTTPClient(cfg.Timeout).Do(request)
	if err != nil {
		return collected{}, fmt.Errorf("direct fetch: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return collected{}, fmt.Errorf("direct fetch returned HTTP %d", response.StatusCode)
	}
	data, err := io.ReadAll(io.LimitReader(response.Body, maxBytes+1))
	if err != nil {
		return collected{}, err
	}
	if len(data) == 0 || int64(len(data)) > maxBytes {
		return collected{}, fmt.Errorf("collected response exceeds %d MB", maxBytes>>20)
	}
	contentType, _, _ := mime.ParseMediaType(response.Header.Get("Content-Type"))
	if contentType == "" || contentType == "application/octet-stream" {
		contentType = strings.Split(http.DetectContentType(data[:min(len(data), 512)]), ";")[0]
	}
	item := collected{Manifest: manifest{
		Version: 1, RequestedURL: rawURL, FinalURL: response.Request.URL.String(), Method: "direct",
		ContentType: contentType, RawPath: rawPath(response.Request.URL.String(), contentType), FetchedAt: time.Now().UTC(),
	}, Raw: data}
	if contentType == "text/html" || contentType == "application/xhtml+xml" {
		item.Manifest.ContentType = "text/html"
		item.Manifest.RawPath = rawPath(response.Request.URL.String(), "text/html")
		item.Manifest.Title, item.Text, item.Tables, item.Links, err = normalizeHTML(data, response.Request.URL.String())
		if err != nil {
			return collected{}, err
		}
	} else if textualContentType(contentType) {
		item.Text = strings.TrimSpace(string(data))
	}
	if item.Manifest.Title == "" {
		item.Manifest.Title = fallbackTitle(response.Request.URL.String())
	}
	return item, nil
}

func textualContentType(value string) bool {
	value = strings.ToLower(strings.TrimSpace(strings.Split(value, ";")[0]))
	return strings.HasPrefix(value, "text/") || value == "application/json" || value == "application/xml" ||
		value == "application/javascript" || value == "application/x-javascript" || value == "application/yaml"
}

func collectBrowser(ctx context.Context, cfg config, rawURL string, maxBytes int64) (collected, error) {
	profile, err := os.MkdirTemp("", "sparktalk-collector-chrome-")
	if err != nil {
		return collected{}, err
	}
	defer os.RemoveAll(profile)
	options := append(chromedp.DefaultExecAllocatorOptions[:],
		chromedp.ExecPath(cfg.ChromiumPath), chromedp.UserDataDir(profile), chromedp.UserAgent(collectorUserAgent),
		chromedp.Flag("headless", true), chromedp.Flag("no-sandbox", true), chromedp.Flag("disable-dev-shm-usage", true),
		chromedp.Flag("disable-background-networking", true), chromedp.Flag("disable-default-apps", true),
		chromedp.Flag("disable-extensions", true), chromedp.Flag("disable-sync", true), chromedp.Flag("metrics-recording-only", true),
	)
	allocatorCtx, cancelAllocator := chromedp.NewExecAllocator(ctx, options...)
	defer cancelAllocator()
	browserCtx, cancelBrowser := chromedp.NewContext(allocatorCtx)
	defer cancelBrowser()
	var resourceMu sync.Mutex
	resources := make([]resourceRecord, 0, 128)

	chromedp.ListenTarget(browserCtx, func(event any) {
		if response, ok := event.(*cdpnetwork.EventResponseReceived); ok {
			requestURL := strings.TrimSpace(response.Response.URL)
			if strings.HasPrefix(requestURL, "https://") || strings.HasPrefix(requestURL, "http://") {
				resourceMu.Lock()
				if len(resources) < 5000 {
					resources = append(resources, resourceRecord{URL: requestURL, MIMEType: response.Response.MimeType, Type: string(response.Type), Status: response.Response.Status})
				}
				resourceMu.Unlock()
			}
		}
		paused, ok := event.(*fetch.EventRequestPaused)
		if !ok {
			return
		}
		go func() {
			requestURL := paused.Request.URL
			allow := strings.HasPrefix(requestURL, "data:") || strings.HasPrefix(requestURL, "blob:") || strings.HasPrefix(requestURL, "about:")
			if !allow {
				_, err := validatePublicURL(browserCtx, requestURL)
				allow = err == nil
			}
			action := chromedp.ActionFunc(func(actionCtx context.Context) error {
				if allow {
					return fetch.ContinueRequest(paused.RequestID).Do(actionCtx)
				}
				return fetch.FailRequest(paused.RequestID, cdpnetwork.ErrorReasonBlockedByClient).Do(actionCtx)
			})
			_ = chromedp.Run(browserCtx, action)
		}()
	})

	var pageHTML, title, finalURL string
	var screenshot []byte
	err = chromedp.Run(browserCtx,
		cdpnetwork.Enable(),
		fetch.Enable().WithPatterns([]*fetch.RequestPattern{{URLPattern: "*", RequestStage: fetch.RequestStageRequest}}),
		chromedp.Navigate(rawURL), chromedp.Sleep(cfg.BrowserWait),
		chromedp.Title(&title), chromedp.Location(&finalURL), chromedp.OuterHTML("html", &pageHTML, chromedp.ByQuery),
		chromedp.CaptureScreenshot(&screenshot),
	)
	if err != nil {
		return collected{}, fmt.Errorf("browser fetch: %w", err)
	}
	if _, err := validatePublicURL(ctx, finalURL); err != nil {
		return collected{}, fmt.Errorf("browser redirect: %w", err)
	}
	if int64(len(pageHTML)) > maxBytes {
		return collected{}, fmt.Errorf("rendered page exceeds %d MB", maxBytes>>20)
	}
	normalizedTitle, text, tables, links, err := normalizeHTML([]byte(pageHTML), finalURL)
	if err != nil {
		return collected{}, err
	}
	if normalizedTitle != "" {
		title = normalizedTitle
	}
	if title == "" {
		title = fallbackTitle(finalURL)
	}
	resourceMu.Lock()
	resources = dedupeResources(resources)
	resourceMu.Unlock()
	item := collected{
		Manifest: manifest{Version: 1, RequestedURL: rawURL, FinalURL: finalURL, Title: title, Method: "browser", ContentType: "text/html", RawPath: "raw/page.html", FetchedAt: time.Now().UTC()},
		Raw:      []byte(pageHTML), Text: text, Tables: tables, Links: links, Resources: resources, Screenshot: screenshot,
	}
	item.Publication = planPublication(ctx, cfg, item)
	return item, nil
}

func dedupeResources(items []resourceRecord) []resourceRecord {
	seen := make(map[string]struct{}, len(items))
	result := make([]resourceRecord, 0, len(items))
	for _, item := range items {
		if _, exists := seen[item.URL]; exists {
			continue
		}
		seen[item.URL] = struct{}{}
		result = append(result, item)
	}
	return result
}

func tempBundle(item collected) ([]byte, error) {
	var output bytes.Buffer
	if err := writeBundle(&output, item); err != nil {
		return nil, err
	}
	return output.Bytes(), nil
}

func collectorExecutableAvailable(path string) bool {
	info, err := os.Stat(filepath.Clean(path))
	return err == nil && !info.IsDir() && info.Mode()&0111 != 0
}
