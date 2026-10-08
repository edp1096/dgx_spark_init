package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/chromedp/cdproto/cdp"
	"github.com/chromedp/cdproto/dom"
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
		if err == nil && (mode == "direct" || direct.Manifest.ContentType != "text/html" || (len([]rune(direct.Text)) >= 300 && !hasLoadingIndicator(direct.Text))) {
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
	browserCtx, cancelBrowserDeadline := context.WithTimeout(browserCtx, 30*time.Second)
	defer cancelBrowserDeadline()
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

	var pageHTML, title, finalURL, shadowText string
	var pendingContent bool
	var screenshot []byte
	err = chromedp.Run(browserCtx,
		cdpnetwork.Enable(),
		fetch.Enable().WithPatterns([]*fetch.RequestPattern{{URLPattern: "*", RequestStage: fetch.RequestStageRequest}}),
		browserPhase("navigation", 15*time.Second, chromedp.Navigate(rawURL)),
		browserPhase("render readiness", max(8*time.Second, cfg.BrowserWait)+time.Second, chromedp.ActionFunc(func(actionCtx context.Context) error {
			var waitErr error
			shadowText, pendingContent, waitErr = waitForRenderedContent(actionCtx, cfg.BrowserWait)
			return waitErr
		})),
		browserPhase("HTML capture", 3*time.Second,
			chromedp.Title(&title), chromedp.Location(&finalURL),
			// DOM.getDocument used for shadow extraction invalidates the node
			// IDs cached by chromedp selectors. Read HTML without ByQuery.
			chromedp.Evaluate("document.documentElement.outerHTML", &pageHTML)),
		chromedp.ActionFunc(func(actionCtx context.Context) error {
			// A preview failure must not discard already-collected text.
			if captureErr := browserPhase("screenshot", 3*time.Second, chromedp.CaptureScreenshot(&screenshot)).Do(actionCtx); captureErr != nil {
				screenshot = nil
				log.Printf("collector optional preview: %v", captureErr)
			}
			return nil
		}),
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
	if shadowText != "" {
		text += "\n\n" + shadowText
	}
	if pendingContent {
		text = "Collector warning: the page still contains a loading indicator; requested content may be incomplete. Repeating the identical call does not change the wait strategy.\n\n" + text
	}
	if int64(len(text)) > maxBytes {
		return collected{}, fmt.Errorf("rendered text exceeds %d MB", maxBytes>>20)
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

// Read browser-owned DOM data, including author-created closed shadow roots.
// Do not alter the page's scripts, access checks, or attachShadow behavior.
func renderedContentSnapshot(root *cdp.Node) (string, bool) {
	var paragraphs []string
	pending := false
	var visit func(*cdp.Node, bool)
	visit = func(node *cdp.Node, shadow bool) {
		if node == nil {
			return
		}
		switch node.NodeName {
		case "SCRIPT", "STYLE", "NOSCRIPT", "TEMPLATE":
			return
		}
		for i := 0; i+1 < len(node.Attributes); i += 2 {
			if node.Attributes[i] == "aria-busy" && node.Attributes[i+1] == "true" {
				pending = true
			}
		}
		if node.NodeType == 3 {
			value := strings.TrimSpace(node.NodeValue)
			label := strings.TrimRight(strings.ToLower(value), ".…! \t\r\n")
			switch label {
			case "loading", "loading content", "please wait", "불러오는 중", "로딩 중", "불러오는 중입니다":
				pending = true
			}
			if shadow && value != "" {
				paragraphs = append(paragraphs, value)
			}
		}
		for _, child := range node.Children {
			visit(child, shadow)
		}
		for _, child := range node.ShadowRoots {
			if string(child.ShadowRootType) != "user-agent" {
				visit(child, true)
			}
		}
	}
	visit(root, false)
	return strings.Join(paragraphs, "\n"), pending
}

func waitForRenderedContent(ctx context.Context, settle time.Duration) (string, bool, error) {
	started := time.Now()
	limit := max(8*time.Second, settle)
	for {
		root, err := dom.GetDocument().WithDepth(-1).WithPierce(true).Do(ctx)
		if err != nil {
			return "", false, err
		}
		text, pending := renderedContentSnapshot(root)
		// A populated shadow tree alone is not enough: another region may
		// still show a loading indicator. Ordinary pages retain the short settle.
		if !pending && (text != "" || time.Since(started) >= settle) || time.Since(started) >= limit {
			return text, pending, nil
		}
		select {
		case <-ctx.Done():
			return "", pending, ctx.Err()
		case <-time.After(200 * time.Millisecond):
		}
	}
}

func browserPhase(name string, timeout time.Duration, actions ...chromedp.Action) chromedp.Action {
	return chromedp.ActionFunc(func(ctx context.Context) error {
		phaseCtx, cancel := context.WithTimeout(ctx, timeout)
		defer cancel()
		started := time.Now()
		for _, action := range actions {
			if err := action.Do(phaseCtx); err != nil {
				return fmt.Errorf("%s after %s: %w", name, time.Since(started).Round(time.Millisecond), err)
			}
		}
		log.Printf("collector browser %s completed in %s", name, time.Since(started).Round(time.Millisecond))
		return nil
	})
}

func hasLoadingIndicator(text string) bool {
	for _, line := range strings.Split(text, "\n") {
		switch strings.TrimRight(strings.ToLower(strings.TrimSpace(line)), ".…! \t\r\n") {
		case "loading", "loading content", "please wait", "불러오는 중", "로딩 중", "불러오는 중입니다":
			return true
		}
	}
	return false
}
