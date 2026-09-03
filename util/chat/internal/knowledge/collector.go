package knowledge

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path"
	"sort"
	"strings"
	"time"
)

const (
	maxCollectorBundleBytes = MaxSourceBytes + 64<<20
	maxCollectorTextBytes   = 32 << 20
	maxCollectorManifest    = 64 << 10
	maxCollectorLinksBytes  = 4 << 20
	maxCollectorResources   = 8 << 20
	maxCollectorPublication = 2 << 20
)

type CollectorManifest struct {
	Version      int       `json:"version"`
	RequestedURL string    `json:"requested_url"`
	FinalURL     string    `json:"final_url"`
	Title        string    `json:"title"`
	Method       string    `json:"method"`
	ContentType  string    `json:"content_type"`
	RawPath      string    `json:"raw_path"`
	FetchedAt    time.Time `json:"fetched_at"`
}

type CollectedSource struct {
	Manifest    CollectorManifest
	Source      Source
	Text        string
	Links       []CollectedLink
	Publication *CollectedPublication
}

type CollectedPublication struct {
	Adapter   string                     `json:"adapter"`
	Title     string                     `json:"title"`
	PageCount int                        `json:"page_count"`
	Pages     []CollectedPublicationPage `json:"pages"`
}

type CollectedPublicationPage struct {
	Number   int    `json:"number"`
	URL      string `json:"url"`
	MIMEType string `json:"mime_type"`
}

type CollectedLink struct {
	Text     string `json:"text"`
	URL      string `json:"url"`
	Kind     string `json:"kind,omitempty"`
	MIMEType string `json:"mime_type,omitempty"`
}

type collectedResource struct {
	URL      string `json:"url"`
	MIMEType string `json:"mime_type"`
	Type     string `json:"type"`
	Status   int64  `json:"status"`
}

type CollectorClient struct {
	endpoint string
	client   *http.Client
}

func NewCollectorClient(endpoint string) *CollectorClient {
	return &CollectorClient{
		endpoint: strings.TrimRight(strings.TrimSpace(endpoint), "/"),
		client:   &http.Client{Timeout: 3 * time.Minute},
	}
}

func (c *CollectorClient) Collect(ctx context.Context, targetURL, mode string, store *Store) (CollectedSource, error) {
	if store == nil {
		return CollectedSource{}, fmt.Errorf("knowledge store is required")
	}
	return c.collect(ctx, targetURL, mode, store)
}

// Inspect reads normalized browser output without retaining the raw source.
// It is used by the chat tool for investigation; explicit knowledge imports
// continue to use Collect so persistence remains a user-directed action.
func (c *CollectorClient) Inspect(ctx context.Context, targetURL, mode string) (CollectedSource, error) {
	return c.collect(ctx, targetURL, mode, nil)
}

func (c *CollectorClient) collect(ctx context.Context, targetURL, mode string, store *Store) (CollectedSource, error) {
	if c.endpoint == "" {
		return CollectedSource{}, fmt.Errorf("SparkTalk Extra Collector endpoint is not configured")
	}
	payload, err := json.Marshal(map[string]any{"url": targetURL, "mode": mode, "max_bytes": MaxSourceBytes})
	if err != nil {
		return CollectedSource{}, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint+"/v1/collect", bytes.NewReader(payload))
	if err != nil {
		return CollectedSource{}, err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := c.client.Do(request)
	if err != nil {
		return CollectedSource{}, fmt.Errorf("SparkTalk Extra Collector: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, 64<<10))
		var body struct {
			Error string `json:"error"`
		}
		_ = json.Unmarshal(detail, &body)
		if strings.TrimSpace(body.Error) == "" {
			body.Error = strings.TrimSpace(string(detail))
		}
		return CollectedSource{}, fmt.Errorf("SparkTalk Extra Collector HTTP %d: %s", response.StatusCode, body.Error)
	}

	temporary, err := os.CreateTemp("", "sparktalk-collector-*.zip")
	if err != nil {
		return CollectedSource{}, err
	}
	temporaryPath := temporary.Name()
	defer os.Remove(temporaryPath)
	written, copyErr := io.Copy(temporary, io.LimitReader(response.Body, maxCollectorBundleBytes+1))
	closeErr := temporary.Close()
	if copyErr != nil {
		return CollectedSource{}, copyErr
	}
	if closeErr != nil {
		return CollectedSource{}, closeErr
	}
	if written < 1 || written > maxCollectorBundleBytes {
		return CollectedSource{}, fmt.Errorf("collector bundle exceeds %d MB", maxCollectorBundleBytes>>20)
	}
	file, err := os.Open(temporaryPath)
	if err != nil {
		return CollectedSource{}, err
	}
	defer file.Close()
	archive, err := zip.NewReader(file, written)
	if err != nil {
		return CollectedSource{}, fmt.Errorf("invalid collector bundle: %w", err)
	}
	return parseCollectorBundle(archive, store)
}

func parseCollectorBundle(archive *zip.Reader, store *Store) (CollectedSource, error) {
	entries := make(map[string]*zip.File, len(archive.File))
	var total uint64
	for _, entry := range archive.File {
		if !validCollectorPath(entry.Name) {
			return CollectedSource{}, fmt.Errorf("invalid collector bundle path %q", entry.Name)
		}
		total += entry.UncompressedSize64
		if total > uint64(maxCollectorBundleBytes) {
			return CollectedSource{}, fmt.Errorf("collector bundle expands beyond %d MB", maxCollectorBundleBytes>>20)
		}
		entries[entry.Name] = entry
	}
	manifestEntry := entries["manifest.json"]
	if manifestEntry == nil {
		return CollectedSource{}, fmt.Errorf("collector bundle has no manifest")
	}
	manifestData, err := readZipEntry(manifestEntry, maxCollectorManifest)
	if err != nil {
		return CollectedSource{}, err
	}
	var manifest CollectorManifest
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		return CollectedSource{}, fmt.Errorf("invalid collector manifest: %w", err)
	}
	if manifest.Version != 1 || !validCollectorPath(manifest.RawPath) || !strings.HasPrefix(manifest.RawPath, "raw/") {
		return CollectedSource{}, fmt.Errorf("unsupported collector manifest")
	}
	rawEntry := entries[manifest.RawPath]
	if rawEntry == nil || rawEntry.UncompressedSize64 < 1 || rawEntry.UncompressedSize64 > uint64(MaxSourceBytes) {
		return CollectedSource{}, fmt.Errorf("collector bundle has no valid raw source")
	}
	result := CollectedSource{Manifest: manifest}
	if store != nil {
		raw, err := rawEntry.Open()
		if err != nil {
			return CollectedSource{}, err
		}
		source, saveErr := store.SaveReader(raw, path.Base(manifest.RawPath), manifest.ContentType)
		closeErr := raw.Close()
		if saveErr != nil {
			return CollectedSource{}, saveErr
		}
		if closeErr != nil {
			return CollectedSource{}, closeErr
		}
		result.Source = source
	}
	if textEntry := entries["normalized/text.txt"]; textEntry != nil {
		textData, err := readZipEntry(textEntry, maxCollectorTextBytes)
		if err != nil {
			return CollectedSource{}, err
		}
		result.Text = normalizeText(string(textData))
	}
	var links []CollectedLink
	if linksEntry := entries["normalized/links.json"]; linksEntry != nil {
		linksData, err := readZipEntry(linksEntry, maxCollectorLinksBytes)
		if err != nil {
			return CollectedSource{}, err
		}
		if err := json.Unmarshal(linksData, &links); err != nil {
			return CollectedSource{}, fmt.Errorf("invalid collector links: %w", err)
		}
	}
	var resources []collectedResource
	if resourcesEntry := entries["normalized/resources.json"]; resourcesEntry != nil {
		resourcesData, err := readZipEntry(resourcesEntry, maxCollectorResources)
		if err != nil {
			return CollectedSource{}, err
		}
		if err := json.Unmarshal(resourcesData, &resources); err != nil {
			return CollectedSource{}, fmt.Errorf("invalid collector resources: %w", err)
		}
	}
	if publicationEntry := entries["normalized/publication.json"]; publicationEntry != nil {
		publicationData, err := readZipEntry(publicationEntry, maxCollectorPublication)
		if err != nil {
			return CollectedSource{}, err
		}
		var publication CollectedPublication
		if err := json.Unmarshal(publicationData, &publication); err != nil {
			return CollectedSource{}, fmt.Errorf("invalid collector publication: %w", err)
		}
		if validCollectedPublication(publication) {
			result.Publication = &publication
		}
	}
	result.Links = mergeCollectedLinks(links, resources)
	return result, nil
}

func validCollectedPublication(publication CollectedPublication) bool {
	if strings.TrimSpace(publication.Adapter) == "" || publication.PageCount < 1 || publication.PageCount > 2000 || len(publication.Pages) != publication.PageCount {
		return false
	}
	for index, page := range publication.Pages {
		if page.Number != index+1 || !validCollectedHTTPURL(page.URL) {
			return false
		}
	}
	return true
}

type scoredCollectedLink struct {
	link  CollectedLink
	score int
}

func mergeCollectedLinks(links []CollectedLink, resources []collectedResource) []CollectedLink {
	candidates := make([]scoredCollectedLink, 0, len(links)+len(resources))
	for _, link := range links {
		link.Text = trimRunes(strings.TrimSpace(link.Text), 300)
		link.URL = strings.TrimSpace(link.URL)
		link.Kind = "link"
		if validCollectedHTTPURL(link.URL) {
			candidates = append(candidates, scoredCollectedLink{link: link, score: 5})
		}
	}
	for _, resource := range resources {
		resource.URL = strings.TrimSpace(resource.URL)
		score := collectedResourceScore(resource)
		if score == 0 || !validCollectedHTTPURL(resource.URL) {
			continue
		}
		label := resourceName(resource.URL)
		if label == "" {
			label = resource.MIMEType
		}
		candidates = append(candidates, scoredCollectedLink{link: CollectedLink{
			Text: trimRunes(label, 300), URL: resource.URL, Kind: "resource", MIMEType: resource.MIMEType,
		}, score: score})
	}
	sort.SliceStable(candidates, func(i, j int) bool { return candidates[i].score > candidates[j].score })
	seen := make(map[string]struct{}, len(candidates))
	result := make([]CollectedLink, 0, min(100, len(candidates)))
	for _, candidate := range candidates {
		if _, exists := seen[candidate.link.URL]; exists {
			continue
		}
		seen[candidate.link.URL] = struct{}{}
		result = append(result, candidate.link)
		if len(result) >= 100 {
			break
		}
	}
	return result
}

func collectedResourceScore(resource collectedResource) int {
	mimeType := strings.ToLower(strings.TrimSpace(strings.Split(resource.MIMEType, ";")[0]))
	parsed, _ := url.Parse(resource.URL)
	extension := strings.ToLower(path.Ext(parsed.Path))
	dataExtensions := map[string]bool{
		".pdf": true, ".json": true, ".csv": true, ".tsv": true, ".xml": true, ".txt": true,
		".md": true, ".yaml": true, ".yml": true, ".toml": true, ".js": true,
	}
	if mimeType == "application/pdf" || mimeType == "application/json" || mimeType == "application/xml" ||
		mimeType == "text/csv" || mimeType == "text/tab-separated-values" || mimeType == "text/plain" || dataExtensions[extension] {
		return 4
	}
	if strings.EqualFold(resource.Type, "Fetch") && strings.Contains(mimeType, "javascript") {
		return 3
	}
	if mimeType == "image/png" || mimeType == "image/jpeg" || mimeType == "image/webp" {
		return 2
	}
	return 0
}

func validCollectedHTTPURL(value string) bool {
	return len(value) <= 4096 && (strings.HasPrefix(value, "https://") || strings.HasPrefix(value, "http://"))
}

func resourceName(value string) string {
	parsed, err := url.Parse(value)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(path.Base(parsed.Path))
}

func trimRunes(value string, limit int) string {
	runes := []rune(value)
	if len(runes) <= limit {
		return value
	}
	return strings.TrimSpace(string(runes[:limit]))
}

func validCollectorPath(name string) bool {
	if name == "" || strings.ContainsAny(name, "\\\x00") || path.IsAbs(name) {
		return false
	}
	clean := path.Clean(name)
	return clean == name && clean != "." && clean != ".." && !strings.HasPrefix(clean, "../")
}

func readZipEntry(entry *zip.File, limit int64) ([]byte, error) {
	if entry.UncompressedSize64 > uint64(limit) {
		return nil, fmt.Errorf("collector bundle entry %q is too large", entry.Name)
	}
	reader, err := entry.Open()
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	data, err := io.ReadAll(io.LimitReader(reader, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > limit {
		return nil, fmt.Errorf("collector bundle entry %q is too large", entry.Name)
	}
	return data, nil
}
