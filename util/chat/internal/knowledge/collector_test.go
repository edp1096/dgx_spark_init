package knowledge

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseCollectorBundleStoresRawAndNormalizedText(t *testing.T) {
	store, err := New(filepath.Join(t.TempDir(), "sparktalk.db"))
	if err != nil {
		t.Fatal(err)
	}
	manifest := CollectorManifest{Version: 1, RequestedURL: "https://example.com", FinalURL: "https://example.com/guide", Title: "장치 안내", Method: "browser", ContentType: "text/html", RawPath: "raw/page.html"}
	bundle := collectorTestBundle(t, manifest, map[string]string{
		"raw/page.html":             "<html><body>원문 자료</body></html>",
		"normalized/text.txt":       "검색 가능한 본문 자료",
		"normalized/links.json":     `[{"text":"자료 PDF","url":"https://example.com/guide.pdf"}]`,
		"normalized/resources.json": `[{"url":"https://example.com/book.pdf?x=1","mime_type":"application/pdf","type":"Fetch","status":200},{"url":"https://example.com/style.css","mime_type":"text/css","type":"Stylesheet","status":200}]`,
	})
	archive, err := zip.NewReader(bytes.NewReader(bundle), int64(len(bundle)))
	if err != nil {
		t.Fatal(err)
	}
	result, err := parseCollectorBundle(archive, store)
	if err != nil {
		t.Fatal(err)
	}
	if result.Manifest.Method != "browser" || result.Text != "검색 가능한 본문 자료" || result.Source.MIMEType != "text/html" || len(result.Links) != 2 || result.Links[1].MIMEType != "application/pdf" {
		t.Fatalf("unexpected collected source: %+v", result)
	}
	file, err := store.Open(result.Source.StoragePath)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	data := new(bytes.Buffer)
	_, _ = data.ReadFrom(file)
	if !strings.Contains(data.String(), "원문 자료") {
		t.Fatalf("raw source was not preserved: %q", data.String())
	}
}

func TestParseCollectorBundleRejectsUnsafePath(t *testing.T) {
	store, _ := New(filepath.Join(t.TempDir(), "sparktalk.db"))
	bundle := collectorTestBundle(t, CollectorManifest{Version: 1, ContentType: "text/plain", RawPath: "raw/source.txt"}, map[string]string{
		"raw/source.txt": "safe",
		"../escape":      "unsafe",
	})
	archive, _ := zip.NewReader(bytes.NewReader(bundle), int64(len(bundle)))
	if _, err := parseCollectorBundle(archive, store); err == nil || !strings.Contains(err.Error(), "invalid collector bundle path") {
		t.Fatalf("unsafe bundle was accepted: %v", err)
	}
}

func collectorTestBundle(t *testing.T, manifest CollectorManifest, entries map[string]string) []byte {
	t.Helper()
	var output bytes.Buffer
	writer := zip.NewWriter(&output)
	manifestData, _ := json.Marshal(manifest)
	file, _ := writer.Create("manifest.json")
	_, _ = file.Write(manifestData)
	for name, value := range entries {
		file, err := writer.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		_, _ = file.Write([]byte(value))
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	return output.Bytes()
}
