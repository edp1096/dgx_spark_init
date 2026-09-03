package main

import (
	"archive/zip"
	"bytes"
	"encoding/json"
	"net"
	"strings"
	"testing"
	"time"
)

func TestPublicIPRejectsLocalNetworks(t *testing.T) {
	for _, value := range []string{"127.0.0.1", "10.0.0.1", "192.168.1.1", "169.254.1.1", "::1", "fc00::1"} {
		if publicIP(net.ParseIP(value)) {
			t.Fatalf("local address was allowed: %s", value)
		}
	}
	for _, value := range []string{"1.1.1.1", "8.8.8.8", "2606:4700:4700::1111"} {
		if !publicIP(net.ParseIP(value)) {
			t.Fatalf("public address was blocked: %s", value)
		}
	}
}

func TestCollectorUserAgentDoesNotExposeAutomationProduct(t *testing.T) {
	if strings.Contains(strings.ToLower(collectorUserAgent), "sparktalk") || !strings.Contains(collectorUserAgent, "Mozilla/5.0") {
		t.Fatalf("collector user agent should be an ordinary browser UA: %s", collectorUserAgent)
	}
}

func TestNormalizeHTMLExtractsTextTablesAndAbsoluteLinks(t *testing.T) {
	source := `<!doctype html><title>부품 자료</title><script>ignore me</script><h1>제품 정보</h1><table><tr><th>이름</th><th>값</th></tr><tr><td>A</td><td>10</td></tr></table><a href="/download/file.pdf">자료 받기</a>`
	title, text, tables, links, err := normalizeHTML([]byte(source), "https://example.com/catalog/item")
	if err != nil {
		t.Fatal(err)
	}
	if title != "부품 자료" || !strings.Contains(text, "제품 정보") || !strings.Contains(text, "이름 | 값\nA | 10") || strings.Contains(text, "ignore me") {
		t.Fatalf("unexpected normalized text: title=%q text=%q", title, text)
	}
	if len(tables) != 1 || len(tables[0]) != 2 || tables[0][1][1] != "10" {
		t.Fatalf("unexpected tables: %+v", tables)
	}
	if len(links) != 1 || links[0].URL != "https://example.com/download/file.pdf" {
		t.Fatalf("unexpected links: %+v", links)
	}
}

func TestWriteBundlePreservesManifestAndArtifacts(t *testing.T) {
	item := collected{
		Manifest: manifest{Version: 1, RequestedURL: "https://example.com", FinalURL: "https://example.com/final", Title: "시험", Method: "direct", ContentType: "text/html", RawPath: "raw/page.html", FetchedAt: time.Unix(1, 0).UTC()},
		Raw:      []byte("<p>원문</p>"), Text: "원문", Tables: [][][]string{{{"항목", "값"}}},
		Resources:   []resourceRecord{{URL: "https://example.com/data.json", MIMEType: "application/json", Type: "XHR", Status: 200}},
		Publication: &publicationPlan{Adapter: "test-viewer", Title: "시험", PageCount: 1, Pages: []publicationPage{{Number: 1, URL: "https://example.com/1.pdf", MIMEType: "application/pdf"}}},
	}
	data, err := tempBundle(item)
	if err != nil {
		t.Fatal(err)
	}
	archive, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	entries := map[string]bool{}
	for _, file := range archive.File {
		entries[file.Name] = true
		if file.Name == "manifest.json" {
			reader, _ := file.Open()
			var decoded manifest
			if err := json.NewDecoder(reader).Decode(&decoded); err != nil || decoded.FinalURL != item.Manifest.FinalURL {
				t.Fatalf("manifest=%+v err=%v", decoded, err)
			}
			reader.Close()
		}
	}
	for _, name := range []string{"manifest.json", "raw/page.html", "normalized/text.txt", "normalized/tables.json", "normalized/resources.json", "normalized/publication.json"} {
		if !entries[name] {
			t.Fatalf("bundle entry missing: %s (%v)", name, entries)
		}
	}
}

func TestParseCBSIBookConfigBuildsPublicationPlan(t *testing.T) {
	source := []byte(`var e_totalPage = 3; var e_arrPageName = ["cover", "page-a", "page-b"]; var e_pageTypes = "pdf";`)
	plan, err := parseCBSIBookConfig(source, "https://ibook.example/CBS_iBook/5496/contents/config/config.js?v=1", "역사")
	if err != nil {
		t.Fatal(err)
	}
	if plan == nil || plan.PageCount != 3 || plan.Pages[0].Number != 1 || plan.Pages[2].URL != "https://ibook.example/CBS_iBook/5496/contents/data/page-b.pdf" {
		t.Fatalf("unexpected plan: %+v", plan)
	}
}

func TestCBSIBookAdapterDetectsViewerSchemaResource(t *testing.T) {
	adapter := cbsIBookAdapter{}
	item := collected{Resources: []resourceRecord{{URL: "https://publisher.example/viewer/contents/config/config.js?cache=1"}}}
	if !adapter.Detect(item) {
		t.Fatal("CBS iBook-compatible viewer was not detected")
	}
}
