package server

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"sparktalk/internal/config"
	"sparktalk/internal/knowledge"
	"sparktalk/internal/llm"
)

func TestWebCollectToolExposesDynamicPublicationPlan(t *testing.T) {
	collector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var output bytes.Buffer
		archive := zip.NewWriter(&output)
		manifest, _ := archive.Create("manifest.json")
		_ = json.NewEncoder(manifest).Encode(map[string]any{
			"version": 1, "requested_url": "https://example.com/viewer", "final_url": "https://example.com/viewer",
			"title": "역사 교과서", "method": "browser", "content_type": "text/html", "raw_path": "raw/page.html", "fetched_at": time.Now(),
		})
		raw, _ := archive.Create("raw/page.html")
		_, _ = raw.Write([]byte("<v-skin></v-skin>"))
		text, _ := archive.Create("normalized/text.txt")
		_, _ = text.Write([]byte("동적 전자책 뷰어"))
		publication, _ := archive.Create("normalized/publication.json")
		pages := make([]map[string]any, 10)
		for index := range pages {
			pages[index] = map[string]any{"number": index + 1, "url": "https://example.com/pages/" + string(rune('a'+index)) + ".pdf", "mime_type": "application/pdf"}
		}
		_ = json.NewEncoder(publication).Encode(map[string]any{"adapter": "cbs-ibook", "title": "역사 교과서", "page_count": 10, "pages": pages})
		_ = archive.Close()
		w.Header().Set("Content-Type", "application/zip")
		_, _ = w.Write(output.Bytes())
	}))
	defer collector.Close()

	server := &Server{cfg: config.Config{Extra: config.ExtraConfig{CollectorEndpoint: collector.URL}}, collector: knowledge.NewCollectorClient(collector.URL)}
	registry := newCompletionToolRegistry(server, "", config.ToolsConfig{Enabled: true, SearchResults: 3, Timeout: "1s"}, true, nil)
	if _, ok := registry.handlers["web_collect"]; !ok {
		t.Fatal("web_collect was not registered")
	}
	result, err := registry.execute(context.Background(), llm.ToolCall{Function: llm.FunctionCall{Name: "web_collect", Arguments: `{"url":"https://example.com/viewer","mode":"browser"}`}}, nil, nil)
	if err != nil || !strings.Contains(result.Result, `"page_count":10`) || !strings.Contains(result.Result, `"adapter":"cbs-ibook"`) {
		t.Fatalf("result=%s err=%v", result.Result, err)
	}
	var decoded struct {
		Publication struct {
			Pages []any `json:"pages"`
		} `json:"publication"`
	}
	if json.Unmarshal([]byte(result.Result), &decoded) != nil || len(decoded.Publication.Pages) != 8 {
		t.Fatalf("publication samples were not compacted: %s", result.Result)
	}
}
