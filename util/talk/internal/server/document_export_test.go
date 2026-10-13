package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

func TestDocumentTextConversionUsesCommittedWholeBookAndBoundedRoute(t *testing.T) {
	var modelCalls, conversions atomic.Int32
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) { modelCalls.Add(1); mockDocumentResponse(w, r, true) })
	source := fixtureSource(t, s, "The book has already been translated.")
	run, err := s.documentAction(context.Background(), "s", documentStart{Action: "start", Kind: "translate", Sources: []string{source}})
	if err != nil {
		t.Fatal(err)
	}
	run = waitDocument(t, s, run.ID)
	text := "FIRST_MARKER\n\n" + strings.Repeat("한국어 전권 본문을 빠짐없이 전달합니다. ", 12000) + "\n\nLAST_MARKER"
	if err = s.db.SaveDocumentOutput("s", run.ID, "output.txt", "text/plain", []byte(text)); err != nil {
		t.Fatal(err)
	}
	converter := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conversions.Add(1)
		if r.URL.Path != "/v1/documents/text" {
			t.Error("whole-book conversion used the short tool route", r.URL.Path)
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			return
		}
		if body["text"] != text || body["format"] != "pdf" || body["sections"] != nil {
			t.Error("committed text was truncated or changed")
		}
		if _, ok := body["title"]; ok {
			t.Error("conversion must not add a heading to committed text")
		}
		json.NewEncoder(w).Encode(map[string]any{"files": []any{map[string]string{"name": "document.pdf", "mime": "application/pdf", "data": base64.StdEncoding.EncodeToString([]byte("%PDF-1.7\nconverted"))}}})
	}))
	defer converter.Close()
	s.cfg.Extra.DocumentsEnabled = true
	s.cfg.Extra.DocumentsEndpoint = converter.URL
	if _, err = s.documentAction(context.Background(), "s", documentStart{Action: "export", RunID: run.ID, Format: "pdf"}); err != nil {
		t.Fatal(err)
	}
	if conversions.Load() != 1 || modelCalls.Load() != 1 {
		t.Fatal("conversion re-invoked model", conversions.Load(), modelCalls.Load())
	}
	if err = s.db.SaveDocumentOutput("s", run.ID, "output.txt", "text/plain", []byte(strings.Repeat("x", 1000001))); err != nil {
		t.Fatal(err)
	}
	if _, err = s.documentAction(context.Background(), "s", documentStart{Action: "export", RunID: run.ID, Format: "docx"}); err == nil || !strings.Contains(err.Error(), "100만 자") {
		t.Fatal("oversized conversion was not rejected before HTTP", err)
	}
	if conversions.Load() != 1 {
		t.Fatal("oversized text reached service")
	}
}
