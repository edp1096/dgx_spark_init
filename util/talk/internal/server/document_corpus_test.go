package server

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"unicode/utf8"

	"sparktalk/internal/sourcebook"
)

// Freeze independent works before model/profile selection. This tool parses
// saved bytes only; it does not query a model or create a production task.
func TestDocumentCorpusFreeze(t *testing.T) {
	dir := os.Getenv("SPARKTALK_DOCUMENT_CORPUS_FREEZE")
	if dir == "" {
		t.Skip("opt-in source corpus preparation")
	}
	path := filepath.Join(dir, "manifest.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var manifest map[string]any
	if err = json.Unmarshal(raw, &manifest); err != nil {
		t.Fatal(err)
	}
	footer := regexp.MustCompile(`(?is)<(?:footer|section|div)\b[^>]*\bid=["']pg-footer["'][^>]*>`)
	long := map[string]bool{}
	for _, v := range manifest["documents"].([]any) {
		d := v.(map[string]any)
		name := d["work_id"].(string)
		source, err := os.ReadFile(d["raw_path"].(string))
		if err != nil {
			t.Fatal(err)
		}
		if documentTextHash(string(source)) != d["sha256"] {
			t.Fatal("source changed", name)
		}
		input := source
		if strings.Contains(d["url"].(string), "gutenberg.org") {
			start := bytes.Index(source, []byte("</header>"))
			end := footer.FindIndex(source)
			if start < 0 || len(end) == 0 || end[0] <= start {
				t.Fatal("missing Gutenberg body boundary", name)
			}
			start += len("</header>")
			input = append([]byte(`<html><head><meta charset="utf-8"></head><body>`), source[start:end[0]]...)
			input = append(input, []byte(`</body></html>`)...)
			d["body_byte_start"], d["body_byte_end"] = start, end[0]
			d["scope"] = "complete book content between publisher's START/END wrappers; hosting preamble and hosting license excluded explicitly"
		}
		book, err := sourcebook.Parse(input, "text/html", d["url"].(string), nil)
		if err != nil {
			t.Fatal(name, err)
		}
		prose := 0
		for _, slot := range book.TranslationSlots() {
			if !slot.Annotation {
				prose += utf8.RuneCountInString(slot.Text)
			}
		}
		if d["partition"] == "final" && prose >= 30000 {
			long[d["language"].(string)] = true
		}
		d["prose_chars"], d["source_view_version"], d["source_view_sha256"] = prose, book.Version, book.Hash
		d["title"] = book.Title
		prepared := filepath.Join(dir, name+".prepared.html")
		if err = os.WriteFile(prepared, input, 0600); err != nil {
			t.Fatal(err)
		}
		d["prepared_path"] = prepared
		payload, _ := json.MarshalIndent(book, "", "  ")
		if err = os.WriteFile(filepath.Join(dir, name+".book.json"), payload, 0600); err != nil {
			t.Fatal(err)
		}
		t.Log(name, d["partition"], prose, "characters", len(book.Assets), "assets")
	}
	if !long["ja"] || !long["en"] {
		t.Fatal("final corpus lacks a complete 30000+ character document in both languages", long)
	}
	manifest["state"] = "frozen_before_model_selection"
	manifest["acceptance"] = []string{"no structural loss/duplication", "review all included prose and metadata", "source-grounded blind audit by language/genre", "no unresolved major semantic error in qualified profile", "no book-specific production conditions", "end-to-end time and memory measured"}
	data, _ := json.MarshalIndent(manifest, "", "  ")
	if err = os.WriteFile(path, append(data, '\n'), 0600); err != nil {
		t.Fatal(err)
	}
}
