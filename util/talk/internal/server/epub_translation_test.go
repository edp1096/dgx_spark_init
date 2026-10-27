package server

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"path/filepath"
	"strings"
	"testing"

	"sparktalk/internal/db"
	"sparktalk/internal/knowledge"
	"sparktalk/internal/llm"
	"sparktalk/internal/media"
	"sparktalk/internal/sourcebook"
)

func TestEPUBAttachmentReadThroughWholeDocumentTranslation(t *testing.T) {
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) { mockDocumentResponse(w, r, true) })
	var err error
	s.media, err = media.New(filepath.Join(t.TempDir(), "media"))
	if err != nil {
		t.Fatal(err)
	}
	s.knowledgeIndex = &knowledge.Extractor{}
	var archive bytes.Buffer
	w := zip.NewWriter(&archive)
	for name, value := range map[string]string{
		"mimetype":               "application/epub+zip",
		"META-INF/container.xml": `<container><rootfiles><rootfile full-path="OPS/package.opf"/></rootfiles></container>`,
		"OPS/package.opf":        `<package version="3.0"><metadata><title>EPUB translation</title></metadata><manifest><item id="first" href="z.xhtml" media-type="application/xhtml+xml"/><item id="last" href="a.svg" media-type="image/svg+xml"/></manifest><spine><itemref idref="first"/><itemref idref="last" linear="no"/></spine></package>`,
		"OPS/z.xhtml":            `<html><body><h1>FIRST CHAPTER</h1>` + strings.Repeat(`<p>A complete paragraph that must be translated beyond the attachment read preview.</p>`, 170) + `</body></html>`,
		"OPS/a.svg":              `<svg xmlns="http://www.w3.org/2000/svg"><text>FINAL FOOTNOTE</text></svg>`,
	} {
		entry, err := w.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err = entry.Write([]byte(value)); err != nil {
			t.Fatal(err)
		}
	}
	if err = w.Close(); err != nil {
		t.Fatal(err)
	}
	a, err := s.media.SaveReader(bytes.NewReader(archive.Bytes()), "book.epub", "application/epub+zip", media.MaxAttachmentBytes)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = s.db.AddMessage("s", "user", "전체를 한국어로 번역해", "", nil, []db.Attachment{a}); err != nil {
		t.Fatal(err)
	}
	// A previous read must not hide newly supported content. Keep that saved
	// source untouched, since a running/completed translation may refer to it.
	old, _ := sourcebook.Parse([]byte("OLD EXTRACTION"), "text/plain", "", nil)
	hash := sha256.Sum256([]byte("s\x00" + a.ID))
	oldID := hex.EncodeToString(hash[:16])
	if err = s.db.SaveSource(oldID, "s", old, archive.Bytes()); err != nil {
		t.Fatal(err)
	}
	if err = s.media.SaveDocument(a.ID, media.DocumentCache{Fingerprint: "sparktalk-document-v1", Text: "OLD PREVIEW", PageCount: 1}); err != nil {
		t.Fatal(err)
	}
	preview, err := s.extractDocumentAttachment(context.Background(), a)
	if err != nil || !strings.Contains(preview.Text, "FINAL FOOTNOTE") || strings.Contains(preview.Text, "OLD PREVIEW") {
		t.Fatalf("stale preview: %v %v", preview, err)
	}
	reg := completionToolRegistry{handlers: make(map[string]registeredToolHandler)}
	s.registerAttachmentReader(&reg, "s")
	args, _ := json.Marshal(map[string]string{"attachment_id": a.ID})
	result, err := reg.handlers["attachment_read"](context.Background(), llm.ToolCall{Function: llm.FunctionCall{Arguments: string(args)}}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	var page struct {
		SourceID string `json:"source_id"`
		HasMore  bool   `json:"has_more"`
		Format   string `json:"source_format"`
	}
	if err = json.Unmarshal([]byte(result.Result), &page); err != nil || page.SourceID == "" || page.SourceID == oldID || !page.HasMore || page.Format != "epub" {
		t.Fatalf("invalid EPUB source: %+v %v", page, err)
	}
	run, err := s.documentAction(context.Background(), "s", documentStart{Action: "start", Kind: "translate", Sources: []string{page.SourceID}})
	if err != nil {
		t.Fatal(err)
	}
	run = waitDocument(t, s, run.ID)
	// SVG layout loss is an explicit review warning, while all text still
	// completes and exports through the ordinary document translation worker.
	if run.Status != "needs_review" || run.Document.Completed != run.Document.Total || !strings.Contains(strings.Join(run.Document.Warnings, " "), "vector shapes") {
		t.Fatalf("translation incomplete or SVG warning missing: %+v", run.Document)
	}
	out, _, err := s.db.DocumentOutput("s", run.ID, "output.txt")
	if err != nil || !bytes.Contains(out, []byte("FIRST CHAPTER")) || !bytes.Contains(out, []byte("FINAL FOOTNOTE")) || bytes.Index(out, []byte("FIRST CHAPTER")) > bytes.Index(out, []byte("FINAL FOOTNOTE")) {
		t.Fatalf("EPUB translation lost content or order: %v", err)
	}
	retained, err := s.db.Source("s", oldID, false)
	if err != nil || retained.Text != old.Text {
		t.Fatalf("old source changed: %v", err)
	}
}
