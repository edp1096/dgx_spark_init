package knowledge

import (
	"bytes"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestOCRSupportRejectsNonImageDocuments(t *testing.T) {
	for _, mimeType := range []string{"application/pdf", "image/png", "image/jpeg", "image/webp"} {
		if !SupportsOCR(mimeType) {
			t.Fatalf("expected OCR support for %s", mimeType)
		}
	}
	for _, mimeType := range []string{"text/html", "text/plain", "application/json"} {
		if SupportsOCR(mimeType) {
			t.Fatalf("unexpected OCR support for %s", mimeType)
		}
	}
	path := filepath.Join(t.TempDir(), "error.html")
	if err := os.WriteFile(path, []byte("<html>not an image</html>"), 0600); err != nil {
		t.Fatal(err)
	}
	visited := false
	_, err := (&Extractor{}).RenderPages(path, "text/html", func(int, int, []byte) error {
		visited = true
		return nil
	})
	if err == nil || !strings.Contains(err.Error(), "does not support") || visited {
		t.Fatalf("HTML reached OCR renderer: visited=%v err=%v", visited, err)
	}
}

func TestStoreExtractAndChunkText(t *testing.T) {
	store, err := New(filepath.Join(t.TempDir(), "sparktalk.db"))
	if err != nil {
		t.Fatal(err)
	}
	header := multipartHeader(t, "guide.md", []byte("# 스파크톡\n\n첫 문단입니다.\n\n두 번째 문단입니다."))
	source, err := store.Save(header)
	if err != nil {
		t.Fatal(err)
	}
	if source.MIMEType != "text/markdown" || len(source.SHA256) != 64 {
		t.Fatalf("unexpected source: %+v", source)
	}
	path, err := store.Path(source.StoragePath)
	if err != nil {
		t.Fatal(err)
	}
	pages, err := (&Extractor{}).Extract(path, source.MIMEType)
	if err != nil {
		t.Fatal(err)
	}
	chunks := ChunkPages("doc", pages)
	if len(chunks) != 1 || !strings.Contains(chunks[0].Content, "두 번째") || chunks[0].PageStart != 1 {
		t.Fatalf("unexpected chunks: %+v", chunks)
	}
	duplicate, err := store.Save(header)
	if err != nil {
		t.Fatal(err)
	}
	if duplicate.StoragePath != source.StoragePath {
		t.Fatalf("content-addressed object was not reused: %q != %q", duplicate.StoragePath, source.StoragePath)
	}
}

func TestChunkLongTextUsesBoundedOverlap(t *testing.T) {
	text := strings.Repeat("가", chunkRunes+400)
	chunks := ChunkPages("doc", []Page{{Number: 3, Text: text}})
	if len(chunks) != 2 {
		t.Fatalf("expected two chunks, got %d", len(chunks))
	}
	if got := len([]rune(chunks[0].Content)); got != chunkRunes {
		t.Fatalf("first chunk has %d runes", got)
	}
	if got := len([]rune(chunks[1].Content)); got != 580 {
		t.Fatalf("second chunk should include overlap, got %d runes", got)
	}
	if chunks[1].PageStart != 3 {
		t.Fatalf("page metadata was lost: %+v", chunks[1])
	}
}

func multipartHeader(t *testing.T, name string, data []byte) *multipart.FileHeader {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("file", name)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := part.Write(data); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	request, err := http.NewRequest(http.MethodPost, "/", &body)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", writer.FormDataContentType())
	if err := request.ParseMultipartForm(1 << 20); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { request.MultipartForm.RemoveAll() })
	return request.MultipartForm.File["file"][0]
}
