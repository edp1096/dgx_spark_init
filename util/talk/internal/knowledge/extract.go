package knowledge

import (
	"bytes"
	"errors"
	"fmt"
	"image/png"
	"io"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/klippa-app/go-pdfium/requests"
	"github.com/klippa-app/go-pdfium/webassembly"
	"golang.org/x/net/html"
)

var ErrOCRRequired = errors.New("document has no usable embedded text and requires OCR")

type Page struct {
	Number int
	Text   string
}

type Extractor struct{ pdfMu sync.Mutex }

func SupportsOCR(mimeType string) bool {
	switch strings.ToLower(strings.TrimSpace(strings.Split(mimeType, ";")[0])) {
	case "application/pdf", "image/png", "image/jpeg", "image/webp":
		return true
	default:
		return false
	}
}

func (e *Extractor) Extract(path, mimeType string) ([]Page, error) {
	switch mimeType {
	case "application/pdf":
		return e.extractPDF(path)
	case "text/html":
		return extractHTML(path)
	case "image/png", "image/jpeg", "image/webp":
		return nil, ErrOCRRequired
	case "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
		"application/vnd.openxmlformats-officedocument.presentationml.presentation",
		"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
		"application/vnd.oasis.opendocument.text", "application/vnd.oasis.opendocument.presentation",
		"application/vnd.oasis.opendocument.spreadsheet", "application/epub+zip", "application/vnd.hancom.hwpx":
		return extractArchiveDocument(path, mimeType)
	default:
		return extractText(path)
	}
}

func (e *Extractor) extractPDF(path string) ([]Page, error) {
	// A PDFium WASM instance peaks around 300 MiB. Serializing documents avoids
	// multiplying that transient cost when several uploads finish together.
	e.pdfMu.Lock()
	defer e.pdfMu.Unlock()

	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	pool, err := webassembly.Init(webassembly.Config{MinIdle: 0, MaxIdle: 0, MaxTotal: 1})
	if err != nil {
		return nil, err
	}
	defer pool.Close()
	instance, err := pool.GetInstance(30 * time.Second)
	if err != nil {
		return nil, err
	}
	defer instance.Close()
	document, err := instance.OpenDocument(&requests.OpenDocument{File: &data})
	if err != nil {
		return nil, err
	}
	defer instance.FPDF_CloseDocument(&requests.FPDF_CloseDocument{Document: document.Document})
	count, err := instance.FPDF_GetPageCount(&requests.FPDF_GetPageCount{Document: document.Document})
	if err != nil {
		return nil, err
	}
	pages := make([]Page, 0, count.PageCount)
	usable := 0
	for index := 0; index < count.PageCount; index++ {
		pageRef := requests.Page{ByIndex: &requests.PageByIndex{Document: document.Document, Index: index}}
		result, err := instance.GetPageText(&requests.GetPageText{Page: pageRef})
		if err != nil {
			return nil, fmt.Errorf("extract PDF page %d: %w", index+1, err)
		}
		text := normalizeText(result.Text)
		if len([]rune(text)) >= 12 {
			usable++
		}
		pages = append(pages, Page{Number: index + 1, Text: text})
	}
	if usable == 0 {
		return pages, ErrOCRRequired
	}
	return pages, nil
}

func extractText(path string) ([]Page, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, MaxSourceBytes+1))
	if err != nil {
		return nil, err
	}
	text := normalizeText(string(data))
	if text == "" {
		return nil, ErrOCRRequired
	}
	return []Page{{Number: 1, Text: text}}, nil
}

func extractHTML(path string) ([]Page, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	document, err := html.Parse(io.LimitReader(file, MaxSourceBytes+1))
	if err != nil {
		return nil, err
	}
	var output strings.Builder
	var walk func(*html.Node, bool)
	walk = func(node *html.Node, hidden bool) {
		if node.Type == html.ElementNode && (node.Data == "script" || node.Data == "style" || node.Data == "noscript") {
			hidden = true
		}
		if !hidden && node.Type == html.TextNode {
			text := strings.TrimSpace(node.Data)
			if text != "" {
				output.WriteString(text)
				output.WriteByte('\n')
			}
		}
		for child := node.FirstChild; child != nil; child = child.NextSibling {
			walk(child, hidden)
		}
	}
	walk(document, false)
	text := normalizeText(output.String())
	if text == "" {
		return nil, ErrOCRRequired
	}
	return []Page{{Number: 1, Text: text}}, nil
}

func normalizeText(value string) string {
	value = strings.ReplaceAll(value, "\x00", "")
	value = strings.ReplaceAll(value, "\r\n", "\n")
	value = strings.ReplaceAll(value, "\r", "\n")
	lines := strings.Split(value, "\n")
	cleaned := make([]string, 0, len(lines))
	empty := false
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			if !empty && len(cleaned) > 0 {
				cleaned = append(cleaned, "")
			}
			empty = true
			continue
		}
		empty = false
		cleaned = append(cleaned, line)
	}
	return strings.TrimSpace(strings.Join(cleaned, "\n"))
}

// RenderPages emits one PNG at a time so a large scanned PDF never becomes a
// directory full of temporary page images or one unbounded in-memory batch.
func (e *Extractor) RenderPages(path, mimeType string, visit func(page, total int, pngData []byte) error) (int, error) {
	if !SupportsOCR(mimeType) {
		return 0, fmt.Errorf("OCR does not support content type %s", mimeType)
	}
	if mimeType != "application/pdf" {
		data, err := os.ReadFile(path)
		if err != nil {
			return 0, err
		}
		if err := visit(1, 1, data); err != nil {
			return 0, err
		}
		return 1, nil
	}
	e.pdfMu.Lock()
	defer e.pdfMu.Unlock()
	data, err := os.ReadFile(path)
	if err != nil {
		return 0, err
	}
	pool, err := webassembly.Init(webassembly.Config{MinIdle: 0, MaxIdle: 0, MaxTotal: 1})
	if err != nil {
		return 0, err
	}
	defer pool.Close()
	instance, err := pool.GetInstance(30 * time.Second)
	if err != nil {
		return 0, err
	}
	defer instance.Close()
	document, err := instance.OpenDocument(&requests.OpenDocument{File: &data})
	if err != nil {
		return 0, err
	}
	defer instance.FPDF_CloseDocument(&requests.FPDF_CloseDocument{Document: document.Document})
	count, err := instance.FPDF_GetPageCount(&requests.FPDF_GetPageCount{Document: document.Document})
	if err != nil {
		return 0, err
	}
	for index := 0; index < count.PageCount; index++ {
		pageRef := requests.Page{ByIndex: &requests.PageByIndex{Document: document.Document, Index: index}}
		rendered, err := instance.RenderPageInDPI(&requests.RenderPageInDPI{Page: pageRef, DPI: 180})
		if err != nil {
			return index, fmt.Errorf("render PDF page %d: %w", index+1, err)
		}
		var output bytes.Buffer
		encodeErr := png.Encode(&output, rendered.Result.RenderedImage)
		rendered.Cleanup()
		if encodeErr != nil {
			return index, fmt.Errorf("encode PDF page %d: %w", index+1, encodeErr)
		}
		if err := visit(index+1, count.PageCount, output.Bytes()); err != nil {
			return index, err
		}
	}
	return count.PageCount, nil
}

func NormalizeOCRText(value string) string {
	value = strings.TrimSpace(value)
	if strings.HasPrefix(value, "```") && strings.HasSuffix(value, "```") {
		lines := strings.Split(value, "\n")
		if len(lines) >= 3 {
			value = strings.Join(lines[1:len(lines)-1], "\n")
		}
	}
	return normalizeText(value)
}
