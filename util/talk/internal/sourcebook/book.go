// Package sourcebook keeps document structure separate from model output.
package sourcebook

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strings"
	"unicode/utf8"
)

const MaxBytes = 32 << 20
const MaxText = 4_000_000
const MaxAssetBytes = 8 << 20
const MaxAssetsBytes = 48 << 20
const MaxAssets = 256
const SnapshotVersion = 6

type Block struct {
	ID      string `json:"id"`
	Kind    string `json:"kind"`
	Text    string `json:"text"`
	Start   int    `json:"start"`
	End     int    `json:"end"`
	Chapter string `json:"chapter,omitempty"`
}
type Node struct {
	Glyph      *EncodedGlyph     `json:"glyph,omitempty"`
	Whitespace string            `json:"whitespace,omitempty"`
	Position   int               `json:"position,omitempty"`
	Tag        string            `json:"tag,omitempty"`
	Attr       map[string]string `json:"attr,omitempty"`
	Text       string            `json:"text,omitempty"`
	Block      string            `json:"block,omitempty"`
	Alt        string            `json:"alt,omitempty"`
	Asset      string            `json:"asset,omitempty"`
	Children   []*Node           `json:"children,omitempty"`
}
type Asset struct {
	Size    int    `json:"size,omitempty"`
	Pending bool   `json:"pending,omitempty"`
	Hash    string `json:"hash,omitempty"`
	ID      string `json:"id"`
	URL     string `json:"url"`
	MIME    string `json:"mime,omitempty"`
	Error   string `json:"error,omitempty"`
	Data    []byte `json:"-"`
}
type Book struct {
	Incomplete bool           `json:"incomplete,omitempty"`
	Metadata   map[string]any `json:"metadata,omitempty"`
	builder    strings.Builder
	chars      int
	Version    int      `json:"version"`
	Title      string   `json:"title"`
	URL        string   `json:"url"`
	Hash       string   `json:"hash"`
	Text       string   `json:"text"`
	Blocks     []Block  `json:"blocks"`
	Root       *Node    `json:"root"`
	Assets     []Asset  `json:"assets,omitempty"`
	Warnings   []string `json:"warnings,omitempty"`
}
type Loader func(string) ([]byte, string, error)

func digest(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }
func (b *Book) add(text, kind, chapter string) string {
	if strings.TrimSpace(text) == "" {
		return ""
	}
	id := fmt.Sprintf("b%06d", len(b.Blocks)+1)
	start := b.chars
	if start > 0 {
		b.builder.WriteString("\n")
		start++
	}
	b.builder.WriteString(text)
	b.chars = start + utf8.RuneCountInString(text)
	b.Blocks = append(b.Blocks, Block{ID: id, Kind: kind, Text: text, Start: start, End: start + utf8.RuneCountInString(text), Chapter: chapter})
	return id
}
func (b *Book) Validate() error {
	if b.Root == nil || len(b.Blocks) == 0 {
		return fmt.Errorf("document has no usable text")
	}
	if utf8.RuneCountInString(b.Text) > MaxText {
		return fmt.Errorf("document exceeds %d characters", MaxText)
	}
	return nil
}
func (b *Book) Page(offset, limit int) (map[string]any, error) {
	text := []rune(b.Text)
	if offset < 0 || offset > len(text) {
		return nil, fmt.Errorf("offset outside source")
	}
	end := min(len(text), offset+limit)
	p := map[string]any{"url": b.URL, "title": b.Title, "content": string(text[offset:end]), "offset": offset, "next_offset": end, "total_chars": len(text), "has_more": end < len(text), "complete": !b.Incomplete, "warnings": b.Warnings}
	p["complete"] = !b.Incomplete
	for k, v := range b.Metadata {
		p[k] = v
	}
	return p, nil
}
func Parse(data []byte, mime, base string, load Loader) (*Book, error) {
	if strings.HasPrefix(strings.ToLower(mime), "image/") {
		return nil, fmt.Errorf("image/OCR sources are not supported by document text processing")
	}
	if len(data) > MaxBytes {
		return nil, fmt.Errorf("source exceeds %d bytes", MaxBytes)
	}
	b := &Book{Version: SnapshotVersion, URL: base, Hash: digest(data), Root: &Node{Tag: "article"}}
	lower := strings.ToLower(mime)
	var err error
	switch {
	case strings.Contains(lower, "zip"), strings.Contains(lower, "epub"), len(data) > 4 && string(data[:2]) == "PK":
		err = b.archive(data, load, strings.Contains(lower, "epub"))
	case strings.Contains(lower, "html"), strings.Contains(strings.ToLower(string(data[:min(512, len(data))])), "<html"):
		err = b.html(data, base, "", mime, load)
	default:
		if !utf8.Valid(data) || strings.ContainsRune(string(data), 0) {
			return nil, fmt.Errorf("source is not supported readable text")
		}
		s := strings.ReplaceAll(string(data), "\r\n", "\n")
		b.Metadata = map[string]any{"source_format": "text"}
		lines := strings.Split(s, "\n")
		for _, line := range lines[:min(120, len(lines))] {
			key, value, ok := strings.Cut(line, ":")
			if !ok {
				continue
			}
			value = strings.TrimSpace(value)
			switch strings.TrimSpace(key) {
			case "Title":
				if value != "" {
					b.Title = value
				}
			case "Author", "Language":
				if b.Metadata == nil {
					b.Metadata = map[string]any{}
				}
				b.Metadata[strings.ToLower(strings.TrimSpace(key))] = value
			}
		}
		chapter := ""
		for _, p := range strings.Split(s, "\n\n") {
			p = strings.TrimSpace(p)
			if p == "" {
				continue
			}
			kind := "p"
			line := strings.TrimSpace(strings.SplitN(p, "\n", 2)[0])
			if len(p) < 180 && (strings.HasPrefix(strings.ToUpper(line), "CHAPTER ") || strings.HasPrefix(line, "제") && strings.Contains(line, "장")) {
				kind = "h2"
				chapter = line
			}
			if b.Title == "" {
				b.Title = line
			}
			b.Root.Children = append(b.Root.Children, &Node{Tag: kind, Whitespace: plainTextWhitespace(p, kind), Children: []*Node{{Block: b.add(p, kind, chapter)}}})
		}
	}
	b.Text = b.builder.String()
	if err != nil {
		return nil, err
	}
	if err = b.Validate(); err != nil {
		return nil, err
	}
	return b, nil
}

func (b *Book) VerifyAssets() {
	for i := range b.Assets {
		a := &b.Assets[i]
		if a.Pending || a.Error != "" {
			continue
		}
		if len(a.Data) == 0 || a.Size > 0 && len(a.Data) != a.Size || a.Hash != "" && digest(a.Data) != a.Hash {
			a.Data = nil
			a.Error = "stored illustration is missing or failed integrity verification"
		}
	}
}
