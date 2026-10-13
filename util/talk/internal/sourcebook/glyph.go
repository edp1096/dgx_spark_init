package sourcebook

import (
	_ "embed"
	"fmt"
	"net/url"
	"path"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	"golang.org/x/net/html"
)

//go:embed assets/jis0213.tsv
var jis0213Data string
var jis0213Once sync.Once
var jis0213 map[string]string
var jisGlyphPath = regexp.MustCompile(`^([12])[-_]([0-9]{1,2})[-_]([0-9]{1,2})\.(?:png|gif|jpe?g)$`)
var unicodeGlyph = regexp.MustCompile(`(?i)^(?:U\+|0x)([0-9a-f]{4,6})$`)

// EncodedGlyph records how a derived text node relates to the original image.
// It is not an OCR guess, and unknown image metadata remains an illustration.
type EncodedGlyph struct {
	URL  string `json:"url"`
	Alt  string `json:"alt,omitempty"`
	Code string `json:"code"`
	Text string `json:"text"`
}

func decodeEncodedGlyph(n *html.Node, base *url.URL) *EncodedGlyph {
	classified := false
	for _, class := range strings.Fields(attr(n, "class")) {
		classified = classified || class == "gaiji"
	}
	codepoint := strings.TrimSpace(attr(n, "data-codepoint"))
	if !classified && codepoint == "" {
		return nil
	}
	src := attr(n, "src")
	u, err := base.Parse(src)
	if err != nil || src == "" {
		return nil
	}
	var text, code string
	if m := jisGlyphPath.FindStringSubmatch(strings.ToLower(path.Base(u.Path))); m != nil {
		row, _ := strconv.Atoi(m[2])
		cell, _ := strconv.Atoi(m[3])
		code = fmt.Sprintf("%s-%02d-%02d", m[1], row, cell)
		jis0213Once.Do(func() {
			jis0213 = map[string]string{}
			for _, line := range strings.Split(jis0213Data, "\n") {
				if k, v, ok := strings.Cut(line, "\t"); ok {
					jis0213[k] = v
				}
			}
		})
		text = jis0213[code]
	}
	if codepoint != "" {
		m := unicodeGlyph.FindStringSubmatch(codepoint)
		if m == nil {
			return nil
		}
		v, err := strconv.ParseInt(m[1], 16, 32)
		if err != nil || !utf8.ValidRune(rune(v)) || !unicode.IsGraphic(rune(v)) || v == 0xfffd {
			return nil
		}
		decoded := string(rune(v))
		if text != "" && text != decoded {
			return nil
		}
		text, code = decoded, fmt.Sprintf("U+%04X", v)
	}
	if text == "" {
		return nil
	}
	return &EncodedGlyph{URL: u.String(), Alt: attr(n, "alt"), Code: code, Text: text}
}
