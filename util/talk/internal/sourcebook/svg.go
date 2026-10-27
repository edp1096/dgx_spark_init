package sourcebook

import (
	"bytes"
	"fmt"
	"strings"

	"golang.org/x/net/html"
	"golang.org/x/net/html/charset"
)

// Convert SVG's readable content into the same inert tree as XHTML. Keeping
// source SVG markup would retain scripts and URLs and would not translate text.
func svgReadingNode(n *html.Node) *html.Node {
	if n.Type == html.TextNode {
		return &html.Node{Type: html.TextNode, Data: n.Data}
	}
	if n.Type != html.ElementNode {
		return nil
	}
	tag := "span"
	switch n.Data {
	case "script", "style", "defs", "metadata", "symbol":
		return nil
	case "svg":
		tag = "figure"
	case "g", "switch", "foreignObject":
		tag = "div"
	case "title", "desc":
		tag = "figcaption"
	case "text":
		tag = "p"
	case "image":
		tag = "img"
	case "a":
		tag = "a"
	default:
		if n.Namespace != "svg" {
			tag = n.Data // foreignObject HTML still passes through the HTML sanitizer.
		}
	}
	// The HTML visitor will filter these attributes. Retain alt, whitespace,
	// and IDs from foreignObject content until that filtering step.
	out := &html.Node{Type: html.ElementNode, Data: tag, Attr: append([]html.Attribute(nil), n.Attr...)}
	if tag == "img" || tag == "a" {
		key := "href"
		if tag == "img" {
			key = "src"
		}
		ref := attr(n, "href") // xlink:href and SVG 2 href share the local name.
		if n.Namespace != "svg" && tag == "img" {
			ref = attr(n, "src")
		}
		out.Attr = append(out.Attr, html.Attribute{Key: key, Val: ref})
	}
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		if child := svgReadingNode(c); child != nil {
			out.AppendChild(child)
		}
	}
	return out
}

func svgDocument(data []byte) (*html.Node, error) {
	if len(data) > MaxAssetBytes {
		return nil, fmt.Errorf("SVG exceeds illustration size limit")
	}
	reader, err := charset.NewReader(bytes.NewReader(data), "image/svg+xml; charset=utf-8")
	if err != nil {
		return nil, err
	}
	doc, err := html.Parse(reader)
	if err != nil {
		return nil, err
	}
	var svg *html.Node
	count := 0
	var check func(*html.Node, int) error
	check = func(n *html.Node, depth int) error {
		count++
		if depth > 128 || count > 50000 {
			return fmt.Errorf("SVG structure exceeds processing limits")
		}
		if n.Type == html.ElementNode && n.Data == "svg" && svg == nil {
			svg = n
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			if err := check(c, depth+1); err != nil {
				return err
			}
		}
		return nil
	}
	if err := check(doc, 0); err != nil {
		return nil, err
	}
	if svg == nil {
		return nil, fmt.Errorf("resource has no SVG document")
	}
	return svg, nil
}

func (b *Book) warnSVG() {
	const warning = "SVG text and embedded images are extracted for translation; vector shapes and layout are not preserved"
	for _, existing := range b.Warnings {
		if existing == warning {
			return
		}
	}
	b.Warnings = append(b.Warnings, warning)
}

func svgReference(target string) bool {
	return strings.HasSuffix(strings.ToLower(target), ".svg")
}
