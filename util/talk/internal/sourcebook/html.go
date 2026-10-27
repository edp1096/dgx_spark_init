package sourcebook

import (
	"bytes"
	"encoding/base64"
	"fmt"
	_ "golang.org/x/image/webp"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"mime"
	"net/http"
	"net/url"
	"strings"

	"golang.org/x/net/html"
	"golang.org/x/net/html/charset"
)

var allowed = map[string]bool{}

func init() {
	for _, s := range strings.Fields("article section div p h1 h2 h3 h4 h5 h6 ul ol li blockquote pre code em strong b i u s sub sup br hr table thead tbody tfoot tr th td caption figure figcaption a span img dl dt dd ruby rb rt rp rtc") {
		allowed[s] = true
	}
}
func attr(n *html.Node, k string) string {
	for _, a := range n.Attr {
		if a.Key == k {
			return a.Val
		}
	}
	return ""
}
func textOf(n *html.Node) string {
	if n.Type == html.TextNode {
		return n.Data
	}
	var s strings.Builder
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		s.WriteString(textOf(c))
	}
	return strings.TrimSpace(s.String())
}
func (b *Book) html(data []byte, base, prefix, contentType string, load Loader) error {
	reader, err := charset.NewReader(bytes.NewReader(data), contentType)
	if err != nil {
		return err
	}
	doc, err := html.Parse(io.LimitReader(reader, MaxBytes+1))
	if err != nil {
		return err
	}
	var body *html.Node
	var preferred, article *html.Node
	baseHref := ""
	tooDeep := false
	nodeCount := 0
	var find func(*html.Node, int)
	find = func(n *html.Node, depth int) {
		nodeCount++
		if depth > 256 || nodeCount > 250000 {
			tooDeep = true
			return
		}
		if n.Type == html.ElementNode {
			if n.Data == "base" && baseHref == "" {
				baseHref = attr(n, "href")
			}
			if n.Data == "meta" && attr(n, "name") == "author" {
				if b.Metadata == nil {
					b.Metadata = map[string]any{}
				}
				if b.Metadata["source_format"] != "epub" || b.Metadata["author"] == nil {
					b.Metadata["author"] = attr(n, "content")
				}
			}
			if n.Data == "title" && b.Title == "" {
				b.Title = textOf(n)
			}
			if n.Data == "main" && preferred == nil {
				preferred = n
			}
			if n.Data == "article" && article == nil {
				article = n
			}
			if n.Data == "body" {
				body = n
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			find(c, depth+1)
		}
	}
	find(doc, 0)
	if tooDeep {
		return fmt.Errorf("HTML document structure exceeds processing limits")
	}
	if body == nil {
		body = doc
	}
	if preferred == nil {
		preferred = article
	}
	if prefix != "" {
		preferred = nil
	}
	baseURL, err := url.Parse(base)
	if err != nil {
		return err
	}
	if baseHref != "" {
		if candidate, e := baseURL.Parse(baseHref); e == nil && (candidate.Scheme == "https" || candidate.Scheme == "http") {
			baseURL = candidate
		}
	}
	chapter := ""
	svgStack := map[string]bool{}
	svgLoads := 0
	svgBytes := 0
	var visit func(*html.Node, string) *Node
	visit = func(n *html.Node, kind string) *Node {
		if n == preferred {
			start := b.chars
			defer func() {
				if b.Metadata == nil {
					b.Metadata = map[string]any{}
				}
				b.Metadata["main_start"] = start
				b.Metadata["main_end"] = b.chars
			}()
		}
		if n.Type == html.TextNode {
			if strings.TrimSpace(n.Data) == "" {
				return &Node{Text: n.Data}
			}
			return &Node{Block: b.add(n.Data, kind, chapter)}
		}
		if n.Type != html.ElementNode {
			return nil
		}
		switch n.Data {
		case "svg":
			b.warnSVG()
			return visit(svgReadingNode(n), kind)
		case "object":
			// Object fallback prose belongs to the book too. SVG objects are
			// read as static content, never executed or embedded in the output.
			if svgReference(attr(n, "data")) || attr(n, "type") == "image/svg+xml" {
				return visit(&html.Node{Type: html.ElementNode, Data: "img", Attr: []html.Attribute{{Key: "src", Val: attr(n, "data")}, {Key: "type", Val: attr(n, "type")}, {Key: "id", Val: attr(n, "id")}}}, kind)
			}
		case "script", "style", "noscript", "iframe", "embed", "template", "head":
			return nil
		}
		if n.Data == "picture" { // Choose its fallback image; srcset is handled on img below.
			holder := &Node{Tag: "span"}
			for c := n.FirstChild; c != nil; c = c.NextSibling {
				if c.Type == html.ElementNode && c.Data == "img" {
					if x := visit(c, kind); x != nil {
						holder.Children = append(holder.Children, x)
					}
				}
			}
			return holder
		}
		tag := n.Data
		if !allowed[tag] {
			tag = "div"
		}
		x := &Node{Tag: tag, Position: b.chars, Attr: map[string]string{}}
		// Preserve only this layout property, never arbitrary source CSS.
		for _, declaration := range strings.Split(attr(n, "style"), ";") {
			key, value, ok := strings.Cut(declaration, ":")
			if ok && strings.EqualFold(strings.TrimSpace(key), "white-space") {
				value = strings.TrimSpace(strings.TrimSuffix(strings.ToLower(strings.TrimSpace(value)), "!important"))
				if mode := whitespacePolicy(value); mode != "" {
					x.Whitespace = mode
				}
			}
		}
		for _, c := range strings.Fields(attr(n, "class")) {
			if c == "poem" || c == "poetry" || c == "verse" || c == "stanza" {
				x.Attr["class"] = "verse"
			}
		}
		for _, k := range []string{"id", "colspan", "rowspan", "start"} {
			if v := attr(n, k); v != "" {
				if k == "id" {
					v = prefix + v
				}
				x.Attr[k] = v
			}
		}
		if strings.HasPrefix(tag, "h") && len(tag) == 2 && tag[1] >= '1' && tag[1] <= '6' {
			chapter = textOf(n)
		}
		if tag == "a" {
			href := attr(n, "href")
			if strings.HasPrefix(href, "#") {
				if u, err := url.Parse(href); err == nil {
					x.Attr["href"] = "#" + prefix + u.Fragment
				}
			} else if u, e := baseURL.Parse(href); href != "" && e == nil && (u.Scheme == "http" || u.Scheme == "https") {
				x.Attr["href"] = u.String()
			}
		}
		if tag == "img" {
			if glyph := decodeEncodedGlyph(n, baseURL); glyph != nil {
				block := b.add(glyph.Text, "glyph", chapter)
				node := &Node{Tag: "span", Block: block, Glyph: glyph}
				if block == "" {
					node.Tag = ""
					node.Text = glyph.Text
				}
				return node
			}
			src := attr(n, "src")
			if src == "" {
				parts := strings.Split(attr(n, "srcset"), ",")
				if len(parts) > 0 {
					f := strings.Fields(parts[len(parts)-1])
					if len(f) > 0 {
						src = f[0]
					}
				}
			}
			if u, e := baseURL.Parse(src); e == nil && src != "" {
				if (svgReference(u.Path) || attr(n, "type") == "image/svg+xml") && load != nil {
					target := u.String()
					var svg *html.Node
					var err error
					if svgStack[target] || len(svgStack) >= 8 || svgLoads >= MaxAssets {
						err = fmt.Errorf("recursive SVG or SVG resource limit exceeded")
					} else {
						svgLoads++
						var data []byte
						data, _, err = load(target)
						if err == nil {
							svgBytes += len(data)
							if svgBytes > MaxAssetsBytes {
								err = fmt.Errorf("SVG resource size budget exceeded")
							} else {
								svg, err = svgDocument(data)
							}
						}
					}
					if err == nil {
						// Relative raster references inside the SVG use its URL.
						previous := baseURL
						baseURL = u
						svgStack[target] = true
						converted := visit(svg, kind)
						delete(svgStack, target)
						baseURL = previous
						converted = &Node{Tag: "figure", Attr: x.Attr, Children: []*Node{converted}}
						if alt := attr(n, "alt"); alt != "" {
							converted.Children = append(converted.Children, &Node{Tag: "figcaption", Children: []*Node{{Block: b.add(alt, "alt", chapter)}}})
						}
						return converted
					}
					b.Incomplete = true
					b.Warnings = append(b.Warnings, "SVG "+target+": "+err.Error())
					// Preserve the failure without loading the same resource again.
					x.Asset = b.asset(target, func(string) ([]byte, string, error) { return nil, "", err })
				} else {
					x.Asset = b.asset(u.String(), load)
				}
			}
			if alt := attr(n, "alt"); alt != "" {
				x.Alt = b.add(alt, "alt", chapter)
			}
			for _, k := range []string{"width", "height"} {
				v := attr(n, k)
				if len(v) < 8 && strings.Trim(v, "0123456789") == "" && v != "" {
					x.Attr[k] = v
				}
			}
			return x
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			if y := visit(c, tag); y != nil {
				x.Children = append(x.Children, y)
			}
		}
		return x
	}
	start := len(b.Root.Children)
	if id := attr(body, "id"); id != "" {
		b.Root.Children = append(b.Root.Children, &Node{Tag: "span", Attr: map[string]string{"id": prefix + id}})
	}
	for c := body.FirstChild; c != nil; c = c.NextSibling {
		if x := visit(c, "p"); x != nil {
			b.Root.Children = append(b.Root.Children, x)
		}
	}
	if prefix != "" {
		children := append([]*Node(nil), b.Root.Children[start:]...)
		b.Root.Children = append(b.Root.Children[:start], &Node{Tag: "section", Attr: map[string]string{"id": prefix}, Children: children})
	}
	return nil
}
func (b *Book) asset(target string, load Loader) string {
	for _, a := range b.Assets {
		if a.URL == target {
			return a.ID
		}
	}
	id := "img-" + digest([]byte(target))[:20]
	a := Asset{ID: id, URL: target}
	if strings.HasPrefix(target, "data:image/") {
		header, encoded, ok := strings.Cut(target, ",")
		embeddedLoad := func(string) ([]byte, string, error) {
			if !ok || !strings.HasSuffix(header, ";base64") || len(encoded) > ((MaxAssetBytes+2)/3)*4 {
				return nil, "", fmt.Errorf("invalid or oversized embedded illustration")
			}
			data, e := base64.StdEncoding.DecodeString(encoded)
			return data, strings.TrimSuffix(strings.TrimPrefix(header, "data:"), ";base64"), e
		}
		load = embeddedLoad
	}
	if len(b.Assets) >= MaxAssets {
		b.Warnings = append(b.Warnings, "image count limit reached")
		return ""
	}
	if load == nil {
		a.Pending = true
	} else {
		data, typ, err := load(target)
		used := 0
		for _, v := range b.Assets {
			used += len(v.Data)
		}
		if err != nil {
			a.Error = err.Error()
		} else if len(data) > MaxAssetBytes || used+len(data) > MaxAssetsBytes {
			a.Error = "image size budget exceeded"
		} else {
			actual := http.DetectContentType(data)
			declared, _, _ := mime.ParseMediaType(typ)
			if !strings.HasPrefix(actual, "image/") || actual == "image/svg+xml" {
				a.Error = fmt.Sprintf("unsupported image: %s (%s)", actual, declared)
			} else if config, _, err := image.DecodeConfig(bytes.NewReader(data)); err != nil || config.Width < 1 || config.Height < 1 || int64(config.Width)*int64(config.Height) > 40_000_000 {
				a.Error = "invalid illustration or dimensions exceed 40 megapixels"
			} else {
				a.Data = data
				a.Size = len(data)
				a.Hash = digest(data)
				a.MIME = actual
			}
		}
	}
	if a.Error != "" {
		b.Warnings = append(b.Warnings, "image "+target+": "+a.Error)
	}
	b.Assets = append(b.Assets, a)
	return id
}

// ResolveAssets is separate from text capture: ordinary web queries do not
// download every illustration. Task export captures each asset at most once.
func (b *Book) ResolveAssets(load Loader) {
	all := append([]Asset(nil), b.Assets...)
	b.Assets = nil
	for _, a := range all {
		if !a.Pending {
			b.Assets = append(b.Assets, a)
		}
	}
	for _, a := range all {
		if !a.Pending {
			continue
		}
		id := b.asset(a.URL, load)
		for i := range b.Assets {
			if b.Assets[i].ID == id {
				b.Assets[i].Pending = false
			}
		}
	}
}
