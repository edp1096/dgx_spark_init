package sourcebook

import (
	"archive/zip"
	"bytes"
	"encoding/base64"
	"fmt"
	"html"
	"sort"
	"strings"
)

const style = `body{max-width:54rem;margin:auto;padding:2rem;line-height:1.8;font-family:serif}img{max-width:100%;height:auto}figure{text-align:center}pre,code,.verse{white-space:pre-wrap}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:.4em}.missing-image{border:1px dashed #999;padding:1em}a{overflow-wrap:anywhere}`

func (b *Book) HTML(translated map[string]string, inline bool) ([]byte, error) {
	if translated != nil {
		if err := ValidateTranslation(b.TranslationSlots(), translated); err != nil {
			return nil, err
		}
	}
	var out strings.Builder
	out.WriteString("<!doctype html><html lang=\"ko\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>" + html.EscapeString(b.Title) + "</title><style>" + style + "</style></head><body>")
	assets := map[string]Asset{}
	for _, a := range b.Assets {
		assets[a.ID] = a
	}
	values := b.renderValues(translated)
	var render func(*Node)
	render = func(n *Node) {
		if n.Block != "" {
			out.WriteString(html.EscapeString(values[n.Block]))
			return
		}
		if n.Tag == "" {
			out.WriteString(html.EscapeString(n.Text))
			return
		}
		if !allowed[n.Tag] {
			return
		}
		if n.Tag == "img" {
			a, ok := assets[n.Asset]
			if !ok || a.Error != "" || len(a.Data) == 0 {
				out.WriteString("<span class=\"missing-image\">[삽화 미확보: " + html.EscapeString(values[n.Alt]) + "]</span>")
				return
			}
		}
		out.WriteString("<" + n.Tag)
		keys := []string{}
		for k := range n.Attr {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			out.WriteString(" " + k + "=\"" + html.EscapeString(n.Attr[k]) + "\"")
		}
		if mode := whitespacePolicy(n.Whitespace); mode != "" {
			out.WriteString(" style=\"white-space:" + mode + "\"")
		}
		if n.Tag == "img" {
			a := assets[n.Asset]
			src := "images/" + a.ID + extension(a.MIME)
			if inline {
				src = "data:" + a.MIME + ";base64," + base64.StdEncoding.EncodeToString(a.Data)
			}
			out.WriteString(" src=\"" + html.EscapeString(src) + "\" alt=\"" + html.EscapeString(values[n.Alt]) + "\"")
		}
		out.WriteString(">")
		for _, c := range n.Children {
			render(c)
		}
		if n.Tag != "img" && n.Tag != "br" && n.Tag != "hr" {
			out.WriteString("</" + n.Tag + ">")
		}
	}
	render(b.Root)
	out.WriteString("</body></html>")
	if out.Len() > 96<<20 {
		return nil, fmt.Errorf("rendered HTML exceeds limit")
	}
	return []byte(out.String()), nil
}
func extension(m string) string {
	switch m {
	case "image/png":
		return ".png"
	case "image/jpeg":
		return ".jpg"
	case "image/gif":
		return ".gif"
	case "image/webp":
		return ".webp"
	}
	return ".img"
}
func (b *Book) ZIP(translated map[string]string) ([]byte, error) {
	page, e := b.HTML(translated, false)
	if e != nil {
		return nil, e
	}
	var buf bytes.Buffer
	z := zip.NewWriter(&buf)
	write := func(name string, v []byte) error {
		f, e := z.Create(name)
		if e != nil {
			return e
		}
		_, e = f.Write(v)
		return e
	}
	if e = write("index.html", page); e != nil {
		return nil, e
	}
	for _, a := range b.Assets {
		if a.Error == "" && len(a.Data) > 0 {
			if e = write("images/"+a.ID+extension(a.MIME), a.Data); e != nil {
				return nil, e
			}
		}
	}
	if e = z.Close(); e != nil {
		return nil, e
	}
	return buf.Bytes(), nil
}
func (b *Book) Plain(translated map[string]string) ([]byte, error) { return b.plain(translated, false) }
func (b *Book) Markdown(translated map[string]string) ([]byte, error) {
	return b.plain(translated, true)
}
func (b *Book) plain(translated map[string]string, markdown bool) ([]byte, error) {
	if err := ValidateTranslation(b.TranslationSlots(), translated); err != nil {
		return nil, err
	}
	values := b.renderValues(translated)
	out := textLayout{markdown: markdown}
	var visit func(*Node, string)
	visit = func(n *Node, mode string) {
		if n == nil {
			return
		}
		mode = nodeWhitespace(n, mode)
		if n.Block != "" {
			out.text(values[n.Block], mode)
			return
		}
		if n.Tag == "" {
			out.text(NormalizeWhitespace(n.Text, mode), mode)
			return
		}
		if textBlock(n.Tag) {
			out.paragraph()
		}
		if markdown && n.Tag == "pre" {
			fence := markdownCodeFence(n, values)
			out.literal(fence + "\n")
			out.markdown = false
			for _, c := range n.Children {
				visit(c, mode)
			}
			if out.trailing == 0 {
				out.literal("\n")
			}
			out.literal(fence)
			out.markdown = markdown
			out.paragraph()
			return
		}
		if n.Tag == "img" {
			label := "삽화"
			if values[n.Alt] != "" {
				label += ": " + values[n.Alt]
			}
			out.text("["+label+"]", "normal")
			return
		}
		if markdown && len(n.Tag) == 2 && n.Tag[0] == 'h' && n.Tag[1] >= '1' && n.Tag[1] <= '6' {
			out.text(strings.Repeat("#", int(n.Tag[1]-'0'))+" ", "normal")
		}
		if markdown && n.Tag == "li" {
			out.text("- ", "normal")
		}
		for _, c := range n.Children {
			visit(c, mode)
		}
		if textBlock(n.Tag) {
			out.paragraph()
		}
		switch n.Tag {
		case "br":
			out.br()
		case "td", "th":
			out.text(" | ", "normal")
		}
	}
	visit(b.Root, "normal")
	return out.finish(), nil
}

func markdownCodeFence(n *Node, values map[string]string) string {
	width := 3
	var scan func(*Node)
	scan = func(node *Node) {
		text := node.Text
		if node.Block != "" {
			text = values[node.Block]
		}
		for _, run := range strings.FieldsFunc(text, func(r rune) bool { return r != '`' }) {
			width = max(width, len(run)+1)
		}
		for _, child := range node.Children {
			scan(child)
		}
	}
	scan(n)
	return strings.Repeat("`", width)
}
