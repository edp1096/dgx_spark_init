package sourcebook

import "strings"

func whitespacePolicy(value string) string {
	switch value {
	case "pre", "pre-wrap", "break-spaces":
		return "pre-wrap"
	case "normal", "nowrap":
		return "normal"
	case "pre-line":
		return "pre-line"
	}
	return ""
}

func nodeWhitespace(n *Node, inherited string) string {
	mode := inherited
	if n.Tag == "pre" || n.Tag == "code" || n.Attr["class"] == "verse" {
		mode = "pre-wrap"
	}
	if explicit := whitespacePolicy(n.Whitespace); explicit != "" {
		mode = explicit
	}
	return mode
}

func (b *Book) textLayouts() map[string]string {
	modes := map[string]string{}
	var visit func(*Node, string)
	visit = func(n *Node, mode string) {
		if n == nil {
			return
		}
		mode = nodeWhitespace(n, mode)
		if n.Block != "" {
			modes[n.Block] = mode
		}
		if n.Alt != "" {
			modes[n.Alt] = "normal"
		}
		for _, child := range n.Children {
			visit(child, mode)
		}
	}
	visit(b.Root, "normal")
	return modes
}

func htmlSpace(r rune) bool { return r == ' ' || r == '\t' || r == '\n' || r == '\r' || r == '\f' }

// NormalizeWhitespace is used only for transient model input and rendered
// output. Immutable source text/offsets and completed translations are untouched.
// HTML collapses ASCII whitespace, not NBSP, ideographic indentation, etc.
func NormalizeWhitespace(text, mode string) string {
	if mode != "normal" && mode != "pre-line" {
		return text
	}
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")
	var out strings.Builder
	space := false
	for _, r := range text {
		if htmlSpace(r) && !(mode == "pre-line" && r == '\n') {
			if !space {
				out.WriteByte(' ')
			}
			space = true
		} else {
			out.WriteRune(r)
			space = false
		}
	}
	return out.String()
}

func (b *Book) renderValues(translated map[string]string) map[string]string {
	modes := b.textLayouts()
	values := map[string]string{}
	for _, block := range b.Blocks {
		text := block.Text
		if translated != nil {
			text = translated[block.ID]
		}
		values[block.ID] = NormalizeWhitespace(PreserveBoundary(block.Text, text), modes[block.ID])
	}
	return values
}

// Paragraph separators are minimum spacing; nesting does not add more blank
// lines. Explicit br elements accumulate separately and are never flattened.
type textLayout struct {
	out                        strings.Builder
	boundary, breaks, trailing int
	space, markdown            bool
}

func (w *textLayout) paragraph() { w.boundary = max(w.boundary, 2); w.space = false }
func (w *textLayout) br()        { w.breaks++; w.space = false }

func (w *textLayout) flush() {
	if w.out.Len() > 0 {
		count := max(w.boundary, w.trailing+w.breaks) - w.trailing
		if w.markdown && count == 1 && w.breaks == 1 && w.boundary == 0 && w.trailing == 0 {
			w.out.WriteString("  ") // Markdown hard break, not a soft source wrap.
		}
		if count > 0 {
			w.out.WriteString(strings.Repeat("\n", count))
			w.trailing += count
		}
		if w.space && w.trailing == 0 {
			w.out.WriteByte(' ')
		}
	}
	w.space, w.boundary, w.breaks = false, 0, 0
}

func (w *textLayout) text(text, mode string) {
	if mode == "pre-wrap" {
		if w.markdown {
			for i, line := range strings.Split(text, "\n") {
				if i > 0 {
					w.br()
				}
				w.literal(line)
			}
		} else {
			w.literal(text)
		}
		return
	}
	for _, r := range text {
		if mode == "pre-line" && r == '\n' {
			w.br()
			continue
		}
		if htmlSpace(r) {
			w.space = true
			continue
		}
		w.flush()
		w.out.WriteRune(r)
		w.trailing = 0
	}
}

func (w *textLayout) literal(text string) {
	if text == "" {
		return
	}
	w.flush()
	w.out.WriteString(text)
	for _, r := range text {
		if r == '\n' {
			w.trailing++
		} else {
			w.trailing = 0
		}
	}
}

func (w *textLayout) finish() []byte {
	if w.out.Len() == 0 {
		return []byte{}
	}
	if w.trailing == 0 {
		w.out.WriteByte('\n')
	}
	return []byte(w.out.String())
}

func textBlock(tag string) bool {
	switch tag {
	case "article", "section", "div", "p", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "blockquote", "pre", "table", "thead", "tbody", "tfoot", "tr", "caption", "figure", "figcaption", "dl", "dt", "dd", "hr":
		return true
	}
	return false
}
