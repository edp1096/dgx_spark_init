package sourcebook

import "strings"

// Select retains the structure and illustrations around selected translated
// blocks. It never fills omitted passages with untranslated source text.
func (b *Book) Select(start, end int, values map[string]string) *Book {
	out := *b
	out.Blocks = nil
	out.Assets = nil
	out.Warnings = append([]string(nil), b.Warnings...)
	if end == 0 {
		end = len([]rune(b.Text))
	}
	for _, block := range b.Blocks {
		if _, ok := values[block.ID]; ok {
			out.Blocks = append(out.Blocks, block)
		}
	}
	assets := map[string]bool{}
	var prune func(*Node) (*Node, bool)
	prune = func(n *Node) (*Node, bool) {
		copyNode := *n
		copyNode.Children = nil
		copyNode.Attr = map[string]string{}
		for k, v := range n.Attr {
			copyNode.Attr[k] = v
		}
		if n.Block != "" {
			_, ok := values[n.Block]
			if !ok {
				return nil, false
			}
			return &copyNode, true
		}
		if n.Tag == "img" {
			_, altSelected := values[n.Alt]
			included := altSelected || n.Position >= start && n.Position < end
			if !included {
				return nil, false
			}
			assets[n.Asset] = true
			if !altSelected {
				copyNode.Alt = ""
			}
			return &copyNode, true
		}
		if n.Tag == "br" || n.Tag == "hr" {
			return &copyNode, n.Position >= start && n.Position < end
		}
		if n.Tag == "" {
			return &copyNode, false
		}
		meaningful := false
		for _, c := range n.Children {
			x, keep := prune(c)
			if x != nil {
				copyNode.Children = append(copyNode.Children, x)
			}
			meaningful = meaningful || keep
		}
		if !meaningful {
			return nil, false
		}
		return &copyNode, true
	}
	out.Root, _ = prune(b.Root)
	if out.Root == nil {
		out.Root = &Node{Tag: "article"}
	}
	for _, a := range b.Assets {
		if assets[a.ID] {
			out.Assets = append(out.Assets, a)
		}
	}
	ids := map[string]bool{}
	var collect func(*Node)
	collect = func(n *Node) {
		if id := n.Attr["id"]; id != "" {
			ids[id] = true
		}
		for _, c := range n.Children {
			collect(c)
		}
	}
	collect(out.Root)
	var links func(*Node)
	links = func(n *Node) {
		if href := n.Attr["href"]; strings.HasPrefix(href, "#") && !ids[strings.TrimPrefix(href, "#")] {
			delete(n.Attr, "href")
			out.Warnings = append(out.Warnings, "internal reference outside selected range: "+href)
		}
		for _, c := range n.Children {
			links(c)
		}
	}
	links(out.Root)
	return &out
}
