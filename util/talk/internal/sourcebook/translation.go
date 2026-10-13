package sourcebook

import (
	"bytes"
	"fmt"
	"strings"
)

// A slot identifies a DOM text node. Group identifies the contiguous paragraph
// around inline markup; it is the minimum unit of translation coverage.
type TranslationSlot struct {
	ID, Group, Text string
	Annotation      bool
	Plain           bool
	Before          string
	// Scope restricts relocation across meaningful inline markup. Group keeps
	// the original paragraph identity for saved-output compatibility.
	Scope      string
	Whitespace string
}

func (b *Book) TranslationSlots() []TranslationSlot {
	roles := map[string]TranslationSlot{}
	protectedGroups := map[string]bool{}
	protectedSlots := map[string]bool{}
	regions := map[string]string{}
	var visit func(*Node, *string, *string, *string, bool, bool)
	visit = func(n *Node, group, scope, region *string, annotation, protected bool) {
		if n == nil {
			return
		}
		annotation = annotation || n.Tag == "rt" || n.Tag == "rp" || n.Tag == "rtc"
		boundary := false
		switch n.Tag {
		case "article", "section", "div", "p", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "blockquote", "pre", "table", "thead", "tbody", "tfoot", "tr", "th", "td", "caption", "figure", "figcaption", "dl", "dt", "dd", "br", "hr":
			boundary = true
		}
		inlineBoundary := false
		switch n.Tag {
		case "a", "em", "strong", "b", "i", "u", "s", "sub", "sup", "code", "pre":
			protected = true
			inlineBoundary = true
		}
		inlineLayout := len(n.Attr) > 0 || n.Whitespace != ""
		inlineBoundary = inlineBoundary || (!boundary && inlineLayout)
		hardBoundary := n.Tag == "img" || (inlineBoundary && len(n.Attr) > 0 && !hasTranslationContent(n)) || (n.Block == "" && strings.TrimSpace(n.Text) != "")
		protected = protected || (!boundary && inlineLayout)
		if !boundary && *group != "" && (protected || n.Tag == "img") {
			protectedGroups[*group] = true
		}
		if boundary {
			*group = "" // text on either side of a nested block is not one paragraph
			parent := group
			local := ""
			defer func() { *parent = "" }()
			group = &local
		}
		if boundary || hardBoundary {
			*scope = ""
			parent := scope
			local := ""
			defer func() { *parent = "" }()
			scope = &local
			*region = ""
			parentRegion := region
			localRegion := ""
			defer func() { *parentRegion = "" }()
			region = &localRegion
		} else if inlineBoundary {
			// Text on either side of nested emphasis may reorder within the
			// enclosing link. The parent span survives the nested span.
			local := ""
			scope = &local
		}
		if n.Block != "" {
			if *group == "" {
				*group = n.Block
			}
			if *scope == "" {
				*scope = n.Block
			}
			if *region == "" {
				*region = n.Block
			}
			regions[n.Block] = *region
			protectedSlots[n.Block] = protected
			roles[n.Block] = TranslationSlot{ID: n.Block, Group: *group, Scope: *scope, Annotation: annotation, Plain: !protected}
		}
		if n.Alt != "" {
			roles[n.Alt] = TranslationSlot{ID: n.Alt, Group: n.Alt, Scope: n.Alt}
			protectedSlots[n.Alt] = true
		}
		for _, c := range n.Children {
			visit(c, group, scope, region, annotation, protected)
		}
	}
	rootGroup := ""
	rootScope := ""
	rootRegion := ""
	visit(b.Root, &rootGroup, &rootScope, &rootRegion, false, false)
	// Whitespace-only DOM nodes have no block ID, but must still separate words
	// when adjacent plain text blocks become one model entry.
	var gaps func(*Node)
	last := ""
	var gap strings.Builder
	gaps = func(n *Node) {
		if n == nil {
			return
		}
		if n.Block != "" {
			current := roles[n.Block]
			previous := roles[last]
			if last != "" && previous.Group == current.Group {
				if strings.TrimSpace(gap.String()) == "" {
					current.Before = gap.String()
				} else {
					current.Plain, previous.Plain = false, false
					roles[last] = previous
				}
			}
			roles[n.Block] = current
			last = n.Block
			gap.Reset()
		} else {
			gap.WriteString(n.Text)
		}
		for _, child := range n.Children {
			gaps(child)
		}
	}
	gaps(b.Root)
	groupRegions := map[string]map[string]bool{}
	for id, slot := range roles {
		if !slot.Annotation && !slot.Plain {
			protectedGroups[slot.Group] = true
		}
		if !slot.Annotation && regions[id] != "" {
			if groupRegions[slot.Group] == nil {
				groupRegions[slot.Group] = map[string]bool{}
			}
			groupRegions[slot.Group][regions[id]] = true
		}
	}
	out := make([]TranslationSlot, 0, len(b.Blocks))
	layouts := b.textLayouts()
	for _, block := range b.Blocks {
		slot, ok := roles[block.ID]
		if !ok {
			slot = TranslationSlot{ID: block.ID, Group: block.ID}
		}
		slot.Plain = slot.Plain && !protectedGroups[slot.Group]
		switch {
		case protectedSlots[slot.ID]:
			slot.Scope = "span:" + slot.Scope
		case len(groupRegions[slot.Group]) > 1:
			slot.Scope = "region:" + regions[slot.ID]
		default:
			slot.Scope = ""
		}
		slot.Text = block.Text
		slot.Whitespace = layouts[block.ID]
		out = append(out, slot)
	}
	return out
}

func hasTranslationContent(n *Node) bool {
	if n.Block != "" || n.Alt != "" {
		return true
	}
	for _, child := range n.Children {
		if hasTranslationContent(child) {
			return true
		}
	}
	return false
}

// ValidateTranslation distinguishes a present empty inline slot from a missing
// result. No prose paragraph may disappear, even when words move between slots.
func ValidateTranslation(slots []TranslationSlot, output map[string]string) error {
	groups := map[string]string{}
	order := []string{}
	for _, slot := range slots {
		text, ok := output[slot.ID]
		if !ok {
			return fmt.Errorf("missing output id %q", slot.ID)
		}
		if slot.Annotation {
			continue
		}
		group := slot.Group
		if group == "" {
			group = slot.ID
		}
		if _, ok := groups[group]; !ok {
			order = append(order, group)
		}
		groups[group] += text
	}
	for _, group := range order {
		if strings.TrimSpace(groups[group]) == "" {
			return fmt.Errorf("empty translation for paragraph %q", group)
		}
	}
	return nil
}

// Old snapshots lost ruby tags and whitespace policies. Recover their structure
// from the immutable raw source, and only if every source ID, text and offset
// matches. Existing checkpoints, text, metadata and downloaded assets stay intact.
func (b *Book) RestoreInlineStructure(raw []byte) bool {
	if b.Version >= SnapshotVersion || len(raw) == 0 || b.Hash != digest(raw) {
		return false
	}
	// V3 did not record the source MIME. Try conservative plain text first;
	// actual markup cannot match its original extracted blocks/offsets.
	mimes := []string{"text/plain", "text/html"}
	if bytes.HasPrefix(raw, []byte("PK")) {
		mimes = []string{"application/zip"}
	}
	for _, mime := range mimes {
		if b.restoreStructure(raw, mime) {
			return true
		}
	}
	return false
}

func (b *Book) restoreStructure(raw []byte, mime string) bool {
	rebuilt, err := Parse(raw, mime, b.URL, nil)
	if err != nil || len(rebuilt.Blocks) != len(b.Blocks) || len(rebuilt.Assets) != len(b.Assets) {
		return false
	}
	for i, block := range b.Blocks {
		x := rebuilt.Blocks[i]
		if block.ID != x.ID || block.Text != x.Text || block.Start != x.Start || block.End != x.End {
			return false
		}
	}
	for i, asset := range b.Assets {
		if asset.ID != rebuilt.Assets[i].ID || asset.URL != rebuilt.Assets[i].URL {
			return false
		}
	}
	b.Root, b.Version = rebuilt.Root, rebuilt.Version
	return true
}
