package sourcebook

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

const rubyFixture = `<html><body><h1>Title</h1><p>A <em>word</em> and <ruby><rb>漢字</rb><rp>(</rp><rt>かんじ</rt><rp>)</rp></ruby>.</p><p>Tail.</p></body></html>`

func TestTranslationParagraphsAndRubyExport(t *testing.T) {
	b, err := Parse([]byte(rubyFixture), "text/html", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	slots := b.TranslationSlots()
	values := map[string]string{}
	for _, slot := range slots {
		values[slot.ID] = ""
	}
	values[slots[0].ID] = "제목"
	values[slots[1].ID] = "단어와 한자."
	values[slots[len(slots)-1].ID] = "마지막."
	if slots[1].Group == slots[0].Group || slots[1].Group != slots[2].Group || slots[1].Group == slots[len(slots)-1].Group {
		t.Fatal("paragraph boundaries lost", slots)
	}
	n := 0
	for _, slot := range slots {
		if slot.Annotation {
			n++
		}
	}
	if n != 3 {
		t.Fatal("ruby annotations must remain separate from prose", n)
	}
	for _, build := range []func(map[string]string) ([]byte, error){b.Plain, b.Markdown, func(v map[string]string) ([]byte, error) { return b.HTML(v, true) }, b.ZIP} {
		if _, err := build(values); err != nil {
			t.Fatal("valid inline relocation rejected", err)
		}
	}
	plain, _ := b.Plain(values)
	if !bytes.Contains(plain, []byte("단어와 한자.")) || bytes.Contains(plain, []byte("かんじ")) {
		t.Fatal(string(plain))
	}
	delete(values, slots[2].ID)
	if _, err := b.HTML(values, true); err == nil {
		t.Fatal("missing empty slot accepted")
	}
	values[slots[2].ID] = ""
	values[slots[len(slots)-1].ID] = ""
	if _, err := b.Plain(values); err == nil {
		t.Fatal("entire missing paragraph accepted")
	}
}

func TestRestoreLegacyRubyPreservesEveryCheckpointCoordinate(t *testing.T) {
	raw := []byte(rubyFixture)
	b, err := Parse(raw, "text/html", "https://example.org/book", nil)
	if err != nil {
		t.Fatal(err)
	}
	b.Version = 2
	var downgrade func(*Node)
	downgrade = func(n *Node) {
		if n.Tag == "ruby" || n.Tag == "rb" || n.Tag == "rp" || n.Tag == "rt" {
			n.Tag = "div"
		}
		for _, c := range n.Children {
			downgrade(c)
		}
	}
	downgrade(b.Root)
	before, _ := json.Marshal(b.Blocks)
	if !b.RestoreInlineStructure(raw) {
		t.Fatal("legacy structure did not recover")
	}
	after, _ := json.Marshal(b.Blocks)
	if !bytes.Equal(before, after) {
		t.Fatal("checkpoint IDs/text/offsets changed")
	}
	html, _ := b.HTML(nil, true)
	if !strings.Contains(string(html), "<ruby>") {
		t.Fatal("ruby still block-level")
	}
	b.Version = 2
	b.Blocks[0].Text = "different immutable source"
	if b.RestoreInlineStructure(raw) {
		t.Fatal("mismatched source was silently replaced")
	}
}

func TestParagraphBoundariesAroundNestedBlocksAndLineBreaks(t *testing.T) {
	b, _ := Parse([]byte(`<div>before<span>inline</span><p>nested</p>after<br>next<img alt="image"></div>`), "text/html", "", nil)
	s := b.TranslationSlots()
	if s[0].Group != s[1].Group {
		t.Fatal("inline split")
	}
	for _, pair := range [][2]int{{1, 2}, {2, 3}, {3, 4}, {4, 5}} {
		if s[pair[0]].Group == s[pair[1]].Group {
			t.Fatal("paragraphs merged", s)
		}
	}
}
