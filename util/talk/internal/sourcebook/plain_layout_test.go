package sourcebook

import (
	"strings"
	"testing"
)

func TestPlainFixedWidthProseAndAuthoredLayout(t *testing.T) {
	prose := "The traveller arrived at the small village before the sun had\nrisen, and waited beside the gate until the old keeper appeared.\nHe was carrying a letter from the neighbouring town."
	for _, tc := range []struct{ name, text, mode string }{
		{"wrapped prose", prose, "normal"},
		{"single line", "A short sentence.", "normal"},
		{"poem", "The rain is falling\nThe night is calling", "pre-wrap"},
		{"indented verse", "The traveller arrived at the small village before the sun had\n  risen, and waited beside the gate until the old keeper appeared.", "pre-wrap"},
		{"table", "Name     Quantity\nPencils  12\nNotebooks 3", "pre-wrap"},
		{"list", "1. The traveller arrived at the small village before the sun had\n2. Another entry describes something entirely different here.", "pre-wrap"},
		{"dialogue lines", "\"Good morning,\" said the traveller.\n\"Welcome,\" replied the keeper.", "pre-wrap"},
		{"code", "if ready:\n    return value", "pre-wrap"},
		{"annotation", "［＃改頁］", "normal"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b, err := Parse([]byte(tc.text), "text/plain", "https://example.org/novel.txt", nil)
			if err != nil {
				t.Fatal(err)
			}
			if b.Root.Children[0].Whitespace != tc.mode {
				t.Fatalf("layout %s", b.Root.Children[0].Whitespace)
			}
			if b.Blocks[0].Text != tc.text {
				t.Fatal("source text changed")
			}
			if tc.mode == "normal" {
				v := identity(b)
				v[b.Blocks[0].ID] = "그는 마을에\n도착했다."
				out, _ := b.Plain(v)
				if string(out) != "그는 마을에 도착했다.\n" {
					t.Fatal(string(out))
				}
			}
		})
	}
	b, _ := Parse([]byte(prose), "text/plain", "", nil)
	before := b.Blocks[0]
	b.Version = 5
	b.Root.Children[0].Whitespace = "pre-wrap"
	if !b.RestoreInlineStructure([]byte(prose)) || b.Blocks[0] != before || b.Root.Children[0].Whitespace != "normal" {
		t.Fatal("legacy source layout recovery changed coordinates")
	}
	rendered, _ := b.HTML(identity(b), true)
	if strings.Contains(string(rendered), `<p style="white-space:pre-wrap">`) {
		t.Fatal("soft wraps still forced in HTML")
	}
}
