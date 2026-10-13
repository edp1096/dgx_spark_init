package sourcebook

import (
	"strings"
	"testing"
)

func TestEncodedGlyphPreservesProseAndRealIllustrations(t *testing.T) {
	raw := []byte(`<html><body><p>『<img class="gaiji" src="../../../gaiji/1-87/1-87-25.png" alt="encoded character">東綺譚』を読む。</p><p><img src="1-87-25.png" alt="A real illustration"></p></body></html>`)
	b, err := Parse(raw, "text/html", "https://example.com/cards/author/files/book.html", nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(b.Text, "濹") || strings.Contains(b.Text, "encoded character") || len(b.Assets) != 1 {
		t.Fatalf("glyph/illustration classification: text=%q assets=%d", b.Text, len(b.Assets))
	}
	values := map[string]string{}
	for _, block := range b.Blocks {
		values[block.ID] = block.Text
	}
	plain, err := b.Plain(values)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(plain), "『濹東綺譚』を読む。") {
		t.Fatalf("fragmented glyph: %s", plain)
	}
	var glyph *EncodedGlyph
	var walk func(*Node)
	walk = func(n *Node) {
		if n == nil {
			return
		}
		if n.Glyph != nil {
			glyph = n.Glyph
		}
		for _, c := range n.Children {
			walk(c)
		}
	}
	walk(b.Root)
	if glyph == nil || glyph.Code != "1-87-25" || glyph.Alt != "encoded character" || b.Hash != digest(raw) {
		t.Fatal("lost original glyph provenance")
	}
}

func TestEncodedGlyphUnknownConflictAndOtherCodes(t *testing.T) {
	for _, tc := range []struct {
		attrs, want string
		assets      int
	}{
		{`class="gaiji" src="1-01-01.png"`, "\u3000", 0},
		{`src="letter.png" data-codepoint="U+03B1"`, "α", 0},
		{`class="gaiji" src="1-87-25.png" data-codepoint="U+03B1"`, "", 1},
		{`class="gaiji" src="1-99-99.png"`, "", 1},
		{`class="gaiji" src="unknown.png"`, "", 1},
	} {
		b, err := Parse([]byte(`<p>before<img `+tc.attrs+` alt="unknown">after</p>`), "text/html", "https://example.com/book", nil)
		if err != nil {
			t.Fatal(err)
		}
		values := map[string]string{}
		for _, block := range b.Blocks {
			values[block.ID] = block.Text
		}
		plain, err := b.Plain(values)
		if err != nil {
			t.Fatal(err)
		}
		if len(b.Assets) != tc.assets || tc.want != "" && !strings.Contains(string(plain), tc.want) {
			t.Errorf("%s: %q assets=%d", tc.attrs, plain, len(b.Assets))
		}
	}
}
