package sourcebook

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestHTMLLayoutSeparatesAuthoredBreaksFromSourceFormatting(t *testing.T) {
	cases := []struct{ name, source, want string }{
		{"wrapped prose", "<p>One long\n  sentence with <em>some\n emphasis</em> here.</p>\n\n<p>Next.</p>", "One long sentence with some emphasis here.\n\nNext.\n"},
		{"nested containers", "<div>\n<section>\n<p>First.</p>\n</section>\n</div>\n<div>\n<p>Second.</p>\n</div>", "First.\n\nSecond.\n"},
		{"aozora dialogue", "<div>\n　地の文。<br>\n「会話」<br>\n　次の文。<br>\n<br>\n場面転換。\n</div>", "　地の文。\n「会話」\n　次の文。\n\n場面転換。\n"},
		{"authored blank lines", "<p>A<br><br><br>B</p>", "A\n\n\nB\n"},
		{"inline gaps", "<p><span>One</span>\n <span>two</span> <a href='#n'>three</a>.</p>", "One two three.\n"},
		{"poem", "<div class='poem'>First line\n  indented line\n\nSecond stanza<br>Last line</div>", "First line\n  indented line\n\nSecond stanza\nLast line\n"},
		{"code", "<pre>if ok:\n    value = 1\n\n    return value</pre><p>Use <code>a  +\nb</code>.</p>", "if ok:\n    value = 1\n\n    return value\n\nUse a  +\nb.\n"},
		{"inline white-space", "<div style='white-space: pre-wrap'>A\n  B<span>C\n D</span></div><p style='white-space: pre-line'>E  F\nG</p>", "A\n  BC\n D\n\nE F\nG\n"},
		{"nested normal override", "<div style='white-space:pre-wrap'>A\n<span style='white-space:normal'>B\n C</span></div>", "A\nB C\n"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b, err := Parse([]byte(tc.source), "text/html", "https://example.org/book", nil)
			if err != nil {
				t.Fatal(err)
			}
			got, err := b.Plain(identity(b))
			if err != nil || string(got) != tc.want {
				t.Fatalf("got %q; want %q (%v)", got, tc.want, err)
			}
		})
	}
}

func TestTranslatedSoftWrapsDoNotBecomeParagraphs(t *testing.T) {
	b, _ := Parse([]byte("<p>A\nlong sentence.</p>\n<div><p>Next.</p></div>"), "text/html", "", nil)
	v := map[string]string{b.Blocks[0].ID: "긴 문장이\n중간에서\n\n끊겼다.", b.Blocks[1].ID: "다음 문단."}
	for _, build := range []func(map[string]string) ([]byte, error){b.Plain, b.Markdown} {
		got, err := build(v)
		if err != nil || string(got) != "긴 문장이 중간에서 끊겼다.\n\n다음 문단.\n" {
			t.Fatalf("unexpected layout %q %v", got, err)
		}
	}
}

func TestPlainTextKeepsItsExplicitLines(t *testing.T) {
	b, err := Parse([]byte("First line\n  second line\n\nNext paragraph."), "text/plain", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	got, err := b.Plain(identity(b))
	if err != nil || string(got) != "First line\n  second line\n\nNext paragraph.\n" {
		t.Fatalf("text layout lost: %q %v", got, err)
	}
}

func TestMarkdownKeepsExplicitBR(t *testing.T) {
	b, _ := Parse([]byte("<p>A<br>\nB</p>\n<p>C</p>"), "text/html", "", nil)
	got, err := b.Markdown(identity(b))
	if err != nil || string(got) != "A  \nB\n\nC\n" {
		t.Fatalf("hard break became soft wrap: %q %v", got, err)
	}
}

func TestMarkdownPreservesVerseAndPreformattedCode(t *testing.T) {
	b, _ := Parse([]byte("<p class='verse'>First line\nSecond line</p><pre>if ok:\n    print(\"```\")\n\n    return</pre><p>End.</p>"), "text/html", "", nil)
	got, err := b.Markdown(identity(b))
	want := "First line  \nSecond line\n\n````\nif ok:\n    print(\"```\")\n\n    return\n````\n\nEnd.\n"
	if err != nil || string(got) != want {
		t.Fatalf("Markdown lost line semantics: %q %v", got, err)
	}
}

func TestLegacyWhitespaceRecoveryKeepsCheckpointsAndRejectsMismatch(t *testing.T) {
	for _, mime := range []string{"text/html", "text/plain"} {
		raw := []byte("A\n  B")
		if mime == "text/html" {
			raw = []byte("<p style='white-space:pre-wrap'>A\n  B</p>")
		}
		b, _ := Parse(raw, mime, "", nil)
		b.Version = 3
		// Simulate a v3 snapshot, which did not store whitespace modes.
		encoded, _ := json.Marshal(b)
		var legacy map[string]any
		json.Unmarshal(encoded, &legacy)
		var strip func(map[string]any)
		strip = func(n map[string]any) {
			delete(n, "whitespace")
			if cs, ok := n["children"].([]any); ok {
				for _, c := range cs {
					strip(c.(map[string]any))
				}
			}
		}
		strip(legacy["root"].(map[string]any))
		encoded, _ = json.Marshal(legacy)
		json.Unmarshal(encoded, b)
		before, _ := json.Marshal(b.Blocks)
		if !b.RestoreInlineStructure(raw) {
			t.Fatal("layout not recovered", mime)
		}
		after, _ := json.Marshal(b.Blocks)
		if !bytes.Equal(before, after) {
			t.Fatal("immutable source changed")
		}
		got, _ := b.Plain(identity(b))
		if string(got) != "A\n  B\n" {
			t.Fatalf("recovered lines lost: %q", got)
		}
		b.Version = 3
		b.Blocks[0].Text = "different"
		if b.RestoreInlineStructure(raw) {
			t.Fatal("mismatched source accepted")
		}
	}
}

func TestHTMLWhitespacePolicyCannotInjectStyles(t *testing.T) {
	b, _ := Parse([]byte(`<p style="color:red;white-space:pre-wrap;background:url(https://example.org/a)">A
 B</p>`), "text/html", "", nil)
	got, err := b.HTML(identity(b), true)
	if err != nil || !strings.Contains(string(got), `style="white-space:pre-wrap"`) || strings.Contains(string(got), "background:") {
		t.Fatalf("unsafe/lost whitespace style: %s %v", got, err)
	}
}
