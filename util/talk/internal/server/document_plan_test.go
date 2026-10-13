package server

import (
	"strings"
	"testing"
	"unicode/utf8"

	"sparktalk/internal/sourcebook"
)

func TestDocumentPlannerKeepsRubyAndInlineParagraphTogether(t *testing.T) {
	raw := `<p>` + strings.Repeat("前の段落です。", 240) + `</p><p>` + strings.Repeat("次の段落。", 80) + `<ruby>漢字<rt>かんじ</rt></ruby><a href="#n">注釈</a>` + strings.Repeat("続きの文章。", 90) + `</p><p id="n">注釈本文。</p>`
	book, err := sourcebook.Parse([]byte(raw), "text/html", "https://example.com/book", nil)
	if err != nil {
		t.Fatal(err)
	}
	units := planDocumentTranslation("s", book, 0, 0)
	owner := map[string]int{}
	covered := map[string]string{}
	for i, u := range units {
		for _, in := range u.Inputs {
			if old, ok := owner[in.Group]; ok && old != i {
				t.Fatal("ordinary paragraph split across checkpoints", in.Group)
			}
			owner[in.Group] = i
			covered[in.Block] += in.Text
		}
	}
	for _, block := range book.Blocks {
		if covered[block.ID] != block.Text {
			t.Fatal("source lost or repeated", block.ID)
		}
	}
	if len(units) < 2 {
		t.Fatal("fixture did not exercise checkpoint boundary")
	}
}

func TestDocumentPlannerLongParagraphKeepsExactSourceOffsets(t *testing.T) {
	text := strings.Repeat("Dr. Hale measured 3.14 litres. Then he closed the notebook. ", 220)
	book, err := sourcebook.Parse([]byte(text), "text/plain", "https://example.com/book", nil)
	if err != nil {
		t.Fatal(err)
	}
	units := planDocumentTranslation("s", book, 0, 0)
	var joined strings.Builder
	pos := 0
	for _, u := range units {
		for _, in := range u.Inputs {
			if in.Part != pos {
				t.Fatal("noncontiguous source coordinate", in.Part, pos)
			}
			joined.WriteString(in.Text)
			pos += utf8.RuneCountInString(in.Text)
			if strings.HasSuffix(strings.TrimSpace(in.Text), "Dr.") || strings.HasSuffix(strings.TrimSpace(in.Text), "3.") {
				t.Fatal("split abbreviation/number")
			}
		}
	}
	if joined.String() != book.Blocks[0].Text || len(units) < 2 {
		t.Fatal("long paragraph coverage", len(units))
	}
	if cut := documentSentenceSplit([]rune(strings.Repeat("x", 1000)), 500); cut != -1 {
		t.Fatal("split an unbroken token")
	}
}

func TestDocumentPlannerLongRubyParagraphPreservesBaseAndReadingOffsets(t *testing.T) {
	raw := `<p>` + strings.Repeat("前の文章は終わった。", 350) + `<ruby>漢字<rt>かんじ</rt></ruby>` + strings.Repeat("後の文章が続きます。", 450) + `</p>`
	book, err := sourcebook.Parse([]byte(raw), "text/html", "https://example.org", nil)
	if err != nil {
		t.Fatal(err)
	}
	units := planDocumentTranslation("s", book, 0, 0)
	if len(units) < 2 {
		t.Fatal("oversized DOM paragraph stayed unsplit")
	}
	covered := map[string]string{}
	offsets := map[string]int{}
	for _, u := range units {
		for i, in := range u.Inputs {
			if in.Part != offsets[in.Block] {
				t.Fatal("noncontiguous source", in)
			}
			offsets[in.Block] += utf8.RuneCountInString(in.Text)
			covered[in.Block] += in.Text
			if in.Annotation && i == 0 {
				t.Fatal("ruby reading detached from source base")
			}
		}
	}
	for _, b := range book.Blocks {
		if covered[b.ID] != b.Text {
			t.Fatal("split source changed", b.ID)
		}
	}
}
