package sourcebook

import (
	"archive/zip"
	"bytes"
	"fmt"
	"image"
	"image/png"
	"io"
	"strings"
	"testing"
)

func picture() []byte {
	var b bytes.Buffer
	png.Encode(&b, image.NewRGBA(image.Rect(0, 0, 2, 2)))
	return b.Bytes()
}
func identity(b *Book) map[string]string {
	m := map[string]string{}
	for _, v := range b.Blocks {
		m[v.ID] = v.Text
	}
	return m
}
func archiveFixture(files map[string][]byte) []byte {
	var b bytes.Buffer
	z := zip.NewWriter(&b)
	for n, v := range files {
		f, _ := z.Create(n)
		f.Write(v)
	}
	z.Close()
	return b.Bytes()
}
func TestHTMLStructureAndOfflineImages(t *testing.T) {
	input := `<!doctype html><html><head><title>Book</title><script>bad()</script></head><body><nav><a href="#ch1">Chapter I</a></nav><h2 id="ch1">Chapter I</h2><p>Hello <em>world</em>.<a href="#note">1</a></p><figure><img src="pics/a.png" alt="Rabbit"><figcaption>A rabbit</figcaption></figure><pre>one\ntwo</pre><table><tr><td>cell</td></tr></table><p id="note">Footnote <a href="#ch1">back</a></p><script>alert(1)</script></body></html>`
	calls := 0
	b, e := Parse([]byte(input), "text/html", "https://example.org/book/index.html", func(u string) ([]byte, string, error) {
		calls++
		if u != "https://example.org/book/pics/a.png" {
			t.Fatal(u)
		}
		return picture(), "image/png", nil
	})
	if e != nil {
		t.Fatal(e)
	}
	if calls != 1 || len(b.Assets) != 1 || strings.Contains(b.Text, "bad()") || strings.Contains(b.Text, "alert") {
		t.Fatalf("bad extraction: %+v", b)
	}
	translated := identity(b)
	for _, x := range b.Blocks {
		if x.Text == "world" {
			translated[x.ID] = "세계"
		}
	}
	data, e := b.ZIP(translated)
	if e != nil {
		t.Fatal(e)
	}
	z, e := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if e != nil {
		t.Fatal(e)
	}
	files := map[string][]byte{}
	for _, f := range z.File {
		r, _ := f.Open()
		files[f.Name], _ = io.ReadAll(r)
		r.Close()
	}
	html := string(files["index.html"])
	for _, want := range []string{`<em>세계</em>`, `href="#note"`, `id="ch1"`, `<figcaption>`, `<table>`, `<pre>`, `src="images/`} {
		if !strings.Contains(html, want) {
			t.Fatalf("missing %s: %s", want, html)
		}
	}
	if len(files) != 2 || strings.Contains(html, "<script") {
		t.Fatal("offline package/active markup incorrect")
	}
	inline, e := b.HTML(translated, true)
	if e != nil || !strings.Contains(string(inline), "data:image/png;base64,") {
		t.Fatal("inline image missing", e)
	}
	delete(translated, b.Blocks[0].ID)
	if _, e = b.ZIP(translated); e == nil {
		t.Fatal("missing translation silently substituted")
	}
}
func TestUnicodePagingExactCoverage(t *testing.T) {
	b, e := Parse([]byte(strings.Repeat("한글😀\n\n", 20000)), "text/plain", "", nil)
	if e != nil {
		t.Fatal(e)
	}
	var out strings.Builder
	for offset := 0; ; {
		p, e := b.Page(offset, 12000)
		if e != nil {
			t.Fatal(e)
		}
		out.WriteString(p["content"].(string))
		offset = p["next_offset"].(int)
		if !p["has_more"].(bool) {
			break
		}
	}
	if out.String() != b.Text {
		t.Fatal("paging changed text")
	}
	if _, e = b.Page(-1, 10); e == nil {
		t.Fatal("negative offset accepted")
	}
}
func TestEPUBUsesSpineNotFilenameOrder(t *testing.T) {
	files := map[string][]byte{"META-INF/container.xml": []byte(`<container><rootfiles><rootfile full-path="OPS/book.opf"/></rootfiles></container>`), "OPS/book.opf": []byte(`<package><manifest><item id="a" href="z.xhtml" media-type="application/xhtml+xml"/><item id="b" href="a.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="a"/><itemref idref="b"/></spine></package>`), "OPS/z.xhtml": []byte(`<html><body><h2 id="chapter">First</h2><p>FIRST TEXT <a href="a.xhtml#end">next</a></p><img src="../img/p.png"/></body></html>`), "OPS/a.xhtml": []byte(`<html><body><h2 id="end">Second</h2><p>SECOND TEXT</p></body></html>`), "img/p.png": picture()}
	b, e := Parse(archiveFixture(files), "application/epub+zip", "", nil)
	if e != nil {
		t.Fatal(e)
	}
	if strings.Index(b.Text, "FIRST TEXT") > strings.Index(b.Text, "SECOND TEXT") {
		t.Fatal("spine ignored")
	}
	if len(b.Assets) != 1 || len(b.Assets[0].Data) == 0 {
		t.Fatal("EPUB picture missing")
	}
	v, e := b.HTML(identity(b), false)
	if e != nil || !strings.Contains(string(v), `href="#d2-end"`) {
		t.Fatal("cross-document anchor missing", string(v), e)
	}
}
func TestSourceLimitsAndImageFailureVisible(t *testing.T) {
	if _, e := Parse(make([]byte, MaxBytes+1), "text/plain", "", nil); e == nil {
		t.Fatal("oversize accepted")
	}
	if _, e := Parse([]byte{0, 1, 2}, "application/octet-stream", "", nil); e == nil {
		t.Fatal("binary text accepted")
	}
	if _, e := Parse(archiveFixture(map[string][]byte{"../x.html": []byte("bad")}), "application/zip", "", nil); e == nil {
		t.Fatal("traversal accepted")
	}
	b, e := Parse([]byte(`<p>Text</p><img src="https://example.com/a.png" alt="Missing">`), "text/html", "https://example.com/", func(string) ([]byte, string, error) { return nil, "", fmt.Errorf("offline") })
	if e != nil {
		t.Fatal(e)
	}
	out, e := b.HTML(identity(b), false)
	if e != nil || len(b.Warnings) == 0 || !strings.Contains(string(out), "삽화 미확보") {
		t.Fatal("missing picture silently removed")
	}
}

func TestInlineSpacingSurvivesTranslation(t *testing.T) {
	b, e := Parse([]byte(`<p>Alice <em>very</em> tired.</p>`), "text/html", "https://example.com", nil)
	if e != nil {
		t.Fatal(e)
	}
	translated := map[string]string{b.Blocks[0].ID: "앨리스는", b.Blocks[1].ID: "매우", b.Blocks[2].ID: "피곤했다."}
	plain, e := b.Plain(translated)
	if e != nil || string(plain) != "앨리스는 매우 피곤했다.\n" {
		t.Fatalf("inline words joined: %q %v", plain, e)
	}
	html, e := b.HTML(translated, false)
	if e != nil || !strings.Contains(string(html), "앨리스는 <em>매우</em> 피곤했다.") {
		t.Fatal("HTML spacing lost", e)
	}
}

func TestMainPreviewRetainsWholeDocument(t *testing.T) {
	b, e := Parse([]byte(`<nav>Navigation before article</nav><main><h1>Article title</h1><p>Useful article text</p></main><footer>Reference after article</footer>`), "text/html", "https://example.com", nil)
	if e != nil {
		t.Fatal(e)
	}
	start, ok := b.Metadata["main_start"].(int)
	if !ok || start <= 0 {
		t.Fatal("main preview position missing")
	}
	page, e := b.Page(start, 12000)
	if e != nil || !strings.Contains(page["content"].(string), "Useful article text") {
		t.Fatal("main content inaccessible", e)
	}
	if !strings.Contains(b.Text, "Navigation before") || !strings.Contains(b.Text, "Reference after") {
		t.Fatal("preview filtering destroyed the complete source")
	}
}

func TestHTMLZIPFollowsContentsLinks(t *testing.T) {
	files := map[string][]byte{"index.html": []byte(`<h1>Contents</h1><a href="z.html">First</a><a href="a.html">Second</a>`), "z.html": []byte(`<p>FIRST CHAPTER</p>`), "a.html": []byte(`<p>SECOND CHAPTER</p>`)}
	b, e := Parse(archiveFixture(files), "application/zip", "", nil)
	if e != nil {
		t.Fatal(e)
	}
	if strings.Index(b.Text, "FIRST CHAPTER") > strings.Index(b.Text, "SECOND CHAPTER") {
		t.Fatal("HTML contents order ignored")
	}
	if len(b.Warnings) != 0 {
		t.Fatal("resolved order marked uncertain", b.Warnings)
	}
}

func TestSelectedChapterKeepsItsIllustrationAndStructure(t *testing.T) {
	html := `<h2 id="one">Chapter I</h2><p>First chapter.</p><img src="one.png" alt="First"><h2 id="two">Chapter II</h2><p>Second chapter.</p><figure><img src="two.png" alt="Second"><figcaption>Second caption</figcaption></figure>`
	b, e := Parse([]byte(html), "text/html", "https://example.com/", func(string) ([]byte, string, error) { return picture(), "image/png", nil })
	if e != nil {
		t.Fatal(e)
	}
	start := 0
	values := map[string]string{}
	for _, x := range b.Blocks {
		if x.Text == "Chapter II" {
			start = x.Start
		}
		if start > 0 && x.Start >= start {
			values[x.ID] = "번역 " + x.Text
		}
	}
	selected := b.Select(start, 0, values)
	if len(selected.Assets) != 1 {
		t.Fatal("wrong chapter images selected", len(selected.Assets))
	}
	out, e := selected.HTML(values, false)
	if e != nil {
		t.Fatal(e)
	}
	text := string(out)
	if !strings.Contains(text, `id="two"`) || !strings.Contains(text, "<figure>") || !strings.Contains(text, "Second caption") || strings.Contains(text, "First chapter") {
		t.Fatal("selected chapter structure lost", text)
	}
}
