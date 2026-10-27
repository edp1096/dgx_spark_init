package sourcebook

import (
	"fmt"
	"strings"
	"testing"
)

func epubFixture(manifest, spine string) map[string][]byte {
	return map[string][]byte{
		"mimetype":               []byte("application/epub+zip"),
		"META-INF/container.xml": []byte(`<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0"><rootfiles><rootfile full-path="OPS/book.opf" media-type="application/oebps-package+xml"/></rootfiles></container>`),
		"OPS/book.opf":           []byte(`<package xmlns="http://www.idpf.org/2007/opf" version="3.0"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Reading order</dc:title><dc:creator>First author</dc:creator><dc:creator>Second author</dc:creator><dc:language>en</dc:language></metadata><manifest>` + manifest + `</manifest><spine>` + spine + `</spine></package>`),
	}
}

func TestEPUBVersionsMetadataAndNonlinearNotes(t *testing.T) {
	for _, version := range []string{"2.0", "3.0"} {
		t.Run(version, func(t *testing.T) {
			files := epubFixture(`<item id="first" href="z.xhtml" media-type="application/xhtml+xml"/><item id="notes" href="a.xhtml" media-type="application/xhtml+xml"/><item id="unused" href="unused.xhtml" media-type="application/xhtml+xml"/>`, `<itemref idref="first"/><itemref idref="notes" linear="no"/>`)
			files["OPS/book.opf"] = []byte(strings.Replace(string(files["OPS/book.opf"]), `version="3.0"`, `version="`+version+`"`, 1))
			files["OPS/z.xhtml"] = []byte(`<!DOCTYPE html><html xmlns="http://www.w3.org/1999/xhtml"><head><title>Wrong title</title><style>STYLE_SENTINEL</style></head><body><h1>Opening</h1><p>First&nbsp;chapter <em>with emphasis</em>.</p><a href="a.xhtml#note">Note</a><script>SCRIPT_SENTINEL</script></body></html>`)
			files["OPS/a.xhtml"] = []byte(`<html><body><p id="note">Last footnote</p><a href="z.xhtml">Back</a></body></html>`)
			files["OPS/unused.xhtml"] = []byte(`<p>UNUSED_SENTINEL</p>`)
			book, err := Parse(archiveFixture(files), "application/epub+zip", "", nil)
			if err != nil {
				t.Fatal(err)
			}
			if book.Title != "Reading order" || book.Metadata["author"] != "First author; Second author" || book.Metadata["language"] != "en" || book.Metadata["epub_version"] != version {
				t.Fatalf("metadata lost: %+v", book.Metadata)
			}
			if !strings.Contains(book.Text, "First\u00a0chapter") || !strings.Contains(book.Text, "Last footnote") || strings.Index(book.Text, "First") > strings.Index(book.Text, "Last") || strings.Contains(book.Text, "SENTINEL") {
				t.Fatalf("wrong source content/order: %q", book.Text)
			}
			out, err := book.HTML(identity(book), false)
			if err != nil || !strings.Contains(string(out), `href="#d2-note"`) || !strings.Contains(string(out), `href="#d1-"`) {
				t.Fatalf("cross-document notes broken: %s %v", out, err)
			}
		})
	}
}

func TestEPUBFallbackSVGAndEscapedResourcePaths(t *testing.T) {
	files := epubFixture(`<item id="first" href="start.xhtml" media-type="application/xhtml+xml"/><item id="foreign" href="page.bin" media-type="application/x-custom" fallback="svg"/><item id="svg" href="drawing%20%2520.svg" media-type="image/svg+xml"/>`, `<itemref idref="first"/><itemref idref="foreign"/>`)
	files["OPS/start.xhtml"] = []byte(`<p>Introduction <a href="page.bin#label">diagram</a></p><img src="art/cover.svg" alt="Cover description"/>`)
	files["OPS/drawing %20.svg"] = []byte(`<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"><title>Diagram title</title><defs><text>HIDDEN_SENTINEL</text></defs><script>SCRIPT_SENTINEL</script><text id="label">Hello <tspan>world</tspan></text><image xlink:href="art/pic%20%2520.png"/><a xlink:href="start.xhtml">Return</a></svg>`)
	files["OPS/art/cover.svg"] = []byte(`<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"><image xlink:href="pic%20%2520.png"/><text>Cover text</text></svg>`)
	files["OPS/art/pic %20.png"] = picture()
	book, err := Parse(archiveFixture(files), "application/epub+zip", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"Introduction", "Cover text", "Cover description", "Diagram title", "Hello", "world"} {
		if !strings.Contains(book.Text, want) {
			t.Fatalf("SVG text lost: %s in %q", want, book.Text)
		}
	}
	if strings.Contains(book.Text, "SENTINEL") || len(book.Assets) != 1 || len(book.Assets[0].Data) == 0 || book.Incomplete {
		t.Fatalf("SVG extraction failed: %+v, warnings %v", book.Assets, book.Warnings)
	}
	translated := identity(book)
	for _, block := range book.Blocks {
		if block.Text == "world" {
			translated[block.ID] = "세계"
		}
	}
	out, err := book.HTML(translated, true)
	for _, want := range []string{"세계", `href="#d2-label"`, `href="#d1-"`, "data:image/png;base64,"} {
		if err != nil || !strings.Contains(string(out), want) {
			t.Fatalf("translation export lost %s: %s %v", want, out, err)
		}
	}
	if strings.Contains(string(out), "<svg") || strings.Contains(string(out), "<script") || len(book.Warnings) == 0 {
		t.Fatal("SVG should become static translated content with a layout warning")
	}
}

func TestEPUBRejectsUnreadableSpineWithoutFallingBackToFileOrder(t *testing.T) {
	for _, tc := range []struct{ name, manifest, spine, want string }{
		{"missing", "", `<itemref idref="missing"/>`, "missing manifest item"},
		{"unsupported", `<item id="a" href="a.bin" media-type="application/octet-stream"/>`, `<itemref idref="a"/>`, "no supported"},
		{"cycle", `<item id="a" href="a.bin" media-type="a/b" fallback="b"/><item id="b" href="b.bin" media-type="a/b" fallback="a"/>`, `<itemref idref="a"/>`, "cycle"},
		{"escape", `<item id="a" href="../../a.xhtml" media-type="application/xhtml+xml"/>`, `<itemref idref="a"/>`, "escapes container"},
		{"external", `<item id="a" href="https://example.com/a.xhtml" media-type="application/xhtml+xml"/>`, `<itemref idref="a"/>`, "local archive reference"},
		{"empty", "", "", "no readable spine"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			files := epubFixture(tc.manifest, tc.spine)
			files["a.xhtml"] = []byte(`<p>This unused file must not hide an error.</p>`)
			if _, err := Parse(archiveFixture(files), "application/epub+zip", "", nil); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("expected %q: %v", tc.want, err)
			}
		})
	}
	if _, err := Parse(archiveFixture(map[string][]byte{"index.html": []byte(`<p>Text</p>`)}), "application/epub+zip", "", nil); err == nil || !strings.Contains(err.Error(), "container") {
		t.Fatalf("EPUB without a container accepted: %v", err)
	}
}

func TestEPUBEncryptedBodyAndObfuscatedFonts(t *testing.T) {
	for _, encrypted := range []string{"OPS/body.xhtml", "OPS/font.otf"} {
		t.Run(encrypted, func(t *testing.T) {
			files := epubFixture(`<item id="a" href="body.xhtml" media-type="application/xhtml+xml"/><item id="font" href="font.otf" media-type="font/otf"/>`, `<itemref idref="a"/>`)
			files["OPS/body.xhtml"] = []byte(`<p>Readable text</p>`)
			files["OPS/font.otf"] = []byte("obfuscated font")
			files["META-INF/encryption.xml"] = []byte(fmt.Sprintf(`<encryption xmlns="urn:oasis:names:tc:opendocument:xmlns:container" xmlns:enc="http://www.w3.org/2001/04/xmlenc#"><enc:EncryptedData><enc:EncryptionMethod Algorithm="http://www.idpf.org/2008/embedding"/><enc:CipherData><enc:CipherReference URI="%s"/></enc:CipherData></enc:EncryptedData></encryption>`, encrypted))
			book, err := Parse(archiveFixture(files), "application/epub+zip", "", nil)
			if strings.HasSuffix(encrypted, ".xhtml") {
				if err == nil || !strings.Contains(err.Error(), "encrypted EPUB resource") {
					t.Fatalf("encrypted body read as text: %v", err)
				}
			} else if err != nil || !strings.Contains(book.Text, "Readable text") {
				t.Fatalf("obfuscated font prevented reading unencrypted body: %v", err)
			}
		})
	}
}

func TestEPUBRecursiveSVGIsReported(t *testing.T) {
	files := epubFixture(`<item id="a" href="body.xhtml" media-type="application/xhtml+xml"/>`, `<itemref idref="a"/>`)
	files["OPS/body.xhtml"] = []byte(`<p>Readable text</p><img src="cycle.svg"/>`)
	files["OPS/cycle.svg"] = []byte(`<svg xmlns="http://www.w3.org/2000/svg"><image href="cycle.svg"/></svg>`)
	book, err := Parse(archiveFixture(files), "application/epub+zip", "", nil)
	if err != nil || !book.Incomplete || !strings.Contains(strings.Join(book.Warnings, " "), "recursive SVG") {
		t.Fatalf("recursive SVG not bounded/reported: %+v %v", book, err)
	}
}

func TestEPUBBodyAnchorsInlineSVGAndObjectText(t *testing.T) {
	files := epubFixture(`<item id="a" href="body.xhtml" media-type="application/xhtml+xml"/>`, `<itemref idref="a"/>`)
	files["OPS/body.xhtml"] = []byte(`<html><head><meta name="author" content="Wrong author"/></head><body id="page"><a href="#%EB%A9%94%EB%AA%A8">Note</a><p id="메모">Korean note</p><a href="body.xhtml#page">Top</a><svg xmlns="http://www.w3.org/2000/svg"><text>Inline text</text><foreignObject><p xmlns="http://www.w3.org/1999/xhtml">HTML inside SVG <img src="pic.png" alt="Picture description"/></p></foreignObject></svg><object data="vector" type="image/svg+xml" id="object"></object><object data="unknown.bin">Object fallback prose</object></body></html>`)
	files["OPS/vector"] = []byte(`<svg xmlns="http://www.w3.org/2000/svg"><text>Object SVG text</text></svg>`)
	files["OPS/pic.png"] = picture()
	book, err := Parse(archiveFixture(files), "application/epub+zip", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"Inline text", "HTML inside SVG", "Picture description", "Object SVG text", "Object fallback prose"} {
		if !strings.Contains(book.Text, want) {
			t.Fatalf("missing %s in %q", want, book.Text)
		}
	}
	if book.Metadata["author"] != "First author; Second author" {
		t.Fatal("chapter metadata replaced the book author")
	}
	out, err := book.HTML(identity(book), true)
	for _, want := range []string{`href="#d1-메모"`, `id="d1-메모"`, `href="#d1-page"`, `id="d1-page"`, `id="d1-object"`, "data:image/png;base64,"} {
		if err != nil || !strings.Contains(string(out), want) {
			t.Fatalf("missing %s in %s: %v", want, out, err)
		}
	}
}
