package knowledge

import (
	"archive/zip"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestZIPAttachmentTextAndListing(t *testing.T) {
	p := filepath.Join(t.TempDir(), "source.zip")
	f, e := os.Create(p)
	if e != nil {
		t.Fatal(e)
	}
	w := zip.NewWriter(f)
	for name, content := range map[string]string{"src/main.py": "print('hello zip')", "../outside.txt": "reference only", "image.png": "\x00binary", "large.txt": strings.Repeat("x", 9<<20)} {
		z, e := w.Create(name)
		if e != nil {
			t.Fatal(e)
		}
		if _, e = z.Write([]byte(content)); e != nil {
			t.Fatal(e)
		}
	}
	if e = w.Close(); e != nil {
		t.Fatal(e)
	}
	f.Close()
	pages, e := (&Extractor{}).Extract(p, "application/zip")
	if e != nil {
		t.Fatal(e)
	}
	text := pages[0].Text
	for _, want := range []string{"src/main.py", "hello zip", "../outside.txt", "Binary contents not read", "text extraction budget exceeded"} {
		if !strings.Contains(text, want) {
			t.Fatalf("missing %q", want)
		}
	}
	if _, e = os.Stat(filepath.Join(filepath.Dir(p), "src")); !os.IsNotExist(e) {
		t.Fatal("archive was extracted to disk")
	}
}
