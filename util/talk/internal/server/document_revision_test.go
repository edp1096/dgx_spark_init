package server

import (
	"strings"
	"testing"

	"sparktalk/internal/sourcebook"
)

func TestDocumentRevisionRetainsUnaffectedTextAcrossGlyphParserUpgrade(t *testing.T) {
	raw := `<html><body><p>First paragraph.</p><p>Book: <img class="gaiji" src="1-87-25.png" alt="encoded letter">東綺譚。</p><p>Last paragraph.</p></body></html>`
	// Emulate the old parser's treatment of the very same image as a normal
	// illustration. The old decoder did not recognize class=gaiji.
	oldBook, err := sourcebook.Parse([]byte(strings.Replace(raw, `class="gaiji"`, `class="image"`, 1)), "text/html", "https://example.com/book", nil)
	if err != nil {
		t.Fatal(err)
	}
	newBook, err := sourcebook.Parse([]byte(raw), "text/html", "https://example.com/book", nil)
	if err != nil {
		t.Fatal(err)
	}
	oldBook.Version = 4
	oldBook.Hash = newBook.Hash
	prior := planDocumentTranslation("old", oldBook, 0, 0)
	for i := range prior {
		prior[i].Output = map[string]string{}
		for _, in := range prior[i].Inputs {
			prior[i].Output[in.ID] = "SAVED:" + in.Text
		}
		prior[i].Status = "completed"
	}
	planned := planDocumentTranslation("new", newBook, 0, 0)
	child := carryDocumentTranslation("old", "new", oldBook, newBook, prior, planned)
	pending, kept := 0, 0
	for _, unit := range child {
		for _, in := range unit.Inputs {
			if strings.Contains(in.Text, "First") || strings.Contains(in.Text, "Last") {
				if unit.Output[in.ID] != "SAVED:"+in.Text || unit.Status != "completed" {
					t.Fatal("unaffected paragraph regenerated", in, unit)
				}
				kept++
			} else {
				if unit.Status != "pending" {
					t.Fatal("changed paragraph declared reusable")
				}
				pending++
			}
		}
	}
	if kept != 2 || pending == 0 {
		t.Fatal("revision coverage", kept, pending)
	}
	for _, unit := range prior {
		for _, in := range unit.Inputs {
			if unit.Output[in.ID] != "SAVED:"+in.Text {
				t.Fatal("parent mutated")
			}
		}
	}
}
