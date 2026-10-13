package sourcebook

import (
	"strings"
	"testing"
)

func TestIDOnlyAnchorDoesNotAcquireSourceURL(t *testing.T) {
	b, err := Parse([]byte(`<html><body><p>Before<a id="page2"><span> </span></a>After</p><p><a href="#page2">Page 2</a></p></body></html>`), "text/html", "https://example.org/book", nil)
	if err != nil {
		t.Fatal(err)
	}
	output := map[string]string{}
	for _, block := range b.Blocks {
		output[block.ID] = block.Text
	}
	page, err := b.HTML(output, true)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(page), `href="https://example.org/book"`) || !strings.Contains(string(page), `<a id="page2">`) || !strings.Contains(string(page), `href="#page2"`) {
		t.Fatal("anchor semantics changed", string(page))
	}
}
