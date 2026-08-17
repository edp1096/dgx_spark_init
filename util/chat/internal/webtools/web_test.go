package webtools

import (
	"context"
	"net/url"
	"testing"
)

func TestParseDuckDuckGoResults(t *testing.T) {
	page := `<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdoc">Example &amp; Doc</a><a class="result__snippet">Useful <b>snippet</b>.</a></div>`
	results := parseDuckDuckGo(page, 5)
	if len(results) != 1 {
		t.Fatalf("expected one result, got %d", len(results))
	}
	if results[0].URL != "https://example.com/doc" || results[0].Title != "Example & Doc" || results[0].Snippet != "Useful snippet." {
		t.Fatalf("unexpected result: %+v", results[0])
	}
}

func TestValidatePublicURLBlocksPrivateNetworks(t *testing.T) {
	for _, raw := range []string{"http://127.0.0.1/", "http://10.0.0.1/", "file:///etc/passwd"} {
		u, _ := url.Parse(raw)
		if err := validatePublicURL(context.Background(), u); err == nil {
			t.Fatalf("expected URL to be blocked: %s", raw)
		}
	}
}
