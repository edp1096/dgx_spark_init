package webtools

import (
	"context"
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"
)

type sourceTransport func(*http.Request) (*http.Response, error)

func (f sourceTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
func TestResourceRejectsSilentTruncation(t *testing.T) {
	r := New(1, 0)
	r.client = &http.Client{Transport: sourceTransport(func(req *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": []string{"text/plain"}}, Body: io.NopCloser(strings.NewReader("123456")), Request: req}, nil
	})}
	// Public literal address avoids dependence on DNS/network for the fixture.
	if _, _, _, e := r.Resource(context.Background(), "https://8.8.8.8/test", 5); e == nil {
		t.Fatal("truncated response accepted")
	}
	data, _, _, e := r.Resource(context.Background(), "https://8.8.8.8/test", 6)
	if e != nil || string(data) != "123456" {
		t.Fatal("exact-sized source lost", e)
	}
	if _, _, _, e = r.Resource(context.Background(), "http://127.0.0.1/private", 6); e == nil {
		t.Fatal("private fetch allowed")
	}
	if e = validatePublicURL(context.Background(), &url.URL{Scheme: "file", Path: "/etc/passwd"}); e == nil {
		t.Fatal("file fetch allowed")
	}
}
