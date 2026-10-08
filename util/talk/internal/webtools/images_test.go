package webtools

import (
	"context"
	"io"
	"net/http"
	"net/url"
	"strings"
	"testing"
)

func TestImageExtraction(t *testing.T) {
	base, _ := url.Parse("https://example.org/wiki/page")
	images := extractImages(base, `<script>"https://evil.example/fake.jpg"</script><p>https://evil.example/text.png</p><img alt='사진' src='../photo.jpg?a=1&amp;b=2'><img src='../photo.jpg?a=1&amp;b=2'><meta content='//cdn.example.org/original.png' property='og:image'><a href='/full.webp'>Original file</a><img src='data:image/png;base64,abc'>`)
	if len(images) != 3 {
		t.Fatalf("unexpected candidates %+v", images)
	}
	if images[0].URL != "https://example.org/photo.jpg?a=1&b=2" || images[0].SourceURL != base.String() || images[0].Alt != "사진" {
		t.Fatalf("lost source %+v", images[0])
	}
}

func TestImageDownloadBlocksPrivateURL(t *testing.T) {
	for _, u := range []string{"http://127.0.0.1/image.png", "http://[::1]/image.png", "http://169.254.169.254/latest", "file:///tmp/image.png"} {
		if _, _, err := New(1, 0).DownloadImage(context.Background(), u, 1024); err == nil {
			t.Fatalf("accepted %s", u)
		}
	}
}

func TestImageDownloadChecksResponseAndRedirect(t *testing.T) {
	runner := New(1, 0)
	private, _ := url.Parse("http://127.0.0.1/secret")
	if err := runner.client.CheckRedirect(&http.Request{URL: private}, nil); err == nil {
		t.Fatal("private redirect accepted")
	}
	for _, tc := range []struct {
		name, body string
		limit      int64
		want       bool
	}{
		{"html", "<html>not an image</html>", 1024, false},
		{"oversize", "\x89PNG\r\n\x1a\n" + strings.Repeat("x", 100), 32, false},
		{"image signature", "\x89PNG\r\n\x1a\n" + strings.Repeat("x", 100), 1024, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			runner.client.Transport = imageRoundTrip(func(req *http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(tc.body)), ContentLength: -1, Request: req, Header: http.Header{"Content-Type": []string{"image/png"}}}, nil
			})
			_, _, err := runner.DownloadImage(context.Background(), "https://example.com/photo.png", tc.limit)
			if (err == nil) != tc.want {
				t.Fatalf("unexpected error %v", err)
			}
		})
	}
}

type imageRoundTrip func(*http.Request) (*http.Response, error)

func (f imageRoundTrip) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
