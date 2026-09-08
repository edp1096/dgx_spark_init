package webtools

import (
	"context"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"
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

func TestNaverFrameURL(t *testing.T) {
	base, _ := url.Parse("https://blog.naver.com/user/123?tracking=x")
	page := `<iframe id="mainFrame" src="/PostView.naver?blogId=user&amp;logNo=123"></iframe>`
	got := naverFrameURL(base, page)
	if got == nil || got.String() != "https://blog.naver.com/PostView.naver?blogId=user&logNo=123" {
		t.Fatalf("unexpected Naver frame URL: %v", got)
	}
}

func TestExtractNaverArticle(t *testing.T) {
	page := `<html><head><meta property="og:title" content="글 제목" /></head><body>
		<div class="se-main-container"><p>첫 문단</p><script>ignore()</script><p>둘째 문단</p></div>
		<div id="post_footer_contents">댓글과 메뉴</div></body></html>`
	got := extractNaverArticle(page)
	if !strings.Contains(got, "글 제목") || !strings.Contains(got, "첫 문단") || !strings.Contains(got, "둘째 문단") {
		t.Fatalf("Naver article content was not extracted: %q", got)
	}
	if strings.Contains(got, "ignore") || strings.Contains(got, "댓글과 메뉴") {
		t.Fatalf("Naver non-content leaked into article: %q", got)
	}
}

func TestExtractReadableHTMLPrefersMainContent(t *testing.T) {
	page := `<html><body><header>` + strings.Repeat("메뉴 ", 200) + `</header>
		<main><h1>기사 제목</h1><p>` + strings.Repeat("본문 내용 ", 80) + `</p></main>
		<footer>` + strings.Repeat("회사 정보 ", 200) + `</footer></body></html>`
	got := extractReadableHTML(page)
	if !strings.Contains(got, "기사 제목") || !strings.Contains(got, "본문 내용") {
		t.Fatalf("main content was not extracted: %.300s", got)
	}
	if strings.Contains(got, "메뉴") || strings.Contains(got, "회사 정보") {
		t.Fatalf("page chrome leaked into main content: %.300s", got)
	}
}

func TestFetchNaverBlogIntegration(t *testing.T) {
	if os.Getenv("SPARKTALK_NETWORK_TEST") == "" {
		t.Skip("set SPARKTALK_NETWORK_TEST=1 to run live web fetch tests")
	}
	runner := New(5, 20*time.Second)
	result, err := runner.Execute(context.Background(), "web_fetch", `{"url":"https://blog.naver.com/kjsw444/224379068596"}`)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result, "30대 중반 남자") || !strings.Contains(result, "요즘 소개팅 시장") {
		t.Fatalf("Naver post body was not fetched: %.500s", result)
	}
}

func TestFetchLargeArticleIntegration(t *testing.T) {
	if os.Getenv("SPARKTALK_NETWORK_TEST") == "" {
		t.Skip("set SPARKTALK_NETWORK_TEST=1 to run live web fetch tests")
	}
	runner := New(5, 20*time.Second)
	result, err := runner.Execute(context.Background(), "web_fetch", `{"url":"https://www.ksolves.com/blog/golang/trends-shaping-the-next-generation"}`)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(result, "As we move into 2026") || !strings.Contains(result, "Golang Trends in 2026") || !strings.Contains(result, "Conclusion") {
		t.Fatalf("large article body was not fetched: %.500s", result)
	}
}
