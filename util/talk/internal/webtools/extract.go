package webtools

import (
	"html"
	"net/url"
	"regexp"
	"strings"
)

var (
	tagPattern         = regexp.MustCompile(`<[^>]*>`)
	spacePattern       = regexp.MustCompile(`\s+`)
	punctuationPattern = regexp.MustCompile(`\s+([.,!?;:])`)
	naverFramePattern  = regexp.MustCompile(`(?is)<iframe\b[^>]*\bid\s*=\s*["']mainFrame["'][^>]*>`)
	srcPattern         = regexp.MustCompile(`(?is)\bsrc\s*=\s*["']([^"']+)["']`)
	naverMainPattern   = regexp.MustCompile(`(?is)<div\b[^>]*\bclass\s*=\s*["'][^"']*\bse-main-container\b[^"']*["'][^>]*>`)
	naverTitlePattern  = regexp.MustCompile(`(?is)<meta\b[^>]*\bproperty\s*=\s*["']og:title["'][^>]*\bcontent\s*=\s*["']([^"']*)["'][^>]*>`)
	articlePattern     = regexp.MustCompile(`(?is)<article\b[^>]*>(.*?)</article\s*>`)
	mainPattern        = regexp.MustCompile(`(?is)<main\b[^>]*>(.*?)</main\s*>`)
)

func naverFrameURL(base *url.URL, page string) *url.URL {
	frame := naverFramePattern.FindString(page)
	match := srcPattern.FindStringSubmatch(frame)
	if len(match) != 2 {
		return nil
	}
	reference, err := url.Parse(html.UnescapeString(match[1]))
	if err != nil {
		return nil
	}
	return base.ResolveReference(reference)
}

func extractNaverArticle(page string) string {
	location := naverMainPattern.FindStringIndex(page)
	if location == nil {
		return ""
	}
	article := page[location[0]:]
	if end := strings.Index(article, `id="post_footer_contents"`); end >= 0 {
		if start := strings.LastIndex(article[:end], "<div"); start >= 0 {
			article = article[:start]
		}
	}
	content := cleanHTML(removeNonContent(article))
	if title := naverTitlePattern.FindStringSubmatch(page); len(title) == 2 {
		titleText := strings.TrimSpace(html.UnescapeString(title[1]))
		if titleText != "" {
			content = titleText + "\n\n" + content
		}
	}
	return strings.TrimSpace(content)
}

func extractReadableHTML(page string) string {
	for _, pattern := range []*regexp.Regexp{articlePattern, mainPattern} {
		matches := pattern.FindAllStringSubmatch(page, -1)
		best := ""
		for _, match := range matches {
			if len(match) != 2 {
				continue
			}
			candidate := cleanHTML(removeNonContent(match[1]))
			if len(candidate) > len(best) {
				best = candidate
			}
		}
		if len(best) >= 200 {
			return best
		}
	}
	return cleanHTML(removeNonContent(page))
}

func attribute(part, name string) string {
	needle := name + `="`
	start := strings.Index(part, needle)
	if start < 0 {
		return ""
	}
	start += len(needle)
	end := strings.Index(part[start:], `"`)
	if end < 0 {
		return ""
	}
	return part[start : start+end]
}

func between(value, startToken, endToken string) string {
	start := strings.Index(value, startToken)
	if start < 0 {
		return ""
	}
	start += len(startToken)
	end := strings.Index(value[start:], endToken)
	if end < 0 {
		return ""
	}
	return value[start : start+end]
}

func cleanHTML(value string) string {
	value = tagPattern.ReplaceAllString(value, " ")
	value = html.UnescapeString(value)
	value = strings.TrimSpace(spacePattern.ReplaceAllString(value, " "))
	return punctuationPattern.ReplaceAllString(value, "$1")
}

func removeNonContent(value string) string {
	lower := strings.ToLower(value)
	for _, tag := range []string{"script", "style", "nav", "footer", "aside", "iframe", "noscript"} {
		for {
			start := strings.Index(lower, "<"+tag)
			if start < 0 {
				break
			}
			end := strings.Index(lower[start:], "</"+tag+">")
			if end < 0 {
				value = value[:start]
				lower = lower[:start]
				break
			}
			end += start + len(tag) + 3
			value = value[:start] + " " + value[end:]
			lower = strings.ToLower(value)
		}
	}
	return value
}
