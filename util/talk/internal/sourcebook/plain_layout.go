package sourcebook

import (
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

var plainListLine = regexp.MustCompile(`^(?:[-*+•]|[0-9]+[.)]|[A-Z][.)])\s+`)

// A physical newline in TXT has no markup explaining its intent. Fold only
// convincing fixed-width prose; short/indented/aligned/ambiguous lines remain
// authored layout. Source bytes, block text and offsets are never rewritten.
func plainTextWhitespace(text, kind string) string {
	if kind != "p" {
		return "pre-wrap"
	}
	lines := strings.Split(text, "\n")
	if len(lines) == 1 {
		return "normal"
	}
	if utf8.RuneCountInString(text) < 100 {
		return "pre-wrap"
	}
	width, narrow, links := 0, 0, 0
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || line != strings.TrimLeftFunc(line, unicode.IsSpace) || strings.ContainsAny(line, "\t|{}") || plainListLine.MatchString(trimmed) {
			return "pre-wrap"
		}
		// Multiple internal columns and decorative separators aren't prose.
		if strings.Contains(trimmed, "  ") {
			return "pre-wrap"
		}
		letters := 0
		for _, r := range line {
			if unicode.IsLetter(r) {
				letters++
			}
		}
		n := utf8.RuneCountInString(line)
		if letters*2 < n {
			return "pre-wrap"
		}
		if i < len(lines)-1 {
			if n < 45 || n > 120 {
				return "pre-wrap"
			}
			width = max(width, n)
		}
		if i > 0 {
			prev := strings.TrimSpace(lines[i-1])
			first, _ := utf8.DecodeRuneInString(trimmed)
			last, _ := utf8.DecodeLastRuneInString(prev)
			if !strings.ContainsRune(".!?。！？:;：；\"”’」』", last) && (unicode.IsLower(first) || unicode.Is(unicode.Hangul, first) || unicode.Is(unicode.Han, first) || unicode.Is(unicode.Hiragana, first) || unicode.Is(unicode.Katakana, first)) {
				links++
			}
		}
	}
	for _, line := range lines[:len(lines)-1] {
		if width-utf8.RuneCountInString(line) > 16 {
			narrow++
		}
	}
	if narrow > 0 || links == 0 || len(lines) > 3 && links*2 < len(lines)-1 {
		return "pre-wrap"
	}
	return "normal"
}
