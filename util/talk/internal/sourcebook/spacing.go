package sourcebook

import (
	"strings"
	"unicode"
)

// PreserveBoundary keeps structural whitespace between independently translated
// inline nodes/paragraph fragments. It never rewrites interior translated text.
func PreserveBoundary(original, translated string) string {
	lead := original[:len(original)-len(strings.TrimLeftFunc(original, unicode.IsSpace))]
	tail := original[len(strings.TrimRightFunc(original, unicode.IsSpace)):]
	if lead != "" && strings.TrimLeftFunc(translated, unicode.IsSpace) == translated {
		translated = lead + translated
	}
	if tail != "" && strings.TrimRightFunc(translated, unicode.IsSpace) == translated {
		translated += tail
	}
	return translated
}
