package server

import (
	"regexp"
	"strings"
)

var historicalEvidenceLine = regexp.MustCompile(`^\[Historical tool evidence: [\w.-]+; archive_id=\d+; use context_read for original\]`)

// Strip only app-generated reference records outside literal code examples.
func cleanInternalEvidence(text string) string {
	lines := strings.Split(text, "\n")
	out := make([]string, 0, len(lines))
	fence := ""
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "```") || strings.HasPrefix(trimmed, "~~~") {
			marker := trimmed[:3]
			if fence == "" {
				fence = marker
			} else if marker == fence {
				fence = ""
			}
			out = append(out, line)
			continue
		}
		if fence == "" && historicalEvidenceLine.MatchString(trimmed) {
			continue
		}
		out = append(out, line)
	}
	return strings.TrimSpace(strings.Join(out, "\n"))
}
