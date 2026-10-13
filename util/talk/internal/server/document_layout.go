package server

import (
	"fmt"
	"strings"

	"sparktalk/internal/sourcebook"
	"sparktalk/internal/workflows"
)

// Keep saved paragraph identity while requiring visible text inside each
// protected span and on either side of images or empty page anchors.
func validateDocumentCoverage(u workflows.DocumentUnit, output map[string]string) error {
	for i, in := range u.Inputs {
		for _, r := range output[in.ID] {
			if r < 0x20 && r != '\n' && r != '\r' && r != '\t' || r == 0xfffe || r == 0xffff {
				return fmt.Errorf("invalid document control character U+%04X at output ID %q", r, documentLocalID(i))
			}
		}
	}
	if err := sourcebook.ValidateTranslation(documentTranslationSlots(u), output); err != nil {
		return err
	}
	text := map[string]string{}
	first := map[string]int{}
	order := []string{}
	for i, in := range u.Inputs {
		if in.Annotation || in.Scope == "" {
			continue
		}
		if _, ok := first[in.Scope]; !ok {
			first[in.Scope] = i
			order = append(order, in.Scope)
		}
		text[in.Scope] += output[in.ID]
	}
	for _, scope := range order {
		if strings.TrimSpace(text[scope]) == "" {
			return fmt.Errorf("protected text missing at output ID %q; keep its meaning inside its layout scope", documentLocalID(first[scope]))
		}
	}
	return nil
}
