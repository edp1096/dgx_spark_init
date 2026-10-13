package server

import (
	"sort"
	"strings"

	"sparktalk/internal/workflows"
)

// A decoded response can be incomplete while containing whole usable paragraphs.
// Ambiguous/malformed JSON never reaches this path: the decoder returns nil.
func incompleteDocumentGroups(u workflows.DocumentUnit) map[string]string {
	bad := shortDocumentOutputs(u)
	for _, group := range documentTextGroups(u) {
		part := workflows.DocumentUnit{Inputs: group.Inputs, Output: u.Output}
		if err := validateDocumentCoverage(part, part.Output); err != nil {
			for _, in := range group.Inputs {
				bad[in.ID] = err.Error()
			}
		}
	}
	return bad
}

// Commit accepted paragraphs and pending repairs together. One inference attempt
// owns usage/performance; other checkpoints reference that same attempt.
func (s *Server) recoverDocumentParagraphs(run *workflows.Run, units *[]workflows.DocumentUnit, indices []int, output map[string]string, attempt workflows.DocumentAttempt) (bool, error) {
	if run.Document.Kind != "translate" || len(output) == 0 {
		return false, nil
	}
	changed := make([]workflows.DocumentUnit, 0, len(indices))
	repairs := []workflows.DocumentUnit{}
	accepted := false
	for _, index := range indices {
		item := (*units)[index]
		item.Attempts = append([]workflows.DocumentAttempt(nil), item.Attempts...)
		item.Warnings = append([]string(nil), item.Warnings...)
		item.Output = map[string]string{}
		for _, in := range item.Inputs {
			if text, ok := output[in.ID]; ok {
				item.Output[in.ID] = text
			}
		}
		bad := incompleteDocumentGroups(item)
		goodProse := 0
		for _, in := range item.Inputs {
			if !in.Annotation {
				if _, failed := bad[in.ID]; !failed {
					goodProse++
				}
			}
		}
		measured := attempt
		measured.BatchUnits = nil
		for _, i := range indices {
			measured.BatchUnits = append(measured.BatchUnits, (*units)[i].Ordinal+1)
		}
		if index != indices[0] {
			owner := (*units)[indices[0]].Ordinal
			measured.SharedWithOrdinal = &owner
			measured.Performance = nil
			measured.PromptTokens, measured.OutputTokens = 0, 0
			measured.OutputTokensEstimated = false
			measured.ReasoningChars, measured.ContentChars = 0, 0
		}
		if goodProse > 0 && (len(bad) == 0 || item.QualityRetries < 2) {
			if len(bad) > 0 {
				repair := extractDocumentRepair(&item, bad, len(*units)+len(repairs))
				if repair != nil {
					repairs = append(repairs, *repair)
				}
			}
			measured.Error = ""
			markDocumentUnitComplete(*run, &item, measured)
			accepted = true
		} else {
			item.Output = nil
			item.Status = "pending"
			item.Error = attempt.Error
			item.Attempts = append(item.Attempts, measured)
			if len(item.Attempts)-item.RetryBase >= 3 {
				item.Status = "needs_review"
			}
		}
		changed = append(changed, item)
	}
	if !accepted {
		return false, nil
	}
	updated := append([]workflows.DocumentUnit(nil), (*units)...)
	for i, index := range indices {
		updated[index] = changed[i]
	}
	updated = append(updated, repairs...)
	previous := *run.Document
	run.Document.Total = len(updated)
	run.Document.Completed = 0
	for _, item := range updated {
		if item.Status == "completed" {
			run.Document.Completed++
		}
	}
	run.Document.Activity = "retrying"
	run.Document.Error = "검증된 문단을 저장했습니다. 누락되거나 불완전한 문단만 다시 처리합니다."
	if err := s.db.SaveDocumentProgress(run, append(changed, repairs...)...); err != nil {
		*run.Document = previous
		return false, err
	}
	*units = updated
	return true, nil
}

// Repairs may run before already committed later paragraphs. Previous context
// must come from the source position, never from the last completed API call.
func documentContextBefore(units []workflows.DocumentUnit, next workflows.DocumentUnit) string {
	if len(next.Inputs) == 0 {
		return ""
	}
	target := next.Inputs[0]
	type fragment struct {
		input workflows.DocumentInput
		text  string
	}
	var fragments []fragment
	for _, unit := range units {
		if unit.Status != "completed" {
			continue
		}
		for _, in := range unit.Inputs {
			if in.Source == target.Source && (in.Block < target.Block || in.Block == target.Block && in.Part < target.Part) {
				if text := unit.Output[in.ID]; text != "" {
					fragments = append(fragments, fragment{in, text})
				}
			}
		}
	}
	sort.SliceStable(fragments, func(i, j int) bool {
		if fragments[i].input.Block != fragments[j].input.Block {
			return fragments[i].input.Block < fragments[j].input.Block
		}
		return fragments[i].input.Part < fragments[j].input.Part
	})
	var context strings.Builder
	for _, fragment := range fragments {
		context.WriteString(fragment.text)
		context.WriteString(" ")
	}
	text := []rune(context.String())
	return string(text[max(0, len(text)-800):])
}
