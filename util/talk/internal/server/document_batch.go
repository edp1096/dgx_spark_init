package server

import (
	"strings"
	"unicode/utf8"

	"sparktalk/internal/workflows"
)

// The immutable checkpoint units remain separate. A request may process adjacent
// pending units together so they share reasoning and narrative context. Any
// failed batch falls back to individual checkpoints; complete paragraphs from an
// unambiguous partial response are retained by recoverDocumentParagraphs.
func documentRequestBatch(run workflows.Run, units []workflows.DocumentUnit, index, window int) (workflows.DocumentUnit, []int) {
	combined := units[index]
	combined.Inputs = documentRequestInputs(combined)
	indices := []int{index}
	if run.Document.Kind != "translate" || (run.Document.ModelType != "qwen38fn_exl3" && run.Document.ModelType != "qwen3.8") || combined.Stage != "process" || len(combined.Attempts) > combined.RetryBase {
		return combined, indices
	}
	chars := 0
	for _, in := range combined.Inputs {
		chars += utf8.RuneCountInString(in.Text)
	}
	for i := index + 1; i < len(units) && len(indices) < 4; i++ {
		next := units[i]
		if next.Status == "completed" || next.Status == "needs_review" || next.Stage != "process" || len(next.Attempts) > next.RetryBase {
			break
		}
		n := 0
		for _, in := range next.Inputs {
			n += utf8.RuneCountInString(in.Text)
		}
		if chars+n > 9000 {
			break
		}
		// Appended split/repair units can precede the next ordinal in source
		// order. Do not batch across an unprocessed source fragment.
		gap := false
		for j, pending := range units {
			if j == i || pending.Status == "completed" {
				continue
			}
			included := false
			for _, chosen := range indices {
				if j == chosen {
					included = true
					break
				}
			}
			if !included && documentUnitBefore(pending, next) {
				gap = true
				break
			}
		}
		if gap {
			break
		}
		candidate := combined
		candidate.Inputs = append(append([]workflows.DocumentInput(nil), combined.Inputs...), documentRequestInputs(next)...)
		candidate.OutputBudget = 0
		wire := documentWireEntries(candidate)
		if len(wire) > 128 {
			break
		}
		if documentReasoningBudget(run)+documentAnswerBudget(run, candidate) > documentOutputBudget(run, candidate) {
			break
		}
		// The request builder also counts the complete prompt/schema before sending.
		// This preflight leaves room for source metadata, context tail and JSON keys.
		estimate := 4096
		for _, entry := range wire {
			estimate += estimateTextTokens(entry.Text) + estimateTextTokens(entry.Readings) + 64
		}
		if window > 0 && estimate+documentOutputBudget(run, candidate)+1024 > window {
			break
		}
		combined = candidate
		indices = append(indices, i)
		chars += n
	}
	if len(indices) > 1 {
		combined.OutputBudget = 0
	}
	return combined, indices
}

func documentContextTail(unit workflows.DocumentUnit) string {
	var text strings.Builder
	for _, in := range unit.Inputs {
		text.WriteString(unit.Output[in.ID])
		text.WriteString(" ")
	}
	runes := []rune(text.String())
	if len(runes) > 800 {
		runes = runes[len(runes)-800:]
	}
	return string(runes)
}

func markDocumentUnitComplete(run workflows.Run, unit *workflows.DocumentUnit, attempt workflows.DocumentAttempt) {
	unit.Attempts = append(unit.Attempts, attempt)
	unit.Status = "completed"
	unit.Error = ""
	if run.Document.Kind != "translate" {
		return
	}
	for source, target := range activeDocumentGlossary(run.Document) {
		for _, group := range documentTextGroups(*unit) {
			if strings.Contains(group.Source, source) && !strings.Contains(group.Output, target) {
				unit.Warnings = append(unit.Warnings, "glossary spelling requires review: "+source+" → "+target)
				break
			}
		}
	}
	bad := shortDocumentOutputs(*unit)
	for _, in := range unit.Inputs {
		if warning, ok := bad[in.ID]; ok {
			unit.Warnings = append(unit.Warnings, warning)
		}
	}
}

// Ordinals identify checkpoints; source positions determine processing order.
// A split is appended in storage so existing checkpoint IDs remain stable.
func documentUnitBefore(a, b workflows.DocumentUnit) bool {
	if len(a.Inputs) == 0 || len(b.Inputs) == 0 {
		return a.Ordinal < b.Ordinal
	}
	x, y := a.Inputs[0], b.Inputs[0]
	if x.Source != y.Source {
		return a.Ordinal < b.Ordinal
	}
	if x.Block != y.Block {
		return x.Block < y.Block
	}
	if x.Part != y.Part {
		return x.Part < y.Part
	}
	return a.Ordinal < b.Ordinal
}
