package server

import (
	"encoding/json"
	"sort"

	"sparktalk/internal/workflows"
)

// Context references have no output IDs. Complete neighbouring source groups
// are admitted under a token budget; the original input spans stay immutable.
func documentSourceContext(units []workflows.DocumentUnit, target workflows.DocumentUnit, budget int) string {
	if len(target.Inputs) == 0 || budget <= 0 {
		return ""
	}
	selected := map[string]bool{}
	for _, in := range target.Inputs {
		selected[in.ID] = true
	}
	first, last := target.Inputs[0], target.Inputs[len(target.Inputs)-1]
	type neighbour struct {
		input workflows.DocumentInput
		text  string
	}
	var before, after []neighbour
	for _, unit := range units {
		for _, group := range documentTextGroups(unit) {
			if len(group.Inputs) == 0 || group.Source == "" {
				continue
			}
			start, end := group.Inputs[0], group.Inputs[len(group.Inputs)-1]
			overlaps := false
			for _, in := range group.Inputs {
				overlaps = overlaps || selected[in.ID]
			}
			if overlaps || start.Source != first.Source {
				continue
			}
			if end.Block < first.Block || end.Block == first.Block && end.Part < first.Part {
				before = append(before, neighbour{start, group.Source})
			} else if start.Block > last.Block || start.Block == last.Block && start.Part > last.Part {
				after = append(after, neighbour{start, group.Source})
			}
		}
	}
	less := func(a, b neighbour) bool {
		if a.input.Block != b.input.Block {
			return a.input.Block < b.input.Block
		}
		return a.input.Part < b.input.Part
	}
	sort.Slice(before, func(i, j int) bool { return less(before[j], before[i]) })
	sort.Slice(after, func(i, j int) bool { return less(after[i], after[j]) })
	var refs []map[string]string
	for n := 0; n < 2; n++ {
		for side, list := range [][]neighbour{before, after} {
			if n >= len(list) {
				continue
			}
			item := list[n]
			direction := "before"
			if side == 1 {
				direction = "after"
			}
			candidate := append(append([]map[string]string(nil), refs...), map[string]string{"source_ref": item.input.ID, "position": direction, "text": item.text})
			raw, _ := json.Marshal(candidate)
			if estimateTextTokens(string(raw)) <= budget {
				refs = candidate
			}
		}
	}
	if len(refs) == 0 {
		return ""
	}
	raw, _ := json.Marshal(refs)
	return string(raw)
}
