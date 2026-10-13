package server

import (
	"fmt"
	"sparktalk/internal/sourcebook"
	"sparktalk/internal/workflows"
)

const documentOutputProtocol = "json_schema_paragraphs_v7"

// encoding/json sorts schema property names. Constrained decoders can emit in
// that order, so p1,p10,p100 would force translation out of source order.
func documentLocalID(index int) string { return fmt.Sprintf("p%06d", index+1) }

func decorateDocumentInputs(units []workflows.DocumentUnit, source string, book *sourcebook.Book) {
	slots := map[string]sourcebook.TranslationSlot{}
	for _, slot := range book.TranslationSlots() {
		slots[slot.ID] = slot
	}
	for i := range units {
		if units[i].Status == "completed" {
			continue
		}
		for j := range units[i].Inputs {
			in := &units[i].Inputs[j]
			if in.Source != source {
				continue
			}
			if slot, ok := slots[in.Block]; ok {
				in.Group, in.Annotation = source+":"+slot.Group, slot.Annotation
				in.Plain, in.Before = slot.Plain, slot.Before
				in.Whitespace = slot.Whitespace
				in.Scope = ""
				if slot.Scope != "" {
					in.Scope = source + ":" + slot.Scope
				}
				if in.Part > 0 {
					in.Before = ""
				}
			}
		}
	}
}

func documentTranslationSlots(u workflows.DocumentUnit) []sourcebook.TranslationSlot {
	slots := make([]sourcebook.TranslationSlot, 0, len(u.Inputs))
	for _, in := range u.Inputs {
		slots = append(slots, sourcebook.TranslationSlot{ID: in.ID, Group: in.Group, Text: in.Text, Annotation: in.Annotation})
	}
	return slots
}

// A request may batch checkpoints, but words must never move to another
// checkpoint: each committed checkpoint must independently cover its prose.
func documentRequestInputs(u workflows.DocumentUnit) []workflows.DocumentInput {
	out := append([]workflows.DocumentInput(nil), u.Inputs...)
	for i := range out {
		if out[i].Group != "" {
			out[i].Group = fmt.Sprintf("unit%d:%s", u.Ordinal, out[i].Group)
		}
		if out[i].Scope != "" {
			out[i].Scope = fmt.Sprintf("unit%d:%s", u.Ordinal, out[i].Scope)
		}
	}
	return out
}

type documentTextGroup struct {
	Inputs         []workflows.DocumentInput
	Source, Output string
}

func documentTextGroups(u workflows.DocumentUnit) []documentTextGroup {
	index := map[string]int{}
	groups := []documentTextGroup{}
	for _, in := range u.Inputs {
		key := in.Group
		if key == "" {
			key = in.ID
		}
		i, ok := index[key]
		if !ok {
			i = len(groups)
			index[key] = i
			groups = append(groups, documentTextGroup{})
		}
		groups[i].Inputs = append(groups[i].Inputs, in)
		if !in.Annotation {
			groups[i].Source += in.Text
			groups[i].Output += u.Output[in.ID]
		}
	}
	return groups
}
