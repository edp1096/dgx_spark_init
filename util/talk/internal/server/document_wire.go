package server

import (
	"fmt"
	"sort"
	"strings"

	"sparktalk/internal/sourcebook"
	"sparktalk/internal/workflows"
)

// Wire entries are transient. Stored source IDs, checkpoints and export slots
// remain unchanged. Only paragraphs proved plain by the saved DOM may coalesce.
type documentWireEntry struct {
	ID, Paragraph, Source, Text, Readings string
	Scope                                 string
	Whitespace                            string
	Indexes                               []int
}

func documentWireEntries(u workflows.DocumentUnit) []documentWireEntry {
	indexes := map[string]int{}
	for i, in := range u.Inputs {
		indexes[in.ID] = i
	}
	var entries []documentWireEntry
	for g, group := range documentTextGroups(u) {
		plain := len(group.Inputs) > 0
		groupScopes := map[string]bool{}
		previous := -1
		mode := ""
		for _, in := range group.Inputs {
			position := indexes[in.ID]
			if previous >= 0 && position != previous+1 {
				plain = false
			}
			previous = position
			if !in.Annotation && (!in.Plain || in.Group == "") {
				plain = false
			}
			if !in.Annotation {
				groupScopes[in.Scope] = true
				if mode != "" && mode != in.Whitespace {
					plain = false
				}
				mode = in.Whitespace
			}
		}
		paragraph := fmt.Sprintf("g%06d", g+1)
		readings := documentReadingHints(group.Inputs)
		first := len(entries)
		for _, in := range group.Inputs {
			if in.Annotation {
				continue
			}
			i := indexes[in.ID]
			if plain && len(entries) > first {
				entry := &entries[len(entries)-1]
				entry.Text += in.Before + in.Text
				entry.Indexes = append(entry.Indexes, i)
			} else {
				scope := in.Scope
				// A span covering the entire paragraph is already protected by
				// paragraph coverage and relocation rules; avoid redundant input.
				if len(groupScopes) == 1 {
					scope = ""
				}
				entries = append(entries, documentWireEntry{ID: documentLocalID(i), Paragraph: paragraph, Scope: scope, Source: in.Source, Text: in.Text, Whitespace: in.Whitespace, Indexes: []int{i}})
			}
		}
		if len(entries) > first {
			entries[first].Readings = readings
		}
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Indexes[0] < entries[j].Indexes[0] })
	for i := range entries {
		entries[i].Text = sourcebook.NormalizeWhitespace(entries[i].Text, entries[i].Whitespace)
	}
	return entries
}

// Readings are small context hints, not output slots. Keep the associated base
// fragment so readings of names are still available after prose is coalesced.
func documentReadingHints(inputs []workflows.DocumentInput) string {
	var result []string
	base := ""
	for i := 0; i < len(inputs); {
		in := inputs[i]
		if !in.Annotation {
			base = in.Text
			i++
			continue
		}
		var hint strings.Builder
		for i < len(inputs) && inputs[i].Annotation {
			hint.WriteString(inputs[i].Text)
			i++
		}
		if text := strings.TrimSpace(hint.String()); text != "" {
			result = append(result, strings.TrimSpace(base)+" → "+text)
		}
	}
	return strings.Join(result, "; ")
}

func documentWirePayload(u workflows.DocumentUnit) ([]map[string]string, []string) {
	entries := documentWireEntries(u)
	input := make([]map[string]string, 0, len(entries))
	ids := make([]string, 0, len(entries))
	scopes := map[string]string{}
	for _, entry := range entries {
		v := map[string]string{"id": entry.ID, "paragraph": entry.Paragraph, "text": entry.Text}
		if entry.Whitespace == "pre-wrap" || entry.Whitespace == "pre-line" {
			v["layout"] = "preserve_lines"
		}
		if entry.Scope != "" {
			if scopes[entry.Scope] == "" {
				scopes[entry.Scope] = fmt.Sprintf("s%06d", len(scopes)+1)
			}
			v["scope"] = scopes[entry.Scope]
		}
		if entry.Readings != "" {
			v["reading_hints"] = entry.Readings
		}
		input = append(input, v)
		ids = append(ids, entry.ID)
	}
	return input, ids
}
