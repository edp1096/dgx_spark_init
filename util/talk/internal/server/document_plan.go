package server

import (
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"

	"sparktalk/internal/sourcebook"
	"sparktalk/internal/workflows"
)

// Choose a sentence/dialogue boundary near the requested position. Failure is
// explicit: an unbroken word, code token or number must not be cut in half.
func documentSentenceSplit(text []rune, preferred int) int {
	best, distance := -1, len(text)
	for i, r := range text {
		if r != '。' && r != '！' && r != '？' && r != '!' && r != '?' && r != '.' {
			continue
		}
		if r == '.' {
			if i > 0 && i+1 < len(text) && unicode.IsDigit(text[i-1]) && unicode.IsDigit(text[i+1]) {
				continue
			}
			start := i
			for start > 0 && unicode.IsLetter(text[start-1]) {
				start--
			}
			word := strings.ToLower(string(text[start:i]))
			if len([]rune(word)) <= 1 || strings.Contains("|mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|e|g|i|", "|"+word+"|") {
				continue
			}
		}
		end := i + 1
		for end < len(text) && strings.ContainsRune("\"'”’」』）)", text[end]) {
			end++
		}
		if r == '.' && end < len(text) && !unicode.IsSpace(text[end]) {
			continue
		}
		for end < len(text) && unicode.IsSpace(text[end]) {
			end++
		}
		if end < 80 || len(text)-end < 80 {
			continue
		}
		d := end - preferred
		if d < 0 {
			d = -d
		}
		if d < distance {
			best, distance = end, d
		}
	}
	return best
}

// Pack complete structural paragraphs into durable checkpoints. The existing
// request batcher still combines checkpoints, avoiding one call per paragraph.
func planDocumentTranslation(source string, book *sourcebook.Book, start, end int) []workflows.DocumentUnit {
	slots := map[string]sourcebook.TranslationSlot{}
	for _, slot := range book.TranslationSlots() {
		slots[slot.ID] = slot
	}
	var groups [][]workflows.DocumentInput
	var groupKeys []string
	for _, block := range book.Blocks {
		if end > 0 && block.Start >= end || block.End <= start {
			continue
		}
		text := []rune(block.Text)
		from := max(0, start-block.Start)
		to := len(text)
		if end > 0 {
			to = min(to, end-block.Start)
		}
		if from >= to {
			continue
		}
		slot := slots[block.ID]
		key := source + ":" + slot.Group
		if slot.Group == "" {
			key = source + ":" + block.ID
		}
		if len(groups) == 0 || groupKeys[len(groupKeys)-1] != key {
			groups = append(groups, nil)
			groupKeys = append(groupKeys, key)
		}
		in := workflows.DocumentInput{ID: fmt.Sprintf("%s:%s:%d", source, block.ID, from), Source: source, Block: block.ID, Part: from, Text: string(text[from:to]), Group: key, Chapter: block.Chapter, Annotation: slot.Annotation, Plain: slot.Plain, Before: slot.Before, Whitespace: slot.Whitespace}
		if slot.Scope != "" {
			in.Scope = source + ":" + slot.Scope
		}
		if from > 0 {
			in.Before = ""
		}
		groups[len(groups)-1] = append(groups[len(groups)-1], in)
	}
	var units []workflows.DocumentUnit
	current := workflows.DocumentUnit{Stage: "process", Status: "pending"}
	chars := 0
	flush := func() {
		if len(current.Inputs) > 0 {
			current.Ordinal = len(units)
			units = append(units, current)
			current = workflows.DocumentUnit{Stage: "process", Status: "pending"}
			chars = 0
		}
	}
	for _, group := range groups {
		n := 0
		for _, in := range group {
			n += utf8.RuneCountInString(in.Text)
		}
		if chars+n > 2600 {
			flush()
		}
		// Pack an exceptional paragraph only at a safe sentence boundary.
		for n > 6000 {
			left, right, ok := splitDocumentPlainParagraph(group, 2600)
			if !ok {
				break
			}
			current.Inputs = left
			flush()
			group = right
			n = 0
			for _, in := range group {
				n += utf8.RuneCountInString(in.Text)
			}
		}
		current.Inputs = append(current.Inputs, group...)
		chars += n
	}
	flush()
	return units
}

// Split long plain paragraphs spanning several DOM/ruby nodes, preserving every
// original block offset and keeping ruby pronunciation with its base fragment.
func splitDocumentPlainParagraph(inputs []workflows.DocumentInput, preferred int) ([]workflows.DocumentInput, []workflows.DocumentInput, bool) {
	var source strings.Builder
	for _, in := range inputs {
		if in.Annotation {
			continue
		}
		if !in.Plain || in.Scope != "" {
			return nil, nil, false
		}
		source.WriteString(in.Before)
		source.WriteString(in.Text)
	}
	cut := documentSentenceSplit([]rune(source.String()), preferred)
	if cut < 0 {
		return nil, nil, false
	}
	at := 0
	for i, in := range inputs {
		if in.Annotation {
			continue
		}
		before, n := utf8.RuneCountInString(in.Before), utf8.RuneCountInString(in.Text)
		if cut > at+before+n {
			at += before + n
			continue
		}
		if cut <= at+before {
			if i == 0 {
				return nil, nil, false
			}
			return append([]workflows.DocumentInput(nil), inputs[:i]...), append([]workflows.DocumentInput(nil), inputs[i:]...), true
		}
		part := cut - at - before
		if part == n {
			end := i + 1
			for end < len(inputs) && inputs[end].Annotation {
				end++
			}
			if end == len(inputs) {
				return nil, nil, false
			}
			return append([]workflows.DocumentInput(nil), inputs[:end]...), append([]workflows.DocumentInput(nil), inputs[end:]...), true
		}
		if i+1 < len(inputs) && inputs[i+1].Annotation {
			return nil, nil, false
		}
		left := append([]workflows.DocumentInput(nil), inputs[:i+1]...)
		right := append([]workflows.DocumentInput(nil), inputs[i:]...)
		chars := []rune(in.Text)
		left[len(left)-1].Text = string(chars[:part])
		right[0].Text = string(chars[part:])
		right[0].Part += part
		right[0].ID = fmt.Sprintf("%s:%s:%d", in.Source, in.Block, right[0].Part)
		right[0].Before = ""
		return left, right, true
	}
	return nil, nil, false
}
