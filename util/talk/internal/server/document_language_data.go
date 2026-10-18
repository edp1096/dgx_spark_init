package server

import (
	"fmt"
	"maps"
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"

	"sparktalk/internal/workflows"
)

const documentLanguageVersion = "language_v1"

var documentLanguageURL = regexp.MustCompile(`(?i)(?:https?://|www\.)[^\s<>"'「」]+`)

// Candidates are observations, never errors by themselves. Mixed alphabets,
// names, code and legitimate quotations are allowed. Source membership uses
// whole Latin words, so a different name's substring cannot hide a candidate.
func documentLatinWords(text string) []workflows.DocumentLanguageMention {
	var spans []workflows.DocumentLanguageMention
	start := -1
	flush := func(end int) {
		if start >= 0 && utf8.RuneCountInString(text[start:end]) >= 2 {
			spans = append(spans, workflows.DocumentLanguageMention{Start: start, End: end, Text: text[start:end]})
		}
		start = -1
	}
	for i, r := range text {
		if unicode.Is(unicode.Latin, r) || start >= 0 && unicode.IsMark(r) {
			if start < 0 {
				start = i
			}
		} else if start >= 0 && strings.ContainsRune("-'’—–", r) {
			next, _ := utf8.DecodeRuneInString(text[i+utf8.RuneLen(r):])
			if !unicode.Is(unicode.Latin, next) {
				flush(i)
			}
		} else {
			flush(i)
		}
	}
	flush(len(text))
	return spans
}

func documentLatinKey(text string) string {
	return strings.ToLower(strings.NewReplacer("’", "'", "—", "-", "–", "-").Replace(text))
}

func documentLanguageProtected(text string) [][2]int {
	var spans [][2]int
	for _, r := range documentLanguageURL.FindAllStringIndex(text, -1) {
		spans = append(spans, [2]int{r[0], r[1]})
	}
	type opening struct {
		start int
		close rune
	}
	var stack []opening
	for i, r := range text {
		if r == '(' || r == '（' {
			close := ')'
			if r == '（' {
				close = '）'
			}
			stack = append(stack, opening{i + utf8.RuneLen(r), close})
			continue
		}
		if len(stack) == 0 || r != stack[len(stack)-1].close {
			continue
		}
		o := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		inside := text[o.start:i]
		if strings.ContainsFunc(inside, func(r rune) bool { return unicode.Is(unicode.Latin, r) }) && !strings.ContainsFunc(inside, func(r rune) bool { return unicode.IsLetter(r) && !unicode.Is(unicode.Latin, r) }) {
			spans = append(spans, [2]int{o.start, i})
		}
	}
	return spans
}

func documentLanguageScan(units []workflows.DocumentUnit) *workflows.DocumentLanguageCheck {
	scan := &workflows.DocumentLanguageCheck{Version: documentLanguageVersion}
	for _, u := range units {
		for _, g := range documentTextGroups(u) {
			scan.Scanned++
			var original strings.Builder
			for _, in := range g.Inputs {
				if !in.Annotation {
					original.WriteString(in.Before)
					original.WriteString(in.Text)
				}
			}
			g.Source = original.String()
			if !strings.ContainsFunc(g.Output, func(r rune) bool { return unicode.Is(unicode.Hangul, r) }) {
				continue
			}
			sourceWords, targetWords := map[string]int{}, map[string]int{}
			for _, w := range documentLatinWords(g.Source) {
				sourceWords[documentLatinKey(w.Text)]++
			}
			for _, w := range documentLatinWords(g.Output) {
				targetWords[documentLatinKey(w.Text)]++
			}
			var mentions []workflows.DocumentLanguageMention
			words := documentLatinWords(g.Output)
			protectedRanges := documentLanguageProtected(g.Output)
			protectedWord := func(w workflows.DocumentLanguageMention) bool {
				for _, p := range protectedRanges {
					if w.Start < p[1] && w.End > p[0] {
						return true
					}
				}
				return false
			}
			for wi := 0; wi < len(words); {
				m := words[wi]
				if protectedWord(m) {
					wi++
					continue
				}
				n := wi + 1
				candidate := targetWords[documentLatinKey(m.Text)] > sourceWords[documentLatinKey(m.Text)]
				for n < len(words) {
					if protectedWord(words[n]) {
						break
					}
					between := g.Output[m.End:words[n].Start]
					if strings.ContainsFunc(between, func(r rune) bool {
						return !unicode.IsSpace(r) && !strings.ContainsRune(",:;.!?\"'’‘“”-—–_", r)
					}) || utf8.RuneCountInString(g.Output[m.Start:words[n].End]) > 512 {
						break
					}
					m.End = words[n].End
					candidate = candidate || targetWords[documentLatinKey(words[n].Text)] > sourceWords[documentLatinKey(words[n].Text)]
					n++
				}
				wi = n
				if !candidate {
					continue
				}
				// Include the attached Korean inflection, which may need to change
				// with a translated noun/pronoun. Never consume adjacent prose.
				for _, r := range g.Output[m.End:] {
					if !unicode.Is(unicode.Hangul, r) {
						break
					}
					m.End += utf8.RuneLen(r)
				}
				m.Text = g.Output[m.Start:m.End]
				protected := false
				for _, url := range protectedRanges {
					protected = protected || m.Start < url[1] && m.End > url[0]
				}
				if protected {
					continue
				}
				m.ID = fmt.Sprintf("m%d", len(mentions)+1)
				mentions = append(mentions, m)
			}
			if len(mentions) == 0 {
				continue
			}
			ids := []string{}
			preserve := false
			for _, in := range g.Inputs {
				if !in.Annotation {
					ids = append(ids, in.ID)
					preserve = preserve || in.Whitespace == "pre-wrap"
				}
			}
			scan.Records = append(scan.Records, workflows.DocumentLanguageRecord{DocumentTermReference: workflows.DocumentTermReference{ID: fmt.Sprintf("l%06d", len(scan.Records)+1), Ordinal: u.Ordinal, InputIDs: ids, Source: g.Source, Target: g.Output, SourceHash: documentTextHash(g.Source), TargetHash: documentTextHash(g.Output), Status: "pending"}, Mentions: mentions, PreserveLines: preserve})
		}
	}
	return scan
}

// The model selects an id, not a reconstructed quotation. Overlapping exact
// spans retain punctuation and provide nearby source context for pronouns.
func documentLanguageEvidence(text string) map[string]string {
	runes := []rune(text)
	out := map[string]string{}
	for i := 0; i < len(runes); i += 140 {
		out[fmt.Sprintf("s%d", len(out)+1)] = string(runes[i:min(i+200, len(runes))])
		if i+200 >= len(runes) {
			break
		}
	}
	return out
}

func applyDocumentLanguage(units []workflows.DocumentUnit, r workflows.DocumentLanguageRecord) (map[string]string, string, error) {
	current, err := termReferenceTarget(units, r.DocumentTermReference)
	if err != nil || documentTextHash(current) != r.TargetHash {
		return nil, "", fmt.Errorf("language-check target changed")
	}
	if len(r.Checks) != len(r.Mentions) {
		return nil, "", fmt.Errorf("incomplete language checks")
	}
	evidence := documentLanguageEvidence(r.Source)
	out := maps.Clone(units[r.Ordinal].Output)
	for i := len(r.Mentions) - 1; i >= 0; i-- {
		m := r.Mentions[i]
		decision, ok := r.Checks[m.ID]
		if !ok || strings.TrimSpace(decision.Reason) == "" {
			return nil, "", fmt.Errorf("missing language decision %s", m.ID)
		}
		if decision.Status == "retain" || decision.Status == "uncertain" {
			if decision.After != "" {
				return nil, "", fmt.Errorf("retained expression has a replacement")
			}
			continue
		}
		phrase := len(strings.Fields(m.Text)) > 1
		limit := 96
		if phrase {
			limit = min(1024, utf8.RuneCountInString(m.Text)*3+32)
		}
		if decision.Status != "translate" || evidence[decision.SourceRef] == "" || strings.TrimSpace(decision.After) == "" || decision.After == m.Text || utf8.RuneCountInString(decision.After) > limit || !phrase && len(strings.Fields(decision.After)) > 8 {
			return nil, "", fmt.Errorf("invalid language repair %s", m.ID)
		}
		// A foreign subject followed by 가 needs the contracted Korean pronoun
		// form. Keep adjectival 저가 (low-cost) outside this particle pattern.
		if strings.HasSuffix(m.Text, "가") && (decision.After == "나가" || decision.After == "저가" || decision.After == "너가") {
			return nil, "", fmt.Errorf("Korean personal subject pronoun requires its contracted nominative form; preserve the source person and register")
		}
		for _, ch := range decision.After {
			if phrase && (unicode.IsSpace(ch) || strings.ContainsRune(",:;.!?\"'’‘“”()「」『』_—–", ch)) {
				continue
			}
			if !unicode.IsLetter(ch) && !unicode.IsDigit(ch) && !unicode.IsMark(ch) && ch != ' ' && ch != '-' && ch != '\'' && ch != '’' {
				return nil, "", fmt.Errorf("language repair exceeds word scope")
			}
		}
		if r.PreserveLines && strings.Count(m.Text, "\n") != strings.Count(decision.After, "\n") {
			return nil, "", fmt.Errorf("language repair changes authored line breaks")
		}
		if m.Start < 0 || m.End > len(current) || m.Start >= m.End || current[m.Start:m.End] != m.Text {
			return nil, "", fmt.Errorf("language occurrence changed")
		}
		pos := 0
		applied := false
		for _, id := range r.InputIDs {
			v := units[r.Ordinal].Output[id]
			if m.Start >= pos && m.End <= pos+len(v) {
				out[id] = out[id][:m.Start-pos] + decision.After + out[id][m.End-pos:]
				applied = true
				break
			}
			pos += len(v)
		}
		if !applied {
			return nil, "", fmt.Errorf("language repair crosses protected inline nodes")
		}
	}
	var joined strings.Builder
	for _, id := range r.InputIDs {
		joined.WriteString(out[id])
	}
	return out, joined.String(), nil
}
