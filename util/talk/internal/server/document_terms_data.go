package server

import (
	"fmt"
	"maps"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"sparktalk/internal/llm"
	"sparktalk/internal/workflows"
)

func sortedTermKeys[T any](m map[string]T) []string {
	a := make([]string, 0, len(m))
	for k := range m {
		a = append(a, k)
	}
	sort.Strings(a)
	return a
}

func termSourceMention(text, name string) bool {
	// TXT line wrapping and HTML inline whitespace do not split an identity.
	// Normalization is for indexing only; source text/offsets stay immutable.
	return documentEntityMention(strings.Join(strings.Fields(text), " "), strings.Join(strings.Fields(name), " "))
}

func documentTermCases(run workflows.Run, units []workflows.DocumentUnit) []workflows.DocumentTermCase {
	names := map[string]bool{}
	for name := range run.Document.UserGlossary {
		names[name] = true
	}
	for _, entity := range run.Document.Terms.Candidates {
		if entity.Source != "" {
			names[entity.Source] = true
		}
	}
	var cases []workflows.DocumentTermCase
	for _, name := range sortedTermKeys(names) {
		c := workflows.DocumentTermCase{Source: name, Pinned: run.Document.UserGlossary[name], Status: "pending", Attempts: map[string]int{}}
		for _, u := range units {
			for _, g := range documentTextGroups(u) {
				var original strings.Builder
				for _, in := range g.Inputs {
					if !in.Annotation {
						original.WriteString(in.Before)
						original.WriteString(in.Text)
					}
				}
				sourceText := original.String()
				if !termSourceMention(sourceText, name) {
					continue
				}
				ids := []string{}
				for _, in := range g.Inputs {
					if !in.Annotation {
						ids = append(ids, in.ID)
					}
				}
				if len(ids) == 0 {
					continue
				}
				c.References = append(c.References, workflows.DocumentTermReference{ID: fmt.Sprintf("r%06d", len(c.References)+1), Ordinal: u.Ordinal, InputIDs: ids, Source: sourceText, Target: g.Output, SourceHash: documentTextHash(sourceText), TargetHash: documentTextHash(g.Output), Status: "pending"})
			}
		}
		if len(c.References) > 0 {
			cases = append(cases, c)
		}
	}
	// Specific compound names before their shorter overlapping forms.
	sort.SliceStable(cases, func(i, j int) bool { return len(cases[i].Source) > len(cases[j].Source) })
	return cases
}

func validateTermExtraction(c *workflows.DocumentTermCase, refs []workflows.DocumentTermReference, result llm.StreamResult) map[string]error {
	var payload struct {
		Checks map[string]struct {
			Status   string   `json:"status"`
			Forms    []string `json:"forms"`
			Evidence string   `json:"evidence"`
			Referent string   `json:"referent"`
			Reason   string   `json:"reason"`
		} `json:"checks"`
	}
	errors := map[string]error{}
	if err := decodeTermJSON(result, &payload); err != nil {
		for _, r := range refs {
			errors[r.ID] = err
		}
		return errors
	}
	expected := map[string]bool{}
	for _, r := range refs {
		expected[r.ID] = true
	}
	for id := range payload.Checks {
		if !expected[id] {
			for _, r := range refs {
				errors[r.ID] = fmt.Errorf("unexpected term reference %s", id)
			}
			return errors
		}
	}
	for _, r := range refs {
		item, ok := payload.Checks[r.ID]
		if !ok {
			errors[r.ID] = fmt.Errorf("missing term reference %s", r.ID)
			continue
		}
		if item.Status != "named" && item.Status != "ordinary" && item.Status != "implicit" && item.Status != "uncertain" {
			errors[r.ID] = fmt.Errorf("invalid mention status")
			continue
		}
		quote, anchored := documentAnchoredQuote(r.Source, item.Evidence)
		if !anchored || !termSourceMention(quote, c.Source) {
			errors[r.ID] = fmt.Errorf("term evidence not anchored to original expression")
			continue
		}
		forms := []string{}
		seen := map[string]bool{}
		for _, form := range item.Forms {
			form = strings.Join(strings.Fields(form), " ")
			if form == "" || utf8.RuneCountInString(form) > 96 || len(termRanges(r.Target, form)) == 0 {
				errors[r.ID] = fmt.Errorf("translated form is absent or inside another word: %q", form)
				break
			}
			if !seen[form] {
				forms = append(forms, form)
				seen[form] = true
			}
		}
		if errors[r.ID] != nil {
			continue
		}
		if item.Status == "named" && (len(forms) == 0 || strings.TrimSpace(item.Referent) == "") {
			errors[r.ID] = fmt.Errorf("named referent needs target forms and identity")
			continue
		}
		for i := range c.References {
			x := &c.References[i]
			if x.ID == r.ID {
				x.Status = item.Status
				x.Forms = forms
				x.Evidence = quote
				x.Referent = item.Referent
				x.Reason = item.Reason
				preserveTermPair(x)
				break
			}
		}
	}
	return errors
}

// A translated name followed by a parenthesized pronunciation/alias is an
// intentional pair, not interchangeable spellings. Do not flatten such pairs
// even when a model calls one of them an error. Once explicitly introduced,
// either member can appear alone later in the paragraph. An unrelated third
// spelling still needs review.
func preserveTermPair(r *workflows.DocumentTermReference) bool {
	if r.Status != "named" || len(r.Forms) == 0 {
		return false
	}
	type mention struct {
		form       string
		start, end int
	}
	var spans []mention
	forms := map[string]bool{}
	wholePairs := map[string][2]string{}
	for _, form := range r.Forms {
		forms[form] = true
		for _, brackets := range [][2]string{{"(", ")"}, {"（", "）"}, {"[", "]"}, {"［", "］"}} {
			left, right, found := strings.Cut(form, brackets[0])
			if !found || !strings.HasSuffix(right, brackets[1]) {
				continue
			}
			left, right = strings.TrimSpace(left), strings.TrimSpace(strings.TrimSuffix(right, brackets[1]))
			if left == "" || right == "" || strings.ContainsAny(right, "()（）[]［］") {
				continue
			}
			forms[left], forms[right] = true, true
			wholePairs[form] = [2]string{left, right}
		}
	}
	for _, form := range sortedTermKeys(forms) {
		for _, span := range termRanges(r.Target, form) {
			spans = append(spans, mention{form, span[0], span[1]})
		}
	}
	paired := map[string]bool{}
	for _, a := range spans {
		for _, b := range spans {
			if a.form == b.form || a.end > b.start {
				continue
			}
			between := strings.TrimSpace(r.Target[a.end:b.start])
			closing := ""
			switch between {
			case "(":
				closing = ")"
			case "（":
				closing = "）"
			case "[":
				closing = "]"
			case "［":
				closing = "］"
			}
			if closing != "" && strings.HasPrefix(strings.TrimLeftFunc(r.Target[b.end:], unicode.IsSpace), closing) {
				paired[a.form], paired[b.form] = true, true
			}
		}
	}
	if len(paired) == 0 {
		return false
	}
	r.Patches = nil
	r.Status = "kept"
	r.Reason = "괄호로 함께 소개한 번역·음역·별칭과 이후 같은 표기의 단독 사용을 보존했습니다."
	for whole, parts := range wholePairs {
		if paired[parts[0]] && paired[parts[1]] {
			paired[whole] = true
		}
	}
	for _, form := range r.Forms {
		if !paired[form] {
			r.Status = "uncertain"
			r.Reason = "괄호 병기와 연결되지 않은 별도 표기가 있어 확인이 필요합니다."
		}
	}
	return true
}

func validateTermDecision(c *workflows.DocumentTermCase, result llm.StreamResult) error {
	var payload struct {
		Groups []workflows.DocumentTermDecision `json:"groups"`
	}
	if err := decodeTermJSON(result, &payload); err != nil {
		return err
	}
	expected := map[string]workflows.DocumentTermReference{}
	for _, r := range c.References {
		if r.Status == "named" {
			expected[r.ID] = r
		}
	}
	seen := map[string]bool{}
	for _, g := range payload.Groups {
		if g.Status == "correctable" && termDecisionDropsComponent(*c, g) {
			return fmt.Errorf("canonical spelling drops a component of a compound source name; keep a defensible full name or return uncertain, never choose the more frequent abbreviation")
		}
		if len(g.References) == 0 || strings.TrimSpace(g.Reason) == "" || g.Status != "consistent" && g.Status != "correctable" && g.Status != "uncertain" {
			return fmt.Errorf("invalid term group")
		}
		if g.Status != "uncertain" && (strings.TrimSpace(g.Canonical) == "" || utf8.RuneCountInString(g.Canonical) > 96) {
			return fmt.Errorf("invalid canonical name")
		}
		if c.Pinned != "" && g.Status != "uncertain" && g.Canonical != c.Pinned {
			return fmt.Errorf("user-pinned spelling changed")
		}
		observed := false
		for _, id := range g.References {
			for _, form := range expected[id].Forms {
				observed = observed || form == g.Canonical
			}
		}
		if g.Status != "uncertain" && c.Pinned == "" && !observed {
			return fmt.Errorf("canonical spelling has no observed translation evidence")
		}
		for _, id := range g.References {
			r, ok := expected[id]
			if !ok || seen[id] {
				return fmt.Errorf("missing/duplicate/unexpected group reference %s", id)
			}
			seen[id] = true
			if g.Status == "consistent" {
				for _, form := range r.Forms {
					if form != g.Canonical {
						return fmt.Errorf("mixed spellings falsely marked consistent")
					}
				}
			}
		}
	}
	if len(seen) != len(expected) {
		return fmt.Errorf("unchecked named references")
	}
	c.Decisions = payload.Groups
	return nil
}

func termDecisionDropsComponent(c workflows.DocumentTermCase, g workflows.DocumentTermDecision) bool {
	if c.Pinned != "" || len(strings.Fields(c.Source)) < 2 || g.Canonical == "" {
		return false
	}
	canonical := strings.Join(strings.Fields(g.Canonical), " ")
	for _, r := range c.References {
		included := false
		for _, id := range g.References {
			included = included || id == r.ID
		}
		if !included {
			continue
		}
		for _, form := range r.Forms {
			longer := strings.Join(strings.Fields(form), " ")
			if len(strings.Fields(longer)) > len(strings.Fields(canonical)) && strings.Contains(" "+longer+" ", " "+canonical+" ") {
				return true
			}
		}
	}
	return false
}

func termReferenceTarget(units []workflows.DocumentUnit, r workflows.DocumentTermReference) (string, error) {
	if r.Ordinal < 0 || r.Ordinal >= len(units) || units[r.Ordinal].Ordinal != r.Ordinal {
		return "", fmt.Errorf("invalid term checkpoint")
	}
	var text strings.Builder
	for _, id := range r.InputIDs {
		v, ok := units[r.Ordinal].Output[id]
		if !ok {
			return "", fmt.Errorf("missing term output %s", id)
		}
		text.WriteString(v)
	}
	return text.String(), nil
}

func validateTermPreserved(r *workflows.DocumentTermReference, source string) error {
	if len(r.Preserved) > 8 {
		return fmt.Errorf("too many preserved mentions")
	}
	for i := range r.Preserved {
		p := &r.Preserved[i]
		s, sourceOK := documentAnchoredQuote(r.Source, p.Source)
		t, targetOK := documentAnchoredQuote(r.Target, p.Target)
		if !sourceOK || !targetOK || s == "" || t == "" || strings.TrimSpace(p.Reason) == "" || utf8.RuneCountInString(s) > 160 || utf8.RuneCountInString(t) > 160 || strings.Count(r.Source, s) != 1 || strings.Count(r.Target, t) != 1 {
			return fmt.Errorf("preserved mention needs unique exact source/target evidence")
		}
		if strings.Join(strings.Fields(s), " ") == strings.Join(strings.Fields(source), " ") {
			return fmt.Errorf("preserved original spelling needs context distinguishing its referent")
		}
		found := false
		for _, form := range r.Forms {
			found = found || len(termRanges(t, form)) > 0
		}
		if !found {
			return fmt.Errorf("preserved quote does not contain an inspected spelling")
		}
		p.Source, p.Target = s, t
	}
	return nil
}

func termPreservedRanges(text string, preserved []workflows.DocumentTermPreserved) [][2]int {
	var spans [][2]int
	for _, p := range preserved {
		if p.Target != "" && strings.Count(text, p.Target) == 1 {
			i := strings.Index(text, p.Target)
			spans = append(spans, [2]int{i, i + len(p.Target)})
		}
	}
	return spans
}

// Return edits on a copy. Every byte outside model-proposed name/particle spans
// is preserved. A cross-inline-node replacement is left unresolved, not flattened.
func applyTermPatches(units []workflows.DocumentUnit, r workflows.DocumentTermReference, canonical string, patches []workflows.DocumentTermPatch, retained ...string) (map[string]string, string, error) {
	target, err := termReferenceTarget(units, r)
	if err != nil {
		return nil, "", err
	}
	if documentTextHash(target) != r.TargetHash {
		return nil, "", fmt.Errorf("term target changed since inspection")
	}
	type change struct {
		start, end int
		after      string
	}
	changes := []change{}
	preservedRanges := termPreservedRanges(target, r.Preserved)
	if len(preservedRanges) != len(r.Preserved) {
		return nil, "", fmt.Errorf("preserved target changed since inspection")
	}
	for _, p := range patches {
		if p.Before == p.After {
			continue
		}
		afterSuffix, afterOK := termNameSuffix(p.After, canonical)
		allowed := false
		patchForm := ""
		for _, form := range r.Forms {
			if beforeSuffix, ok := termNameSuffix(p.Before, form); ok && afterOK {
				// Exact unchanged trailing context is an anchor, not an edit.
				// Retain it to disambiguate punctuation and repeated names.
				valid := beforeSuffix == afterSuffix ||
					(beforeSuffix == "" || termParticleSuffix(beforeSuffix)) && (afterSuffix == "" || termParticleSuffix(afterSuffix))
				allowed = allowed || valid
				if valid && len(form) > len(patchForm) {
					patchForm = form
				}
			}
		}
		if !allowed || utf8.RuneCountInString(p.Before) > 192 || utf8.RuneCountInString(p.After) > 192 {
			return nil, "", fmt.Errorf("patch exceeds name/particle scope")
		}
		ranges := termRanges(target, p.Before)
		if len(ranges) == 0 {
			return nil, "", fmt.Errorf("patch text not found as a name")
		}
		bare := false
		for _, form := range r.Forms {
			bare = bare || strings.Join(strings.Fields(p.Before), " ") == form
		}
		for _, span := range ranges {
			protected := false
			for _, keep := range preservedRanges {
				if span[0] < keep[1] && span[1] > keep[0] {
					protected = true
				}
			}
			if protected {
				continue
			}
			// A shorter form inside an already complete compound name is not a
			// second mention. In particular, never add the canonical prefix twice.
			contained := false
			for _, own := range termRanges(target, patchForm) {
				if own[0] != span[0] {
					continue
				}
				for _, longer := range append(append([]string(nil), r.Forms...), canonical) {
					if len(longer) <= len(patchForm) {
						continue
					}
					for _, full := range termRanges(target, longer) {
						if own[0] >= full[0] && own[1] <= full[1] {
							contained = true
						}
					}
				}
			}
			if contained {
				continue
			}
			if strings.ContainsAny(target[span[0]:span[1]], "\n\r") {
				for _, in := range units[r.Ordinal].Inputs {
					if in.Whitespace == "pre-wrap" {
						for _, id := range r.InputIDs {
							if in.ID == id {
								return nil, "", fmt.Errorf("name edit crosses an authored line break")
							}
						}
					}
				}
			}
			next, _ := utf8.DecodeRuneInString(target[span[1]:])
			if bare && unicode.Is(unicode.Hangul, next) {
				continue
			}
			changes = append(changes, change{span[0], span[1], p.After})
		}
	}
	sort.Slice(changes, func(i, j int) bool { return changes[i].start > changes[j].start })
	output := maps.Clone(units[r.Ordinal].Output)
	for i, ch := range changes {
		if i > 0 && ch.end > changes[i-1].start {
			return nil, "", fmt.Errorf("overlapping term patches")
		}
		offset := 0
		applied := false
		for _, id := range r.InputIDs {
			original := units[r.Ordinal].Output[id]
			end := offset + len(original)
			if ch.start >= offset && ch.end <= end {
				a, b := ch.start-offset, ch.end-offset
				output[id] = output[id][:a] + ch.after + output[id][b:]
				applied = true
				break
			}
			offset = end
		}
		if !applied {
			return nil, "", fmt.Errorf("name crosses protected inline nodes")
		}
	}
	var joined strings.Builder
	for _, id := range r.InputIDs {
		joined.WriteString(output[id])
	}
	proposed := joined.String()
	preservedRanges = termPreservedRanges(proposed, r.Preserved)
	if len(preservedRanges) != len(r.Preserved) {
		return nil, "", fmt.Errorf("preserved phrase was changed")
	}
	canonicalRanges := termRanges(proposed, canonical)
	if len(canonicalRanges) == 0 {
		return nil, "", fmt.Errorf("canonical name missing after correction")
	}
	for _, form := range r.Forms {
		keep := false
		for _, original := range retained {
			keep = keep || form == original
		}
		if form == canonical || keep {
			continue
		}
		for _, span := range termRanges(proposed, form) {
			covered := false
			for _, good := range canonicalRanges {
				covered = covered || span[0] >= good[0] && span[1] <= good[1]
			}
			for _, keep := range preservedRanges {
				covered = covered || span[0] >= keep[0] && span[1] <= keep[1]
			}
			if !covered {
				return nil, "", fmt.Errorf("unrepaired spelling %q", form)
			}
		}
	}
	return output, proposed, nil
}

func termNameSuffix(text, name string) (string, bool) {
	for _, span := range termRanges(text, name) {
		if span[0] == 0 {
			return text[span[1]:], true
		}
	}
	return "", false
}

// Progress polling carries evidence and counts, not another full copy of the
// book for every name. Immutable sources remain in the task's stored snapshot.
func documentPublicTask(d *workflows.DocumentTask) *workflows.DocumentTask {
	if d == nil || d.Terms == nil {
		return d
	}
	copy := *d
	terms := *d.Terms
	copy.Terms = &terms
	terms.Candidates = nil
	if d.Terms.Language != nil {
		language := *d.Terms.Language
		language.Records = append([]workflows.DocumentLanguageRecord(nil), language.Records...)
		for i := range language.Records {
			language.Records[i].Source = ""
			language.Records[i].Target = ""
			language.Records[i].BeforeOutput = nil
		}
		terms.Language = &language
	}
	terms.Cases = append([]workflows.DocumentTermCase(nil), d.Terms.Cases...)
	for i := range terms.Cases {
		terms.Cases[i].Attempts = nil
		terms.Cases[i].References = append([]workflows.DocumentTermReference(nil), terms.Cases[i].References...)
		for j := range terms.Cases[i].References {
			terms.Cases[i].References[j].Source = ""
			terms.Cases[i].References[j].Target = ""
			terms.Cases[i].References[j].BeforeOutput = nil
		}
	}
	return &copy
}
