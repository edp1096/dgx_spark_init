package server

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"sparktalk/internal/workflows"
)

func TestTermEvidenceOptionsAreExactAcrossScriptsAndWraps(t *testing.T) {
	for _, tc := range [][2]string{{"“The Silver\nFox returned,” she said. Silver Fox waved.", "Silver Fox"}, {"トックは言った。彼はトックだった。", "トック"}, {"华盛顿来了。", "华盛顿"}, {"It was Ann, not Anna, who called Ann.", "Ann"}} {
		options := termEvidenceOptions(tc[0], tc[1])
		if len(options) == 0 {
			t.Fatal(tc)
		}
		for _, quote := range options {
			if !strings.Contains(tc[0], quote) || !termSourceMention(quote, tc[1]) {
				t.Fatal("invented quote", tc, quote)
			}
		}
	}
}

func TestTermFailedCopyOptionsExcludeInventedForms(t *testing.T) {
	text := "페린단이 모리만이 쓴 '은빛 여우의 노래'를 읽었다."
	options := termObservedFormOptions(text)
	seen := map[string]bool{}
	for _, option := range options {
		seen[option] = true
		if len(termRanges(text, option)) == 0 {
			t.Fatal("not in target", option)
		}
	}
	for _, want := range []string{"페린단", "모리", "은빛 여우"} {
		if !seen[want] {
			t.Fatal("missing actual form", want)
		}
	}
	for _, bad := range []string{"페린", "모리단", "없는이름"} {
		if seen[bad] {
			t.Fatal("invented/partial form", bad)
		}
	}
}

func TestTermSharedWordInDifferentReferentIsPreserved(t *testing.T) {
	for _, title := range []string{"여우의 노래", "여우가"} {
		before := "여우가 '" + title + "'를 불렀다."
		r := workflows.DocumentTermReference{Ordinal: 0, InputIDs: []string{"a"}, Source: "The Silver Fox sang 'Fox Song'.", Target: before, TargetHash: documentTextHash(before), Forms: []string{"여우"}}
		units := []workflows.DocumentUnit{{Ordinal: 0, Output: map[string]string{"a": before}}}
		patches := []workflows.DocumentTermPatch{{Before: "여우가", After: "은빛 여우가"}}
		if _, _, err := applyTermPatches(units, r, "은빛 여우", patches); err == nil && title == "여우의 노래" {
			t.Fatal("unexplained leftover spelling accepted")
		}
		r.Preserved = []workflows.DocumentTermPreserved{{Source: "'Fox Song'", Target: "'" + title + "'", Reason: "별도의 노래 제목"}}
		if err := validateTermPreserved(&r, "Silver Fox"); err != nil {
			t.Fatal(err)
		}
		_, text, err := applyTermPatches(units, r, "은빛 여우", patches)
		if err != nil || text != "은빛 여우가 '"+title+"'를 불렀다." {
			t.Fatal("title changed or person missed", text, err)
		}
		for _, bad := range []workflows.DocumentTermPreserved{{Source: "Silver Fox", Target: "여우가", Reason: "skip"}, {Source: "not in original", Target: "'" + title + "'", Reason: "skip"}, {Source: "'Fox Song'", Target: "여우", Reason: "ambiguous"}} {
			r.Preserved = []workflows.DocumentTermPreserved{bad}
			if validateTermPreserved(&r, "Silver Fox") == nil {
				t.Fatal("unanchored/ambiguous exemption", bad)
			}
		}
	}
	// Identical source spelling can still have different meanings. The
	// contextual quotation is passed to independent verification, not silently
	// accepted as proof of a distinct referent.
	r := workflows.DocumentTermReference{Source: "May arrived in May.", Target: "메이가 오월에 왔다.", Forms: []string{"메이", "오월"}, Preserved: []workflows.DocumentTermPreserved{{Source: "in May", Target: "오월에", Reason: "month"}}}
	if err := validateTermPreserved(&r, "May"); err != nil {
		t.Fatal(err)
	}
}

func TestTermReuseRequiresSourceAndVerifiedTargetLineage(t *testing.T) {
	before, after, final := "페린단이 메이를 보았다.", "페린이 메이를 보았다.", "페린이 메이를 만났다."
	makeUnits := func(prefix, text string) []workflows.DocumentUnit {
		return []workflows.DocumentUnit{{Ordinal: 0, Inputs: []workflows.DocumentInput{{ID: prefix + ":b1:0", Block: "b1", Text: "Perrin saw May."}}, Output: map[string]string{prefix + ":b1:0": text}}}
	}
	for _, tc := range []struct {
		name, text         string
		changedSource, pin bool
		reused             int
	}{
		{"verified edit", after, false, false, 1},
		{"unrecorded later prose edit", final, false, false, 0},
		{"source changed", after, true, false, 0},
		{"user pin changed", after, false, true, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			oldUnits, units := makeUnits("old", after), makeUnits("new", tc.text)
			ref := workflows.DocumentTermReference{ID: "r1", Ordinal: 0, InputIDs: []string{"old:b1:0"}, Source: "Perrin saw May.", SourceHash: documentTextHash("Perrin saw May."), Target: before, TargetHash: documentTextHash(before), ResultHash: documentTextHash(after), Status: "repaired", Evidence: "Perrin saw May.", Referent: "person", Forms: []string{"페린단"}}
			parent := workflows.Run{Document: &workflows.DocumentTask{Terms: &workflows.DocumentTerms{Status: "complete", Cases: []workflows.DocumentTermCase{{Source: "Perrin", Status: "complete", References: []workflows.DocumentTermReference{ref}, Decisions: []workflows.DocumentTermDecision{{References: []string{"r1"}, Canonical: "페린", Status: "correctable"}}}}}}}
			run := workflows.Run{Document: &workflows.DocumentTask{RequestSpec: &workflows.DocumentRequestSpec{Origin: "api"}, Terms: &workflows.DocumentTerms{Candidates: []workflows.DocumentEntity{{Source: "Perrin"}}}}}
			if tc.changedSource {
				units[0].Inputs[0].Text = "Perrin greeted May."
			}
			if tc.pin {
				run.Document.UserGlossary = map[string]string{"Perrin": "페랭"}
			}
			reuseDocumentTerms(parent, oldUnits, units, &run)
			if run.Document.Terms.Reused != tc.reused {
				t.Fatalf("reused %d want %d", run.Document.Terms.Reused, tc.reused)
			}
			if tc.reused > 0 {
				r := run.Document.Terms.Cases[0].References[0]
				if r.InputIDs[0] != "new:b1:0" || r.Status != "kept" || r.Forms[0] != "페린" || r.ResultHash != "" || run.Document.Terms.Changed != 0 {
					t.Fatal(r)
				}
			}
			if parent.Document.Terms.Cases[0].References[0].Forms[0] != "페린단" {
				t.Fatal("parent mutated")
			}
		})
	}
}

func TestTermReuseFollowsOtherVerifiedNameChanges(t *testing.T) {
	first, middle, last := "페린단과 메이단.", "페린과 메이단.", "페린과 메이."
	source := "Perrin and May."
	units := []workflows.DocumentUnit{{Ordinal: 0, Inputs: []workflows.DocumentInput{{ID: "old:b1:0", Block: "b1", Text: source}}, Output: map[string]string{"old:b1:0": last}}}
	ref := func(before, after, form string) workflows.DocumentTermReference {
		return workflows.DocumentTermReference{ID: "r1", Ordinal: 0, InputIDs: []string{"old:b1:0"}, Source: source, SourceHash: documentTextHash(source), Target: before, TargetHash: documentTextHash(before), ResultHash: documentTextHash(after), Status: "repaired", Forms: []string{form}, Evidence: source, Referent: "person"}
	}
	parent := workflows.Run{Document: &workflows.DocumentTask{Terms: &workflows.DocumentTerms{Status: "complete", Cases: []workflows.DocumentTermCase{
		{Source: "Perrin", Status: "complete", References: []workflows.DocumentTermReference{ref(first, middle, "페린단")}, Decisions: []workflows.DocumentTermDecision{{References: []string{"r1"}, Canonical: "페린", Status: "correctable"}}},
		{Source: "May", Status: "complete", References: []workflows.DocumentTermReference{ref(middle, last, "메이단")}, Decisions: []workflows.DocumentTermDecision{{References: []string{"r1"}, Canonical: "메이", Status: "correctable"}}},
	}}}}
	run := workflows.Run{Document: &workflows.DocumentTask{RequestSpec: &workflows.DocumentRequestSpec{Origin: "api"}, Terms: &workflows.DocumentTerms{Candidates: []workflows.DocumentEntity{{Source: "Perrin"}, {Source: "May"}}}}}
	reuseDocumentTerms(parent, units, units, &run)
	if run.Document.Terms.Reused != 2 || run.Document.Terms.Checked != 2 {
		t.Fatal(run.Document.Terms)
	}
	// Losing the intervening verified edge invalidates only the affected
	// earlier checkpoint, even though its source and name still match.
	parent.Document.Terms.Cases[1].References[0].TargetHash = "invalid"
	run.Document.Terms.Reused = 0
	reuseDocumentTerms(parent, units, units, &run)
	if run.Document.Terms.Reused != 0 {
		t.Fatal("unverified edge reused")
	}
}

// Optional replay of a saved completed run; never opens a production database
// or sends inference. Useful for checking how much evidence a revision reuses.
func TestDocumentTermsReuseReplay(t *testing.T) {
	dir := os.Getenv("SPARKTALK_TERMS_REUSE_REPLAY")
	if dir == "" {
		t.Skip("explicit saved checkpoint replay")
	}
	var parent workflows.Run
	var units []workflows.DocumentUnit
	for _, item := range []struct {
		name  string
		value any
	}{{"run.json", &parent}, {"units.json", &units}} {
		raw, err := os.ReadFile(filepath.Join(dir, item.name))
		if err != nil {
			t.Fatal(err)
		}
		if err = json.Unmarshal(raw, item.value); err != nil {
			t.Fatal(err)
		}
	}
	d := *parent.Document
	d.RequestSpec = &workflows.DocumentRequestSpec{Origin: "api"}
	d.Terms = &workflows.DocumentTerms{Status: "pending", Repair: true, Candidates: parent.Document.Terms.Candidates}
	run := workflows.Run{Document: &d}
	reuseDocumentTerms(parent, units, units, &run)
	pending, named := 0, 0
	for _, c := range d.Terms.Cases {
		p, n := 0, 0
		for _, r := range c.References {
			if r.Status == "pending" {
				p++
			}
			if r.Status == "named" {
				n++
			}
		}
		pending += p
		named += n
		if c.Status != "complete" {
			t.Log(c.Source, c.Status, "pending", p, "named", n)
		}
	}
	t.Log("reused", d.Terms.Reused, "pending", pending, "named", named, "complete", d.Terms.Checked)
	raw, _ := json.MarshalIndent(run, "", "  ")
	if err := os.WriteFile(filepath.Join(dir, "reuse-preview.json"), raw, 0600); err != nil {
		t.Fatal(err)
	}
}
