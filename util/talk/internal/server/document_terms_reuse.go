package server

import (
	"fmt"
	"strings"

	"sparktalk/internal/workflows"
)

// Reuse only source-bound inspections whose target hash is unchanged or is
// connected to the current text by already verified, paragraph-scoped edits.
// Failed extraction is retried; failed decisions retain valid evidence but are
// decided again. No model output or old inferred glossary becomes a user pin.
func reuseDocumentTerms(parent workflows.Run, previous, units []workflows.DocumentUnit, run *workflows.Run) {
	prior := parent.Document.Terms
	if prior == nil || prior.Status != "complete" || run.Document.RequestSpec.Origin == "user_correction" {
		return
	}
	keyMap := func(us []workflows.DocumentUnit) map[string]string {
		out := map[string]string{}
		for _, u := range us {
			for _, in := range u.Inputs {
				out[in.ID] = fmt.Sprintf("%s:%d", in.Block, in.Part)
			}
		}
		return out
	}
	oldKeys, newKeys := keyMap(previous), keyMap(units)
	key := func(r workflows.DocumentTermReference, keys map[string]string) string {
		parts := []string{}
		for _, id := range r.InputIDs {
			if keys[id] == "" {
				return ""
			}
			parts = append(parts, keys[id])
		}
		return strings.Join(parts, "\x00")
	}
	type edge struct{ before, after string }
	edges := map[string][]edge{}
	for _, c := range prior.Cases {
		for _, r := range c.References {
			if r.Status == "repaired" && r.ResultHash != "" && documentTextHash(r.Target) == r.TargetHash {
				edges[key(r, oldKeys)] = append(edges[key(r, oldKeys)], edge{r.TargetHash, r.ResultHash})
			}
		}
	}
	reaches := func(k, from, to string) bool {
		seen := map[string]bool{from: true}
		for changed := true; changed; {
			if seen[to] {
				return true
			}
			changed = false
			for _, e := range edges[k] {
				if seen[e.before] && !seen[e.after] {
					seen[e.after], changed = true, true
				}
			}
		}
		return seen[to]
	}
	run.Document.Terms.Cases = documentTermCases(*run, units)
	for ci := range run.Document.Terms.Cases {
		c := &run.Document.Terms.Cases[ci]
		var old *workflows.DocumentTermCase
		for i := range prior.Cases {
			if prior.Cases[i].Source == c.Source {
				old = &prior.Cases[i]
				break
			}
		}
		if old == nil || c.Pinned != old.Pinned {
			continue
		}
		oldRefs := map[string]workflows.DocumentTermReference{}
		groups := map[string]workflows.DocumentTermDecision{}
		for _, r := range old.References {
			oldRefs[key(r, oldKeys)] = r
		}
		for _, g := range old.Decisions {
			for _, id := range g.References {
				groups[id] = g
			}
		}
		all := old.Status == "complete" && len(c.References) == len(old.References)
		for ri := range c.References {
			r := &c.References[ri]
			k := key(*r, newKeys)
			x, ok := oldRefs[k]
			if !ok || k == "" || x.SourceHash != r.SourceHash || documentTextHash(x.Source) != r.SourceHash || documentTextHash(x.Target) != x.TargetHash {
				all = false
				continue
			}
			hash := x.TargetHash
			if x.Status == "repaired" {
				hash = x.ResultHash
			}
			if hash == "" || !reaches(k, hash, r.TargetHash) {
				all = false
				continue
			}
			r.Reason = x.Reason
			quote, anchored := documentAnchoredQuote(r.Source, x.Evidence)
			if !anchored || !termSourceMention(quote, c.Source) {
				all = false
				continue
			}
			forms := append([]string(nil), x.Forms...)
			g := groups[x.ID]
			if x.Status == "repaired" {
				if g.Status != "correctable" || g.Canonical == "" || termDecisionDropsComponent(*old, g) {
					all = false
					continue
				}
				forms = []string{g.Canonical}
				if c.Pinned == "" && c.Source != g.Canonical && len(termRanges(r.Target, c.Source)) > 0 {
					forms = append(forms, c.Source)
				}
			}
			valid := true
			for _, form := range forms {
				valid = valid && len(termRanges(r.Target, form)) > 0
			}
			if !valid || (len(forms) == 0 && x.Status != "ordinary" && x.Status != "implicit") {
				all = false
				continue
			}
			r.Forms, r.Evidence, r.Referent = forms, quote, x.Referent
			r.Preserved = append([]workflows.DocumentTermPreserved(nil), x.Preserved...)
			r.Status = "named"
			if x.Status == "ordinary" || x.Status == "implicit" {
				r.Status = x.Status
			}
			if r.Status == "named" && r.Referent == "" {
				r.Status = "pending"
				all = false
				continue
			}
			preserveTermPair(r)
			if x.Status == "uncertain" || r.Status == "uncertain" || g.Status == "uncertain" || termDecisionDropsComponent(*old, g) {
				all = false
			}
			run.Document.Terms.Reused++
		}
		if all {
			c.Status = "complete"
			for i := range c.References {
				if c.References[i].Status == "named" {
					c.References[i].Status = "kept"
				}
			}
			// Reference ids may differ in a differently planned revision; remap
			// every decision rather than inheriting positional ids blindly.
			for _, oldGroup := range old.Decisions {
				g := oldGroup
				g.References = nil
				for _, r := range c.References {
					x := oldRefs[key(r, newKeys)]
					for _, id := range oldGroup.References {
						if x.ID == id {
							g.References = append(g.References, r.ID)
						}
					}
				}
				c.Decisions = append(c.Decisions, g)
			}
		}
	}
	termCounters(run)
}
