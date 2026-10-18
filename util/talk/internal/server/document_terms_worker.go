package server

import (
	"context"
	"fmt"
	"strings"
	"time"

	"sparktalk/internal/llm"
	"sparktalk/internal/workflows"
)

func termRequestUnit(units []workflows.DocumentUnit, refs []workflows.DocumentTermReference) workflows.DocumentUnit {
	u := workflows.DocumentUnit{Output: map[string]string{}}
	seen := map[string]bool{}
	for _, r := range refs {
		for _, id := range r.InputIDs {
			if seen[id] {
				continue
			}
			seen[id] = true
			for _, in := range units[r.Ordinal].Inputs {
				if in.ID == id {
					u.Inputs = append(u.Inputs, in)
					u.Output[id] = units[r.Ordinal].Output[id]
				}
			}
		}
	}
	return u
}

func termBatch(refs []workflows.DocumentTermReference) []workflows.DocumentTermReference {
	n, chars := 0, 0
	for n < len(refs) && n < 8 {
		size := estimateTextTokens(refs[n].Source) + estimateTextTokens(refs[n].Target)
		if n > 0 && chars+size > 6000 {
			break
		}
		chars += size
		n++
	}
	return refs[:n]
}

func (s *Server) termModelCall(ctx context.Context, run *workflows.Run, units []workflows.DocumentUnit, refs []workflows.DocumentTermReference, stage string, request documentStageRequest, started time.Time, elapsed int) (llm.StreamResult, *workflows.DocumentRequestAttempt, error) {
	if run.Document.Calls >= run.Document.MaxCalls || run.Document.OutputTokens >= run.Document.MaxOutputTokens || elapsed+int(time.Since(started).Seconds()) >= run.Document.MaxSeconds {
		return llm.StreamResult{}, nil, fmt.Errorf("task execution budget reached")
	}
	p := run.Document.Profiles.Review
	requestRun := documentProfileRun(*run, p)
	run.Document.Phase = stage
	run.Document.ActiveUnits = nil
	seen := map[int]bool{}
	for _, r := range refs {
		if !seen[r.Ordinal] {
			run.Document.ActiveUnits = append(run.Document.ActiveUnits, r.Ordinal+1)
			seen[r.Ordinal] = true
		}
	}
	run.Document.ReasoningBudgetTokens = p.ReasoningBudget
	result, attempt, err := s.requestReviewedStage(ctx, run, requestRun, termRequestUnit(units, refs), p, stage, started, elapsed, request)
	if attempt != nil {
		attempt.ExpectedUnitOutputHashes = map[int]string{}
		for _, r := range refs {
			attempt.ExpectedUnitOutputHashes[r.Ordinal] = documentUnitOutputHash(units[r.Ordinal])
		}
	}
	return result, attempt, err
}

// Convert legacy translation accounting into immutable records once before the
// final consistency phase. Shared batch children never multiply token counts.
func (s *Server) recordTermTranslationAttempts(run *workflows.Run, units []workflows.DocumentUnit) error {
	if run.Document.Terms.Status != "pending" {
		return nil
	}
	for _, u := range units {
		for i, a := range u.Attempts {
			if a.SharedWithOrdinal != nil {
				continue
			}
			attempt := &workflows.DocumentRequestAttempt{ID: fmt.Sprintf("translation-%d-%d", u.Ordinal, i), Stage: "translate", Measurement: a, Units: []int{u.Ordinal + 1}}
			if err := s.db.SaveDocumentReviewProgress(run, attempt, nil, nil); err != nil {
				return err
			}
		}
	}
	run.Document.Terms.Status = "running"
	return s.db.SaveDocumentProgress(run)
}

func termCounters(run *workflows.Run) {
	d := run.Document.Terms
	d.Checked = 0
	d.Changed = 0
	d.Unresolved = 0
	changed := map[string]bool{}
	for _, c := range d.Cases {
		if c.Status == "complete" || c.Status == "unresolved" {
			d.Checked++
		}
		for _, r := range c.References {
			if r.Status == "uncertain" {
				d.Unresolved++
			}
			if r.ResultHash != "" && r.ResultHash != r.TargetHash {
				changed[strings.Join(r.InputIDs, "\x00")] = true
			}
		}
	}
	if d.Language != nil {
		d.Language.Checked = 0
		for _, r := range d.Language.Records {
			if r.Status == "kept" || r.Status == "repaired" || r.Status == "uncertain" {
				d.Language.Checked++
			}
			if r.Status == "uncertain" {
				d.Unresolved++
			}
			if r.ResultHash != "" && r.ResultHash != r.TargetHash {
				changed[strings.Join(r.InputIDs, "\x00")] = true
			}
		}
	}
	d.Changed = len(changed)
}

func (s *Server) processDocumentTerms(ctx context.Context, run *workflows.Run, units []workflows.DocumentUnit, started time.Time, elapsed int) error {
	d := run.Document.Terms
	if d == nil {
		return fmt.Errorf("missing terms task state")
	}
	if err := s.revokeUnsafeTermDecisions(run, units); err != nil {
		return err
	}
	d = run.Document.Terms
	if d.Status == "complete" {
		return nil
	}
	if err := s.recordTermTranslationAttempts(run, units); err != nil {
		return err
	}
	if d.Cases == nil {
		d.Cases = documentTermCases(*run, units)
	}
	save := func(attempt *workflows.DocumentRequestAttempt, changed ...workflows.DocumentUnit) error {
		run.Document.ElapsedSeconds = elapsed + int(time.Since(started).Seconds())
		termCounters(run)
		if attempt != nil {
			return s.db.SaveDocumentReviewProgress(run, attempt, nil, d.Candidates, changed...)
		}
		return s.db.SaveDocumentProgress(run, changed...)
	}
	for ci := range d.Cases {
		c := &d.Cases[ci]
		if c.Status == "complete" || c.Status == "unresolved" {
			continue
		}
		if c.Attempts == nil {
			c.Attempts = map[string]int{}
		}
		for {
			if err := ctx.Err(); err != nil {
				return err
			}
			pending := []workflows.DocumentTermReference{}
			for i := range c.References {
				r := &c.References[i]
				if r.Status == "pending" {
					target, err := termReferenceTarget(units, *r)
					if err != nil {
						return err
					}
					r.Target = target
					r.TargetHash = documentTextHash(target)
					pending = append(pending, *r)
				}
			}
			if len(pending) == 0 {
				break
			}
			refs := termBatch(pending)
			// Isolate a failed paragraph and send its validation error back. A
			// retry must not repeat an unchanged batch or hold good siblings up.
			for _, r := range pending {
				if c.Attempts["extract:"+r.ID] > 0 || r.Reason != "" {
					refs = []workflows.DocumentTermReference{r}
					break
				}
			}
			result, attempt, err := s.termModelCall(ctx, run, units, refs, "terms_extract", extractTermRequest(c.Source, refs), started, elapsed)
			if attempt == nil {
				return err
			}
			if err != nil {
				attempt.FailureKind = "transport"
				if e := save(attempt); e != nil {
					return e
				}
				return err
			}
			invalid := validateTermExtraction(c, refs, result)
			for i := range c.References {
				r := &c.References[i]
				if e := invalid[r.ID]; e != nil {
					c.Attempts["extract:"+r.ID]++
					r.Reason = e.Error()
					if c.Attempts["extract:"+r.ID] >= 2 {
						r.Status = "uncertain"
					}
				}
			}
			if len(invalid) > 0 {
				attempt.FailureKind = "invalid_term_evidence"
			}
			if err = save(attempt); err != nil {
				return err
			}
		}
		named := []workflows.DocumentTermReference{}
		forms := map[string]bool{}
		for _, r := range c.References {
			if r.Status == "named" {
				named = append(named, r)
				for _, f := range r.Forms {
					if f != c.Source {
						forms[f] = true
					}
				}
			}
		}
		if len(c.Decisions) == 0 && len(named) > 0 {
			if len(forms) <= 1 && (c.Pinned == "" || forms[c.Pinned]) {
				ids := []string{}
				for _, r := range named {
					ids = append(ids, r.ID)
				}
				canonical := c.Source
				if len(forms) > 0 {
					canonical = sortedTermKeys(forms)[0]
				}
				c.Decisions = []workflows.DocumentTermDecision{{References: ids, Canonical: canonical, Status: "consistent", Reason: "검사한 위치에서 표기 차이가 발견되지 않았습니다."}}
			} else {
				for len(c.Decisions) == 0 && c.Attempts["decide"] < 2 {
					result, attempt, err := s.termModelCall(ctx, run, units, named, "terms_decide", decideTermRequest(*c), started, elapsed)
					if attempt == nil {
						return err
					}
					if err != nil {
						attempt.FailureKind = "transport"
						if e := save(attempt); e != nil {
							return e
						}
						return err
					}
					c.Attempts["decide"]++
					if e := validateTermDecision(c, result); e != nil {
						c.Reason = e.Error()
						attempt.FailureKind = "invalid_term_decision"
					}
					if e := save(attempt); e != nil {
						return e
					}
				}
				if len(c.Decisions) == 0 {
					for i := range c.References {
						if c.References[i].Status == "named" {
							c.References[i].Status = "uncertain"
							c.References[i].Reason = c.Reason
						}
					}
				}
			}
		}
		for _, g := range c.Decisions {
			if err := s.repairTermGroup(ctx, run, units, c, g, started, elapsed, save); err != nil {
				return err
			}
		}
		c.Status = "complete"
		for _, r := range c.References {
			if r.Status == "uncertain" {
				c.Status = "unresolved"
			}
		}
		if err := save(nil); err != nil {
			return err
		}
	}
	if err := s.processDocumentLanguage(ctx, run, units, started, elapsed, save); err != nil {
		return err
	}
	d.Status = "complete"
	for _, r := range d.Language.Records {
		if r.Status == "uncertain" {
			run.Document.Warnings = append(run.Document.Warnings, "translation language requires review: "+r.ID)
		}
	}
	for _, c := range d.Cases {
		if c.Status == "unresolved" {
			run.Document.Warnings = append(run.Document.Warnings, "term consistency unresolved: "+c.Source)
		}
	}
	return save(nil)
}
