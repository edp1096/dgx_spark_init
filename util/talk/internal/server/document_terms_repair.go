package server

import (
	"context"
	"encoding/json"
	"fmt"
	"maps"
	"strings"
	"time"

	"sparktalk/internal/workflows"
)

type termCheckpoint func(*workflows.DocumentRequestAttempt, ...workflows.DocumentUnit) error

func (s *Server) repairTermGroup(ctx context.Context, run *workflows.Run, units []workflows.DocumentUnit, c *workflows.DocumentTermCase, g workflows.DocumentTermDecision, started time.Time, elapsed int, save termCheckpoint) error {
	retained := []string{}
	if c.Pinned == "" {
		retained = append(retained, c.Source)
	}
	members := map[string]*workflows.DocumentTermReference{}
	for _, id := range g.References {
		var r *workflows.DocumentTermReference
		for i := range c.References {
			if c.References[i].ID == id {
				r = &c.References[i]
				break
			}
		}
		if r == nil {
			return fmt.Errorf("missing stored term reference")
		}
		members[id] = r
		if r.Status == "repaired" || r.Status == "kept" || r.Status == "uncertain" {
			continue
		}
		if preserveTermPair(r) {
			continue
		}
		if g.Status == "uncertain" {
			r.Status = "uncertain"
			r.Reason = g.Reason
			continue
		}
		needs := false
		for _, f := range r.Forms {
			needs = needs || (f != g.Canonical && (f != c.Source || c.Pinned != ""))
		}
		if !needs {
			r.Status = "kept"
			continue
		}
		if !run.Document.Terms.Repair {
			r.Status = "uncertain"
			r.Reason = "표기 충돌 확인: " + g.Reason
		}
	}
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		pending, proposals := []workflows.DocumentTermReference{}, []workflows.DocumentTermReference{}
		for _, id := range g.References {
			r := members[id]
			if r.Status == "repaired" || r.Status == "kept" || r.Status == "uncertain" {
				continue
			}
			if len(r.Patches) > 0 || len(r.Preserved) > 0 {
				proposals = append(proposals, *r)
			} else if c.Attempts["repair:"+id] < 2 {
				pending = append(pending, *r)
			} else {
				r.Status = "uncertain"
			}
		}
		if len(proposals) == 0 && len(pending) == 0 {
			return nil
		}
		if len(proposals) == 0 {
			refs := termBatch(pending[:min(4, len(pending))])
			result, attempt, err := s.termModelCall(ctx, run, units, refs, "terms_repair", patchTermRequest(c.Source, g.Canonical, refs), started, elapsed)
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
			var payload struct {
				Patches   map[string][]workflows.DocumentTermPatch     `json:"patches"`
				Preserved map[string][]workflows.DocumentTermPreserved `json:"preserved"`
			}
			decodeErr := decodeTermJSON(result, &payload)
			allowed := map[string]bool{}
			for _, r := range refs {
				allowed[r.ID] = true
			}
			for id := range payload.Patches {
				if !allowed[id] {
					decodeErr = fmt.Errorf("unexpected patch reference")
				}
			}
			for id := range payload.Preserved {
				if !allowed[id] {
					decodeErr = fmt.Errorf("unexpected preserved reference")
				}
			}
			for _, ref := range refs {
				r := members[ref.ID]
				c.Attempts["repair:"+r.ID]++
				err := decodeErr
				if err == nil {
					p, ok := payload.Patches[r.ID]
					if !ok {
						err = fmt.Errorf("missing patch reference")
					} else {
						r.Patches = p
						r.Preserved = payload.Preserved[r.ID]
						err = validateTermPreserved(r, c.Source)
						if err == nil {
							_, _, err = applyTermPatches(units, *r, g.Canonical, p, retained...)
						}
					}
				}
				if err != nil {
					r.Patches = nil
					r.Preserved = nil
					r.Reason = err.Error()
					attempt.FailureKind = "invalid_term_patch"
				}
			}
			if err = save(attempt); err != nil {
				return err
			}
			continue
		}
		refs := termBatch(proposals[:min(4, len(proposals))])
		proposed := map[string]string{}
		outputs := map[string]map[string]string{}
		validRefs := []workflows.DocumentTermReference{}
		for _, ref := range refs {
			r := members[ref.ID]
			out, target, err := applyTermPatches(units, *r, g.Canonical, r.Patches, retained...)
			if err != nil {
				r.Patches = nil
				r.Preserved = nil
				r.Reason = err.Error()
				continue
			}
			proposed[r.ID] = target
			outputs[r.ID] = out
			validRefs = append(validRefs, *r)
		}
		if len(validRefs) == 0 {
			continue
		}
		result, attempt, err := s.termModelCall(ctx, run, units, validRefs, "terms_verify", verifyTermRequest(c.Source, validRefs, proposed), started, elapsed)
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
		var payload struct {
			Checks map[string]struct {
				Valid  bool   `json:"valid"`
				Reason string `json:"reason"`
			} `json:"checks"`
		}
		decodeErr := decodeTermJSON(result, &payload)
		for id := range payload.Checks {
			if _, ok := proposed[id]; !ok {
				decodeErr = fmt.Errorf("unexpected verification reference")
			}
		}
		originals := map[int]map[string]string{}
		oldRefs := map[string]workflows.DocumentTermReference{}
		for _, ref := range validRefs {
			r := members[ref.ID]
			check, ok := payload.Checks[r.ID]
			err := decodeErr
			if err == nil && (!ok || !check.Valid || strings.TrimSpace(check.Reason) == "") {
				err = fmt.Errorf("교정 확인 보류: %s", check.Reason)
			}
			if err != nil {
				r.Patches = nil
				r.Preserved = nil
				r.Reason = err.Error()
				attempt.FailureKind = "term_verification_rejected"
				continue
			}
			oldRefs[r.ID] = *r
			if originals[r.Ordinal] == nil {
				originals[r.Ordinal] = maps.Clone(units[r.Ordinal].Output)
			}
			r.BeforeOutput = map[string]string{}
			for _, id := range r.InputIDs {
				r.BeforeOutput[id] = units[r.Ordinal].Output[id]
				units[r.Ordinal].Output[id] = outputs[r.ID][id]
			}
			r.Status = "repaired"
			r.ResultHash = documentTextHash(proposed[r.ID])
			if r.ResultHash == r.TargetHash {
				r.Status = "kept"
				r.Patches = nil
			}
			r.Reason = check.Reason
		}
		changed := []workflows.DocumentUnit{}
		for _, u := range units {
			if originals[u.Ordinal] != nil {
				changed = append(changed, u)
			}
		}
		if err = save(attempt, changed...); err != nil {
			for ordinal, output := range originals {
				units[ordinal].Output = output
			}
			for id, ref := range oldRefs {
				*members[id] = ref
			}
			termCounters(run)
			return err
		}
	}
}

// Revalidate retained decisions on resume. A stronger guard may revoke a
// previously accepted correction, but only the exact saved result may be
// restored; later edits must never be overwritten.
func (s *Server) revokeUnsafeTermDecisions(run *workflows.Run, units []workflows.DocumentUnit) error {
	raw, err := json.Marshal(run.Document.Terms)
	if err != nil {
		return err
	}
	var terms workflows.DocumentTerms
	if err = json.Unmarshal(raw, &terms); err != nil {
		return err
	}
	copyUnits := append([]workflows.DocumentUnit(nil), units...)
	changed := map[int]bool{}
	revoked := false
	const reason = "복합 이름의 의미 요소를 빼는 축약형으로 자동 통일하지 않습니다. 해당 교정은 보류하고 수정 전 표기를 보존했습니다."
	for ci := range terms.Cases {
		c := &terms.Cases[ci]
		for gi := range c.Decisions {
			g := &c.Decisions[gi]
			if g.Status != "correctable" || !termDecisionDropsComponent(*c, *g) {
				continue
			}
			for ri := range c.References {
				r := &c.References[ri]
				included := false
				for _, id := range g.References {
					included = included || id == r.ID
				}
				if !included {
					continue
				}
				if r.Status == "repaired" {
					current, e := termReferenceTarget(copyUnits, *r)
					if e != nil || documentTextHash(current) != r.ResultHash {
						return fmt.Errorf("교정 결과가 이후에 변경되어 자동 원복을 보류했습니다")
					}
					before := r.BeforeOutput
					if len(before) == 0 && len(r.InputIDs) == 1 {
						before = map[string]string{r.InputIDs[0]: r.Target}
					}
					var joined strings.Builder
					for _, id := range r.InputIDs {
						value, ok := before[id]
						if !ok {
							return fmt.Errorf("수정 전 인라인 구간이 없어 자동 원복을 보류했습니다")
						}
						joined.WriteString(value)
					}
					if documentTextHash(joined.String()) != r.TargetHash {
						return fmt.Errorf("수정 전 문단 해시가 일치하지 않습니다")
					}
					if !changed[r.Ordinal] {
						copyUnits[r.Ordinal].Output = maps.Clone(copyUnits[r.Ordinal].Output)
					}
					for _, id := range r.InputIDs {
						copyUnits[r.Ordinal].Output[id] = before[id]
					}
					changed[r.Ordinal] = true
					r.ResultHash = ""
				}
				if r.Status != "kept" {
					r.Status, r.Reason, r.Patches = "uncertain", reason, nil
				}
			}
			g.Status, g.Canonical, g.Reason = "uncertain", "", reason
			c.Status, c.Reason = "unresolved", reason
			revoked = true
		}
	}
	if !revoked {
		return nil
	}
	candidate := *run
	document := *run.Document
	candidate.Document = &document
	document.Terms = &terms
	terms.Status = "running"
	termCounters(&candidate)
	updated := []workflows.DocumentUnit{}
	for i := range copyUnits {
		if changed[i] {
			updated = append(updated, copyUnits[i])
		}
	}
	if err = s.db.SaveDocumentProgress(&candidate, updated...); err != nil {
		return err
	}
	*run = candidate
	copy(units, copyUnits)
	return nil
}
