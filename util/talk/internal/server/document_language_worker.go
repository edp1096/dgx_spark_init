package server

import (
	"context"
	"fmt"
	"maps"
	"strings"
	"time"

	"sparktalk/internal/workflows"
)

func (s *Server) processDocumentLanguage(ctx context.Context, run *workflows.Run, units []workflows.DocumentUnit, started time.Time, elapsed int, save termCheckpoint) error {
	if run.Document.Terms.Language == nil {
		run.Document.Terms.Language = documentLanguageScan(units)
	}
	scan := run.Document.Terms.Language
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		var pending []workflows.DocumentLanguageRecord
		proposal := -1
		for i, r := range scan.Records {
			if r.Status == "proposed" {
				proposal = i
				break
			}
			if r.Status == "pending" {
				pending = append(pending, r)
			}
		}
		if proposal < 0 && len(pending) == 0 {
			return save(nil)
		}
		if proposal < 0 {
			var refs []workflows.DocumentTermReference
			for _, r := range pending {
				refs = append(refs, r.DocumentTermReference)
			}
			// Keep the structured decision batch small: long reasoning across eight
			// paragraphs can end without the required JSON response.
			refs = termBatch(refs[:min(4, len(refs))])
			for _, r := range pending {
				if r.Attempts > 0 {
					pending = []workflows.DocumentLanguageRecord{r}
					refs = []workflows.DocumentTermReference{r.DocumentTermReference}
					break
				}
			}
			batch := pending[:len(refs)]
			result, attempt, err := s.termModelCall(ctx, run, units, refs, "language_check", inspectDocumentLanguageRequest(batch), started, elapsed)
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
				Checks map[string]map[string]workflows.DocumentLanguageDecision `json:"checks"`
			}
			decodeErr := decodeTermJSON(result, &payload)
			expected := map[string]bool{}
			for _, r := range batch {
				expected[r.ID] = true
			}
			for id := range payload.Checks {
				if !expected[id] {
					decodeErr = fmt.Errorf("unexpected language reference %s", id)
				}
			}
			for i := range scan.Records {
				r := &scan.Records[i]
				if !expected[r.ID] {
					continue
				}
				r.Attempts++
				err := decodeErr
				if err == nil {
					r.Checks = payload.Checks[r.ID]
					_, _, err = applyDocumentLanguage(units, *r)
				}
				if err != nil {
					r.Checks = nil
					r.Reason = err.Error()
					attempt.FailureKind = "invalid_language_evidence"
					if r.Attempts >= 2 {
						r.Status = "uncertain"
					}
					continue
				}
				r.Status = "kept"
				needs, uncertain := false, false
				for _, c := range r.Checks {
					needs = needs || c.Status == "translate"
					uncertain = uncertain || c.Status == "uncertain"
				}
				if needs {
					r.Status = "proposed"
				}
				if uncertain || needs && !run.Document.Terms.Repair {
					r.Status = "uncertain"
					r.Reason = "번역 언어 확인이 필요한 표현이 있습니다."
				}
			}
			if err = save(attempt); err != nil {
				return err
			}
			continue
		}
		r := &scan.Records[proposal]
		out, target, err := applyDocumentLanguage(units, *r)
		if err != nil {
			r.Status = "uncertain"
			r.Reason = err.Error()
			if e := save(nil); e != nil {
				return e
			}
			continue
		}
		result, attempt, err := s.termModelCall(ctx, run, units, []workflows.DocumentTermReference{r.DocumentTermReference}, "language_verify", verifyDocumentLanguageRequest(*r, target), started, elapsed)
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
		var check struct {
			Valid  bool   `json:"valid"`
			Reason string `json:"reason"`
		}
		err = decodeTermJSON(result, &check)
		if err == nil && (!check.Valid || strings.TrimSpace(check.Reason) == "") {
			err = fmt.Errorf("번역 어휘 교정 검증 실패: %s", check.Reason)
		}
		if err != nil {
			r.Reason = err.Error()
			r.Checks = nil
			r.Status = "pending"
			if r.Attempts >= 2 {
				r.Status = "uncertain"
			}
			attempt.FailureKind = "language_verification_rejected"
			if e := save(attempt); e != nil {
				return e
			}
			continue
		}
		old := *r
		original := maps.Clone(units[r.Ordinal].Output)
		r.BeforeOutput = map[string]string{}
		for _, id := range r.InputIDs {
			r.BeforeOutput[id] = original[id]
			units[r.Ordinal].Output[id] = out[id]
		}
		r.Status = "repaired"
		r.Reason = check.Reason
		r.ResultHash = documentTextHash(target)
		if err = save(attempt, units[r.Ordinal]); err != nil {
			units[r.Ordinal].Output = original
			*r = old
			return err
		}
	}
}
