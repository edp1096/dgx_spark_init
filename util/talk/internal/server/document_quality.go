package server

import (
	"fmt"
	"sparktalk/internal/workflows"
	"strings"
	"unicode/utf8"
)

func shortDocumentOutputs(u workflows.DocumentUnit) map[string]string {
	bad := map[string]string{}
	for _, group := range documentTextGroups(u) {
		if utf8.RuneCountInString(group.Source) > 160 && utf8.RuneCountInString(group.Output)*8 < utf8.RuneCountInString(group.Source) {
			for _, in := range group.Inputs {
				bad[in.ID] = "unexpectedly short translation: " + in.ID
			}
		}
	}
	return bad
}

// Move only failed fragments to a repair unit; retain good translations and
// preserve exact source positions. A repeated short result is visibly unverified.
func extractDocumentRepair(u *workflows.DocumentUnit, bad map[string]string, ordinal int) *workflows.DocumentUnit {
	good := []workflows.DocumentInput{}
	failed := []workflows.DocumentInput{}
	goodOutput := map[string]string{}
	for _, in := range u.Inputs {
		if _, ok := bad[in.ID]; ok {
			failed = append(failed, in)
		} else {
			good = append(good, in)
			goodOutput[in.ID] = u.Output[in.ID]
		}
	}
	if len(failed) == 0 {
		return nil
	}
	repair := &workflows.DocumentUnit{Ordinal: ordinal, Stage: "repair", Status: "pending", Inputs: failed, QualityRetries: u.QualityRetries + 1}
	if len(good) == 0 {
		u.QualityRetries++
		u.Stage = "repair"
		u.RetryBase = len(u.Attempts)
		u.Status = "pending"
		u.Output = nil
		return nil
	}
	u.Inputs = good
	u.Output = goodOutput
	return repair
}
func (s *Server) queueDocumentQualityRepair(run *workflows.Run) error {
	units, e := s.db.DocumentUnits(run.SessionID, run.ID)
	if e != nil {
		return e
	}
	units, e = prepareDocumentQualityRepair(run, units)
	if e != nil {
		return e
	}
	return s.db.SaveDocumentRepair(run, units...)
}

func prepareDocumentQualityRepair(run *workflows.Run, units []workflows.DocumentUnit) ([]workflows.DocumentUnit, error) {
	changed := false
	for i := 0; i < len(units); i++ {
		units[i].Quality = nil
		if units[i].Status != "completed" {
			changed = true // A revised source range also requires translation.
			continue
		}
		bad := shortDocumentOutputs(units[i])
		if len(bad) == 0 {
			continue
		}
		changed = true
		units[i].QualityRetries = 0
		warnings := units[i].Warnings[:0]
		for _, warning := range units[i].Warnings {
			if !strings.HasPrefix(warning, "unexpectedly short translation: ") {
				warnings = append(warnings, warning)
			}
		}
		units[i].Warnings = warnings
		repair := extractDocumentRepair(&units[i], bad, len(units))
		if repair != nil {
			repair.QualityRetries = 0
			units = append(units, *repair)
		}
	}
	if !changed {
		return nil, fmt.Errorf("no short translation fragments to retry")
	}
	run.Status = "paused"
	run.Current = 0
	run.Document.Phase = "process"
	run.Document.Error = ""
	run.Document.Formats = nil
	for i := range run.Steps {
		run.Steps[i].Status = "pending"
	}
	warnings := run.Document.Warnings[:0]
	for _, w := range run.Document.Warnings {
		if !strings.HasPrefix(w, "unexpectedly short translation") {
			warnings = append(warnings, w)
		}
	}
	run.Document.Warnings = warnings
	run.Document.Total = len(units)
	run.Document.Completed = 0
	for _, u := range units {
		if u.Status == "completed" {
			run.Document.Completed++
		}
	}
	return units, nil
}
