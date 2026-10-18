package server

import (
	"crypto/rand"
	"fmt"
	"maps"
	"strings"

	"sparktalk/internal/workflows"
)

// Changing validation/recovery rules must not return an older completed
// correction merely because its parent and model are unchanged.
const documentTermsCorrectionRevision = "6"

// A narrow correction is its own execution. Original downloads, source snapshots
// and translation checkpoints remain accessible and are never overwritten.
func (s *Server) createDocumentTermsRevision(parent workflows.Run, repair bool, requests ...*workflows.DocumentRequestSpec) (workflows.Run, error) {
	if parent.Document == nil || parent.Document.Kind != "translate" || !parent.DocumentFinished() || len(parent.Document.Sources) != 1 {
		return workflows.Run{}, fmt.Errorf("완료된 번역 작업에서 표기를 확인할 수 있습니다")
	}
	cfg, _ := s.snapshot()
	requestScope := ""
	if len(requests) > 0 && requests[0] != nil {
		requestScope = requests[0].Hash
	}
	requestKey := requestScope + fmt.Sprintf("terms/%s/%t/%s/%s/%s/%s", parent.ID, repair, workflows.DocumentTermsVersion, documentTermsCorrectionRevision, cfg.Model.DefaultModel, cfg.Model.ReasoningEffort)
	for {
		existing, err := s.db.DocumentRequest(parent.SessionID, requestKey)
		if err != nil {
			break
		}
		if existing.Status != "canceled" {
			return existing, nil
		}
		// A later explicit request after cancellation creates a fresh revision;
		// repeated clicks still return that same replacement execution.
		requestKey += "/retry"
	}
	oldID := parent.Document.Sources[0]
	book, err := s.db.Source(parent.SessionID, oldID, true)
	if err != nil {
		return workflows.Run{}, err
	}
	raw, err := s.db.RawSource(parent.SessionID, oldID)
	if err != nil {
		return workflows.Run{}, err
	}
	previous, err := s.db.DocumentUnits(parent.SessionID, parent.ID)
	if err != nil {
		return workflows.Run{}, err
	}
	if book.Metadata == nil {
		book.Metadata = map[string]any{}
	}
	book.Metadata["parent_source_id"] = oldID
	newID, err := s.storeSource(parent.SessionID, book, raw)
	if err != nil {
		return workflows.Run{}, err
	}
	saved := false
	defer func() {
		if !saved {
			_ = s.db.DeleteSource(parent.SessionID, newID)
		}
	}()
	planned := planDocumentTranslation(newID, book, parent.Document.Start, parent.Document.End)
	units := carryDocumentTranslation(oldID, newID, book, book, previous, planned)
	for i := range units {
		if units[i].Status != "completed" {
			return workflows.Run{}, fmt.Errorf("원문 범위에 대응하는 저장 번역이 없습니다")
		}
		units[i].Attempts = nil
		units[i].Quality = nil
		units[i].Warnings = nil
	}
	spec, err := s.documentRevisionRequest(parent)
	if err != nil {
		return workflows.Run{}, err
	}
	if len(requests) > 0 && requests[0] != nil {
		combined := *spec
		combined.UserText = spec.UserText + "\n\nCurrent user correction request:\n" + requests[0].UserText
		combined.Hash = documentTextHash(combined.UserText)
		combined.MessageIDs = append(append([]int64(nil), spec.MessageIDs...), requests[0].MessageIDs...)
		combined.Origin = "user_correction"
		spec = &combined
	}
	terms := &workflows.DocumentTerms{Repair: repair, Status: "pending"}
	// Old inferred targets are deliberately not carried as authoritative values.
	names := map[string]bool{}
	for name := range parent.Document.Glossary {
		names[name] = true
	}
	if parent.Document.Terms != nil {
		for _, candidate := range parent.Document.Terms.Candidates {
			names[candidate.Source] = true
		}
	}
	for _, name := range sortedTermKeys(names) {
		terms.Candidates = append(terms.Candidates, workflows.DocumentEntity{ID: documentTextHash(newID + "\x00" + name)[:24], Version: 1, Source: name, Kind: "unknown", Origin: "legacy_candidate", Status: "candidate"})
	}
	// Old layout and new layout share source IDs/text offsets; use recorded input
	// policies to count the paragraphs whose soft wraps are now classified as prose.
	for _, u := range previous {
		for _, in := range u.Inputs {
			if in.Whitespace == "pre-wrap" {
				for _, n := range units {
					for _, next := range n.Inputs {
						if next.Block == in.Block && next.Part == in.Part && next.Whitespace == "normal" {
							terms.LayoutBlocks++
						}
					}
				}
			}
		}
	}
	d := &workflows.DocumentTask{RequestID: requestKey, RequestHash: documentTextHash(requestKey), Pipeline: workflows.DocumentTermsVersion, Terms: terms, RequestSpec: spec, Kind: "translate", Sources: []string{newID}, SourceURLs: map[string]string{newID: book.URL}, SourceTitles: map[string]string{newID: book.Title}, Model: cfg.Model.DefaultModel, ModelType: cfg.Model.ModelType, Endpoint: cfg.Model.Endpoint, Effort: cfg.Model.ReasoningEffort, Instructions: spec.UserText, Revision: parent.Document.Revision + 1, Parent: parent.ID, Start: parent.Document.Start, End: parent.Document.End, UserGlossary: maps.Clone(parent.Document.UserGlossary), RequestedFormat: parent.Document.RequestedFormat, Formats: []string{"txt", "md", "html", "zip"}, Phase: "terms_extract", MaxCalls: 2048, MaxOutputTokens: 2_000_000, MaxSeconds: 12 * 60 * 60, OutputLimit: cfg.Context.EffectiveOutputReserve(cfg.Context.WindowTokens), Total: len(units), Completed: len(units)}
	d.Profiles = documentProfiles(cfg, d.Effort)
	// Regenerate previously requested conversions from the corrected text too.
	for _, f := range parent.Document.Formats {
		if f == "pdf" || f == "docx" || f == "hwp" || f == "hwpx" {
			terms.Formats = append(terms.Formats, f)
		}
	}
	run := workflows.Run{ID: "document-" + rand.Text(), SessionID: parent.SessionID, Status: "paused", Goal: parent.Goal, Definition: parent.Definition, Document: d, Steps: []workflows.StepState{{Status: "pending"}, {Status: "pending"}}}
	reuseDocumentTerms(parent, previous, units, &run)
	for _, w := range parent.Document.Warnings {
		if !strings.HasPrefix(w, "glossary spelling requires review:") && !strings.HasPrefix(w, "term consistency unresolved:") && !strings.HasPrefix(w, "translation language requires review:") {
			d.Warnings = append(d.Warnings, w)
		}
	}
	err = s.db.CreateDocumentRun(&run, units)
	saved = err == nil
	return run, err
}
