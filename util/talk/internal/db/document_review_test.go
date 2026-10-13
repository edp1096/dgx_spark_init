package db

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"path/filepath"
	"testing"

	"sparktalk/internal/performance"
	"sparktalk/internal/workflows"
)

func TestDocumentStagePerformanceUsesImmutableRequestTotals(t *testing.T) {
	d, run := reviewedRunFixture(t)
	for i, sample := range []performance.Sample{
		{PromptTokens: 1000, CachedTokens: 900, OutputTokens: 21, PrefillTokens: 100, PrefillSeconds: 1, DecodeTokens: 20, DecodeSeconds: 2},
		{PromptTokens: 300, OutputTokens: 11, PrefillTokens: 300, PrefillSeconds: 2, DecodeTokens: 10, DecodeSeconds: 2, TGEstimated: true},
	} {
		var measured performance.Accumulator
		measured.Start()(sample)
		attempt := &workflows.DocumentRequestAttempt{ID: []string{"first", "second"}[i], Stage: "review", Measurement: workflows.DocumentAttempt{
			PromptTokens: sample.PromptTokens, OutputTokens: sample.OutputTokens, Seconds: float64(i + 3), Performance: measured.Summary(false)}}
		for repeat := 0; repeat < 2; repeat++ {
			if err := d.SaveDocumentReviewProgress(&run, attempt, nil, nil); err != nil {
				t.Fatal(err)
			}
		}
	}
	// A request without timing still contributes to usage, but not a fabricated rate.
	if err := d.SaveDocumentReviewProgress(&run, &workflows.DocumentRequestAttempt{ID: "old", Stage: "review", Measurement: workflows.DocumentAttempt{Seconds: 3, PromptTokens: 50, OutputTokens: 5}}, nil, nil); err != nil {
		t.Fatal(err)
	}
	m := run.Document.StageMetrics["review"]
	if m.Calls != 3 || m.Seconds != 10 || m.PromptTokens != 1350 || m.OutputTokens != 37 ||
		m.Performance.Calls != 2 || *m.Performance.PP != 400.0/3 || *m.Performance.TG != 7.5 || !m.Performance.TGEstimated {
		t.Fatalf("wrong canonical stage metrics: %+v performance=%+v", m, m.Performance)
	}
}

func reviewedRunFixture(t *testing.T) (*DB, workflows.Run) {
	t.Helper()
	database, err := Open(filepath.Join(t.TempDir(), "test.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { database.Close() })
	if _, err = database.CreateSession("s", "test", "model", "low"); err != nil {
		t.Fatal(err)
	}
	run := workflows.Run{ID: "r", SessionID: "s", Status: "paused", Document: &workflows.DocumentTask{Pipeline: workflows.DocumentPipelineVersion, Kind: "translate"}}
	if err = database.CreateDocumentRun(&run, nil); err != nil {
		t.Fatal(err)
	}
	run, err = database.ClaimDocumentRun("s", "r")
	if err != nil {
		t.Fatal(err)
	}
	return database, run
}

func TestDocumentReviewCheckpointAtomicAccountingAndStaleWorker(t *testing.T) {
	d, run := reviewedRunFixture(t)
	unit := workflows.DocumentUnit{Ordinal: 0, Status: "completed", Output: map[string]string{"a": "kept"}}
	attempt := &workflows.DocumentRequestAttempt{ID: "a1", Stage: "review", Measurement: workflows.DocumentAttempt{PromptTokens: 100, OutputTokens: 20, Seconds: 1.5}}
	review := &workflows.DocumentReview{ID: "v1", Status: "reviewed"}
	if err := d.SaveDocumentReviewProgress(&run, attempt, review, nil, unit); err != nil {
		t.Fatal(err)
	}
	run.Document.Calls = 999 // A repeated checkpoint must not duplicate usage.
	if err := d.SaveDocumentReviewProgress(&run, attempt, review, nil, unit); err != nil {
		t.Fatal(err)
	}
	if run.Document.Calls != 1 || run.Document.OutputTokens != 20 {
		t.Fatal("batch usage duplicated", run.Document)
	}
	conflict := *attempt
	conflict.Measurement.OutputTokens = 500
	unit.Output = map[string]string{"a": "must roll back"}
	if err := d.SaveDocumentReviewProgress(&run, &conflict, review, nil, unit); err == nil {
		t.Fatal("conflicting request record replaced")
	}
	units, err := d.DocumentUnits("s", "r")
	if err != nil || units[0].Output["a"] != "kept" {
		t.Fatal("partial checkpoint escaped rollback", err, units)
	}
	stale := run
	doc := *run.Document
	stale.Document = &doc
	run.Status = "paused"
	if err = d.SaveDocumentProgress(&run); err != nil {
		t.Fatal(err)
	}
	run, err = d.ClaimDocumentRun("s", "r")
	if err != nil {
		t.Fatal(err)
	}
	if run.Document.ExecutionID == stale.Document.ExecutionID {
		t.Fatal("resume retained stale execution token")
	}
	if err = d.SaveDocumentReviewProgress(&stale, &workflows.DocumentRequestAttempt{ID: "late", Stage: "review"}, nil, nil, unit); err == nil {
		t.Fatal("late previous-worker result committed")
	}
	var n int
	d.conn.QueryRow(`SELECT COUNT(*) FROM document_attempts WHERE id='late'`).Scan(&n)
	if n != 0 {
		t.Fatal("stale usage persisted")
	}
}

func TestDocumentPublicationRequiresCompleteManifestAndPreservesOldRevision(t *testing.T) {
	d, run := reviewedRunFixture(t)
	if err := d.SaveDocumentOutput("s", "r", "output.txt", "text/plain", []byte("original")); err != nil {
		t.Fatal(err)
	}
	if err := d.PublishDocumentRevision(&run, []string{"output.txt", "output.pdf"}, "hash"); err == nil {
		t.Fatal("partial manifest published")
	}
	saved, err := d.DocumentRun("s", "r")
	if err != nil || saved.Document.Publication != nil {
		t.Fatal("failed publication leaked", err)
	}
	if err = d.PublishDocumentRevision(&run, []string{"output.txt"}, "hash"); err != nil {
		t.Fatal(err)
	}
	if len(run.Document.Publication.Files) != 1 {
		t.Fatal("missing publication")
	}
	if err = d.SaveDocumentOutput("s", "r", "output.txt", "text/plain", []byte("silently changed")); err == nil {
		t.Fatal("published text overwritten")
	}
	if err = d.ClearDocumentOutputs("s", "r"); err == nil {
		t.Fatal("published revision cleared")
	}
	if err = d.SaveDocumentRepair(&run); err == nil {
		t.Fatal("versioned repair deleted published outputs")
	}
	child := workflows.Run{ID: "child", SessionID: "s", Status: "paused", Document: &workflows.DocumentTask{Pipeline: workflows.DocumentPipelineVersion, Kind: "translate", Parent: "r", Revision: 2}}
	if err = d.CreateDocumentRun(&child, nil); err != nil {
		t.Fatal(err)
	}
	if err = d.SaveDocumentOutput("s", "child", "output.txt", "text/plain", []byte("new unpublished version")); err != nil {
		t.Fatal(err)
	}
	raw, _, err := d.DocumentOutput("s", "r", "output.txt")
	if err != nil || string(raw) != "original" {
		t.Fatal("parent output replaced", err)
	}
	if err = d.DeleteDocumentRun("s", "child"); err != nil {
		t.Fatal(err)
	}
	if _, _, err = d.DocumentOutput("s", "r", "output.txt"); err != nil {
		t.Fatal("child deletion removed parent", err)
	}
}

func TestDocumentEntityVersionsAndSessionCascade(t *testing.T) {
	d, run := reviewedRunFixture(t)
	entity := workflows.DocumentEntity{ID: "animal", Version: 1, Source: "Pip", Target: "핍", Kind: "animal", Evidence: []workflows.DocumentEntityEvidence{{InputID: "s:b1:0", Quote: "Pip was a dog."}}}
	if err := d.SaveDocumentReviewProgress(&run, nil, nil, []workflows.DocumentEntity{entity}); err != nil {
		t.Fatal(err)
	}
	entity.Version = 2
	entity.Target = "피프"
	if err := d.SaveDocumentReviewProgress(&run, nil, nil, []workflows.DocumentEntity{entity}); err != nil {
		t.Fatal(err)
	}
	items, err := d.DocumentEntities("s", "r")
	if err != nil || len(items) != 1 || items[0].Version != 2 {
		t.Fatal(items, err)
	}
	if _, err = d.DocumentEntities("other", "r"); err == nil {
		t.Fatal("cross-session entity access")
	}
	if err = d.DeleteSession("s"); err != nil {
		t.Fatal(err)
	}
	for _, table := range []string{"document_attempts", "document_reviews", "document_entities"} {
		var n int
		if err = d.conn.QueryRow("SELECT COUNT(*) FROM " + table).Scan(&n); err != nil || n != 0 {
			t.Fatal("cascade", table, n, err)
		}
	}
}

func TestDocumentReviewRejectsChangedTargetWithinSameExecution(t *testing.T) {
	d, run := reviewedRunFixture(t)
	unit := workflows.DocumentUnit{Ordinal: 0, Status: "completed", Output: map[string]string{"a": "first"}}
	if err := d.SaveDocumentProgress(&run, unit); err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal(unit.Output)
	hash := sha256.Sum256(raw)
	attempt := &workflows.DocumentRequestAttempt{ID: "review-old-text", Stage: "review", ExpectedUnitOutputHashes: map[int]string{0: hex.EncodeToString(hash[:])}}
	unit.Output = map[string]string{"a": "corrected"}
	if err := d.SaveDocumentProgress(&run, unit); err != nil {
		t.Fatal(err)
	}
	if err := d.SaveDocumentReviewProgress(&run, attempt, &workflows.DocumentReview{ID: "late-review"}, nil, unit); !errors.Is(err, ErrStaleDocumentExecution) {
		t.Fatal("stale target accepted", err)
	}
	reviews, err := d.DocumentReviews("s", "r")
	if err != nil || len(reviews) != 0 {
		t.Fatal("stale review leaked", reviews, err)
	}
	saved, err := d.DocumentRun("s", "r")
	if err != nil || saved.Document.Calls != 0 {
		t.Fatal("rejected request charged", err)
	}
}
