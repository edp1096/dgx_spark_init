package server

import (
	"strings"
	"testing"

	"sparktalk/internal/llm"
	"sparktalk/internal/workflows"
)

func TestDocumentReviewedContextDoesNotCrossRepairedGap(t *testing.T) {
	u := workflows.DocumentUnit{Status: "completed", Inputs: []workflows.DocumentInput{
		{ID: "before", Source: "s", Block: "b000001", Group: "before", Text: "A prior sentence."},
		{ID: "after", Source: "s", Block: "b000003", Group: "after", Text: "A later sentence."},
	}, Output: map[string]string{"before": "앞에서 검수한 문장.", "after": "아직 오지 않은 뒤쪽 문장."}}
	review, err := validateDocumentReview(u, llm.StreamResult{FinishReason: "stop", Content: `{"reviews":{"p000001":[],"p000002":[]}}`})
	if err != nil {
		t.Fatal(err)
	}
	documentMergeReview(&u, review)
	next := workflows.DocumentUnit{Inputs: []workflows.DocumentInput{{ID: "gap", Source: "s", Block: "b000002", Text: "Repair this gap."}}}
	got := documentReviewedContext([]workflows.DocumentUnit{u}, next)
	if !strings.Contains(got, u.Output["before"]) || strings.Contains(got, u.Output["after"]) {
		t.Fatal("future translation leaked across a repaired gap", got)
	}
	// A failed sibling does not remove the hash-valid previous paragraph.
	u.Quality.Status = "pending"
	if got = documentReviewedContext([]workflows.DocumentUnit{u}, next); !strings.Contains(got, u.Output["before"]) || strings.Contains(got, u.Output["after"]) {
		t.Fatal("valid sibling context was lost", got)
	}
	u.Quality.Reviews[0].Findings = []workflows.DocumentFinding{{Severity: "minor", Uncertain: true}}
	if got = documentReviewedContext([]workflows.DocumentUnit{u}, next); got != "" {
		t.Fatal("uncertain paragraph became accepted context", got)
	}
	u.Quality.Reviews[0].Findings = nil
	u.Output["before"] = "Changed after the review."
	if got = documentReviewedContext([]workflows.DocumentUnit{u}, next); got != "" {
		t.Fatal("stale review authorized changed context", got)
	}
}

func TestDocumentReviewedContextRejectsOverlappingSourceRange(t *testing.T) {
	u := workflows.DocumentUnit{Status: "completed", Inputs: []workflows.DocumentInput{{ID: "s:b:0", Source: "s", Block: "b", Group: "g", Text: "앞문장"}}, Output: map[string]string{"s:b:0": "Accepted prefix."}}
	review, err := validateDocumentReview(u, llm.StreamResult{FinishReason: "stop", Content: `{"reviews":{"p000001":[]}}`})
	if err != nil {
		t.Fatal(err)
	}
	documentMergeReview(&u, review)
	next := workflows.DocumentUnit{Inputs: []workflows.DocumentInput{{ID: "s:b:2", Source: "s", Block: "b", Part: 2, Text: "Overlapping range."}}}
	if got := documentReviewedContext([]workflows.DocumentUnit{u}, next); got != "" {
		t.Fatal("overlapping source became prior context", got)
	}
	next.Inputs[0].Part = 3
	if got := documentReviewedContext([]workflows.DocumentUnit{u}, next); !strings.Contains(got, "Accepted prefix.") {
		t.Fatal("adjacent source range lost", got)
	}
}
