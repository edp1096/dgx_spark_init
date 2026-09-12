package db

import (
	"path/filepath"
	"testing"

	"sparktalk/internal/performance"
)

func TestPerformancePersistsWithCompletedRetryEditedAndPartialVariants(t *testing.T) {
	path := filepath.Join(t.TempDir(), "performance.db")
	d, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	d.CreateSession("s", "test", "model", "none")
	u, _ := d.AddPendingMessage("s", "question", nil)
	first := &performance.Summary{Calls: 1, OutputTokens: 21}
	a, err := d.CompletePendingTurnWithAttachments(u.ID, "first", "reasoning", nil, nil, first)
	if err != nil {
		t.Fatal(err)
	}
	second := &performance.Summary{Calls: 2, OutputTokens: 45}
	if err = d.ReplaceAssistantWithAttachments(a.ID, "retry", "", nil, nil, 0, second); err != nil {
		t.Fatal(err)
	}
	third := &performance.Summary{Calls: 3, OutputTokens: 90}
	if err = d.AppendEditedBranchWithAnswerAttachments(u.ID, "edited question", nil, "edited answer", "", nil, nil, third); err != nil {
		t.Fatal(err)
	}
	d.Close()
	d, err = Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	rows, _ := d.Messages("s")
	if len(rows) != 2 || rows[1].Performance == nil || rows[1].Performance.OutputTokens != 90 || len(rows[1].Variants) != 3 {
		t.Fatalf("current metrics lost after reopen: %+v", rows)
	}
	for i, count := range []int{21, 45, 90} {
		if rows[1].Variants[i].Performance.OutputTokens != count {
			t.Fatal("variant metrics overwritten")
		}
	}
	u, _ = d.AddPendingMessage("s", "cancelled question", nil)
	if err = d.FailPendingTurn(u.ID, MessageCancelled, "cancelled", "partial", "", nil, first); err != nil {
		t.Fatal(err)
	}
	rows, _ = d.Messages("s")
	last := rows[len(rows)-1]
	if last.Status != MessageCancelled || last.Performance == nil || last.Performance.OutputTokens != 21 {
		t.Fatalf("partial metrics lost: %+v", last)
	}
}
