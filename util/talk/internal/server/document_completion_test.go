package server

import (
	"context"
	"net/http"
	"reflect"
	"sync/atomic"
	"testing"
)

func TestFinishedDocumentWarningsDoNotReopenOrCancelTheTask(t *testing.T) {
	var calls atomic.Int32
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		mockDocumentResponse(w, r, true)
	})
	source := fixtureSource(t, s, "Holmes waited for Watson.")
	run, err := s.documentAction(context.Background(), "s", documentStart{Action: "start", Kind: "translate", Sources: []string{source}, Glossary: map[string]string{"Holmes": "홈즈"}})
	if err != nil {
		t.Fatal(err)
	}
	run = waitDocument(t, s, run.ID)
	if run.Status != "needs_review" || !run.DocumentFinished() || len(run.Document.Warnings) == 0 {
		t.Fatalf("fixture did not finish with warnings: %+v", run)
	}
	if documentView(run)["finished"] != true {
		t.Fatal("task API did not distinguish completion from review warnings")
	}
	before, err := s.db.DocumentUnits("s", run.ID)
	if err != nil {
		t.Fatal(err)
	}
	for _, action := range []string{"pause", "cancel"} {
		after, err := s.documentAction(context.Background(), "s", documentStart{Action: action, RunID: run.ID})
		if err != nil || !reflect.DeepEqual(run, after) {
			t.Fatalf("%s changed a finished task: %v", action, err)
		}
	}
	if _, err = s.documentAction(context.Background(), "s", documentStart{Action: "resume", RunID: run.ID, ExtendBudget: true}); err == nil {
		t.Fatal("generic resume accepted a finished task")
	}
	if _, err = s.db.ClaimDocumentRun("s", run.ID); err == nil {
		t.Fatal("finished warnings task was claimed directly")
	}
	after, err := s.db.DocumentUnits("s", run.ID)
	if err != nil || !reflect.DeepEqual(before, after) || calls.Load() != 1 {
		t.Fatal("completed checkpoints were reprocessed", err, calls.Load())
	}
	if data, _, err := s.db.DocumentOutput("s", run.ID, "output.html"); err != nil || len(data) == 0 {
		t.Fatal("finished output lost", err)
	}
}
