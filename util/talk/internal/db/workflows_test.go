package db

import (
	"path/filepath"
	"sparktalk/internal/workflows"
	"sync"
	"testing"
)

func TestWorkflowRestartClaimAndSessionIsolation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "workflows.db")
	store, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = store.CreateSession("s", "test", "model", "none"); err != nil {
		t.Fatal(err)
	}
	run := workflows.Run{ID: "r", SessionID: "s", Status: "running", Current: 1, Steps: []workflows.StepState{{Status: "completed", Summary: "first"}, {Status: "running", Summary: "partial"}}}
	if err = store.SaveWorkflowRun(&run); err != nil {
		t.Fatal(err)
	}
	store.Close()
	store, err = Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	runs, err := store.WorkflowRuns("s")
	if err != nil || len(runs) != 1 || runs[0].Status != "paused" || runs[0].Steps[1].Status != "paused" || runs[0].Steps[0].Summary != "first" {
		t.Fatalf("restart lost state: %+v %v", runs, err)
	}
	if _, err = store.ClaimWorkflowRun("r", "other"); err == nil {
		t.Fatal("cross-session claim accepted")
	}
	var wg sync.WaitGroup
	success := make(chan workflows.Run, 2)
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			claimed, e := store.ClaimWorkflowRun("r", "s")
			if e == nil {
				success <- claimed
			}
		}()
	}
	wg.Wait()
	close(success)
	if len(success) != 1 {
		t.Fatalf("expected exactly one claim, got %d", len(success))
	}
	claimed := <-success
	claimed.Status = "completed"
	if err = store.SaveWorkflowRun(&claimed); err != nil {
		t.Fatal(err)
	}
	if _, err = store.ClaimWorkflowRun("r", "s"); err == nil {
		t.Fatal("completed run restarted")
	}
	if err = store.DeleteSession("s"); err != nil {
		t.Fatal(err)
	}
	runs, err = store.WorkflowRuns("s")
	if err != nil || len(runs) != 0 {
		t.Fatal("deleted session retained workflow runs", err)
	}
}
