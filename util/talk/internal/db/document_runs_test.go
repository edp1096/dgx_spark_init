package db

import (
	"path/filepath"
	"sparktalk/internal/sourcebook"
	"sparktalk/internal/workflows"
	"testing"
	"time"
)

func TestSourceSnapshotRecoveryIsolationAndRetention(t *testing.T) {
	path := filepath.Join(t.TempDir(), "db.sqlite")
	d, e := Open(path)
	if e != nil {
		t.Fatal(e)
	}
	if _, e = d.CreateSession("s", "test", "model", "off"); e != nil {
		t.Fatal(e)
	}
	b, e := sourcebook.Parse([]byte("Chapter one.\n\nFinal paragraph."), "text/plain", "https://example.org", nil)
	if e != nil {
		t.Fatal(e)
	}
	if e = d.SaveSource("src", "s", b, []byte("raw")); e != nil {
		t.Fatal(e)
	}
	if _, e = d.Source("other", "src", false); e == nil {
		t.Fatal("source crossed session boundary")
	}
	r := workflows.Run{ID: "r", SessionID: "s", Status: "running", Document: &workflows.DocumentTask{Kind: "translate", Sources: []string{"src"}}, Steps: []workflows.StepState{{Status: "running"}}}
	if e = d.SaveWorkflowRun(&r); e != nil {
		t.Fatal(e)
	}
	unit := workflows.DocumentUnit{Ordinal: 0, Status: "completed", Output: map[string]string{"b1": "보존된 결과"}, Attempts: []workflows.DocumentAttempt{{FinishReason: "tool_calls", OutputTokens: 10}}}
	if e = d.SaveDocumentProgress(&r, unit); e != nil {
		t.Fatal(e)
	}
	if e = d.SaveDocumentOutput("s", "r", "output.txt", "text/plain", []byte("result")); e != nil {
		t.Fatal(e)
	}
	d.conn.Exec(`UPDATE source_snapshots SET created=?`, time.Now().Add(-48*time.Hour).Unix())
	if e = d.CleanupSources(); e != nil {
		t.Fatal(e)
	}
	if !d.SourceExists("s", "src") {
		t.Fatal("referenced source expired")
	}
	d.Close()
	d, e = Open(path)
	if e != nil {
		t.Fatal(e)
	}
	defer d.Close()
	got, e := d.DocumentRun("s", "r")
	if e != nil || got.Status != "paused" {
		t.Fatal("restart did not pause", e)
	}
	units, e := d.DocumentUnits("s", "r")
	if e != nil || len(units) != 1 || units[0].Output["b1"] != "보존된 결과" {
		t.Fatal("restart lost result", e)
	}
	// Check a second pooled connection too: cascading ownership must not depend on
	// which SQLite connection happens to execute a deletion.
	held, e := d.conn.Conn(t.Context())
	if e != nil {
		t.Fatal(e)
	}
	if e = d.DeleteSession("s"); e != nil {
		t.Fatal(e)
	}
	held.Close()
	if d.SourceExists("s", "src") {
		t.Fatal("session deletion retained source")
	}
	var n int
	for _, table := range []string{"source_snapshots", "source_assets", "workflow_runs", "workflow_units", "workflow_outputs"} {
		if e = d.conn.QueryRow("SELECT COUNT(*) FROM " + table).Scan(&n); e != nil || n != 0 {
			t.Fatalf("cascade failed for %s: %d %v", table, n, e)
		}
	}
	if e = d.SaveDocumentProgress(&r); e == nil {
		t.Fatal("deleted run recreated by stale worker")
	}
}
