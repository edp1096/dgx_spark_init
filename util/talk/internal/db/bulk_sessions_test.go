package db

import (
	"path/filepath"
	"testing"
)

func TestDeleteSessionsAtomicAndPreservesUnselected(t *testing.T) {
	d, err := Open(filepath.Join(t.TempDir(), "bulk.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	for _, id := range []string{"a", "b", "keep"} {
		if _, err = d.CreateSession(id, id, "model", ""); err != nil {
			t.Fatal(err)
		}
		if _, err = d.AddMessage(id, "user", "retained text", "", nil, nil); err != nil {
			t.Fatal(err)
		}
	}
	_, err = d.conn.Exec(`CREATE TRIGGER block_delete BEFORE DELETE ON sessions WHEN OLD.id='b' BEGIN SELECT RAISE(ABORT,'test rollback'); END`)
	if err != nil {
		t.Fatal(err)
	}
	if d.DeleteSessions([]string{"a", "b"}) == nil {
		t.Fatal("expected failure")
	}
	for _, id := range []string{"a", "b", "keep"} {
		m, e := d.Messages(id)
		if e != nil || len(m) != 1 {
			t.Fatalf("rollback lost %s", id)
		}
	}
	if _, err = d.conn.Exec(`DROP TRIGGER block_delete`); err != nil {
		t.Fatal(err)
	}
	if err = d.DeleteSessions([]string{"a", "b"}); err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"a", "b"} {
		m, e := d.Messages(id)
		if e != nil || len(m) != 0 {
			t.Fatal("messages not deleted")
		}
	}
	var count int
	if err = d.conn.QueryRow(`SELECT COUNT(*) FROM sessions`).Scan(&count); err != nil || count != 1 {
		t.Fatalf("unselected session affected: %v %d", err, count)
	}
	m, e := d.Messages("keep")
	if e != nil || len(m) != 1 {
		t.Fatal("unselected messages affected")
	}
}
