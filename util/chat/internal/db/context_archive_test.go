package db

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestCompactedRecallBoundaryAndArchiveIsolation(t *testing.T) {
	d, err := Open(filepath.Join(t.TempDir(), "test.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	d.CreateSession("s", "Deployment path", "m", "low")
	d.CreateSession("other", "Other", "m", "low")
	old, _ := d.AddMessage("s", "user", "Deployment path /mnt/samsung/ple", "", nil, nil)
	d.AddMessage("s", "user", "Deployment path /recent", "", nil, nil)
	d.AddMessage("other", "user", "Deployment path /other", "", nil, nil)
	if text, err := d.ReadContextMessage("s", old.ID); err != nil || text != old.Content {
		t.Fatal("original message read failed")
	}
	if _, err := d.ReadContextMessage("other", old.ID); err == nil {
		t.Fatal("cross-session message exposed")
	}
	rows, err := d.SearchCompactedMessages("Deployment path", "s", old.ID, 5)
	if err != nil || len(rows) != 1 || rows[0].MessageID != old.ID {
		t.Fatalf("incorrect scope: %+v %v", rows, err)
	}
	raw := strings.Repeat("full result\n", 1000)
	id, err := d.ArchiveContextTool("s", "test", raw)
	if err != nil {
		t.Fatal(err)
	}
	got, err := d.ReadContextTool("s", id)
	if err != nil || got != raw {
		t.Fatal("archive changed")
	}
	if _, err := d.ReadContextTool("other", id); err == nil {
		t.Fatal("cross-session archive exposed")
	}
	anchored, err := d.ArchiveContextTool("s", "test", "anchored", old.ID)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := d.conn.Exec(`DELETE FROM messages WHERE id=?`, old.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := d.ReadContextTool("s", anchored); err == nil {
		t.Fatal("deleted message archive remains")
	}
	if err := d.DeleteSession("s"); err != nil {
		t.Fatal(err)
	}
	if _, err := d.ReadContextTool("s", id); err == nil {
		t.Fatal("deleted conversation archive remains")
	}

}
