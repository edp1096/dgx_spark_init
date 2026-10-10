package db

import (
	"database/sql"
	"errors"
	"testing"
)

func TestToolFolderMovePreservesNewerUIAssignment(t *testing.T) {
	d, err := Open(t.TempDir() + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	for _, id := range []string{"current", "other"} {
		if _, err = d.CreateSession(id, id, "model", "none"); err != nil {
			t.Fatal(err)
		}
	}
	for _, id := range []string{"news", "work"} {
		if _, err = d.CreateGroup(id, id); err != nil {
			t.Fatal(err)
		}
	}
	if _, err = d.MoveSessionGroupIfUnchanged("current", "", "missing"); !errors.Is(err, sql.ErrNoRows) {
		t.Fatalf("unknown folder: %v", err)
	}
	if changed, err := d.MoveSessionGroupIfUnchanged("current", "", "news"); err != nil || !changed {
		t.Fatalf("move: %v %v", changed, err)
	}
	if changed, err := d.MoveSessionGroupIfUnchanged("current", "news", "news"); err != nil || changed {
		t.Fatalf("same folder: %v %v", changed, err)
	}
	if err = d.SetSessionGroup("current", "work"); err != nil {
		t.Fatal(err)
	}
	if changed, err := d.MoveSessionGroupIfUnchanged("current", "news", ""); err != nil || changed {
		t.Fatalf("stale move: %v %v", changed, err)
	}
	if current, _ := d.Session("current"); current.GroupID != "work" {
		t.Fatal("overwrote UI move")
	}
	if other, _ := d.Session("other"); other.GroupID != "" {
		t.Fatal("moved unrelated conversation")
	}
	if changed, err := d.MoveSessionGroupIfUnchanged("current", "work", ""); err != nil || !changed {
		t.Fatalf("ungroup: %v %v", changed, err)
	}
	if err = d.DeleteGroup("news"); err != nil {
		t.Fatal(err)
	}
	if _, err = d.MoveSessionGroupIfUnchanged("current", "", "news"); !errors.Is(err, sql.ErrNoRows) {
		t.Fatalf("deleted folder: %v", err)
	}
}
