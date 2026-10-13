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

func TestToolFolderDeleteRollsBackUngroupingOnFailure(t *testing.T) {
	d, err := Open(t.TempDir() + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	d.CreateSession("s", "kept", "model", "low")
	d.CreateGroup("g", "번역")
	d.SetSessionGroup("s", "g")
	if _, err = d.conn.Exec(`CREATE TRIGGER deny_group_delete BEFORE DELETE ON chat_groups BEGIN SELECT RAISE(ABORT,'write failure'); END`); err != nil {
		t.Fatal(err)
	}
	if changed, _, err := d.DeleteGroupFromTool("g", "번역"); err == nil || changed {
		t.Fatal("delete failure reported success", changed, err)
	}
	session, err := d.Session("s")
	if err != nil || session.GroupID != "g" {
		t.Fatal("partial ungrouping escaped rollback", session, err)
	}
}

func TestToolFolderDuplicateNamesAreNotGuessed(t *testing.T) {
	d, err := Open(t.TempDir() + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	d.CreateGroup("one", "번역")
	d.CreateGroup("two", "번역")
	d.CreateGroup("three", "독서")
	if _, _, err = d.CreateGroupFromTool("new", "번역"); err == nil {
		t.Fatal("ambiguous existing folder was guessed")
	}
	if _, err = d.RenameGroupFromTool("three", "독서", "번역"); err == nil {
		t.Fatal("renamed into ambiguous existing name")
	}
	groups, _ := d.Groups()
	if len(groups) != 3 {
		t.Fatal(groups)
	}
}
