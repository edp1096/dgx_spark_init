package db

import (
	"database/sql"
	"errors"
	"path/filepath"
	"reflect"
	"sync"
	"testing"
)

func TestArtifactAtomicVersionsAndPersistence(t *testing.T) {
	path := filepath.Join(t.TempDir(), "code.db")
	d, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { d.Close() }()
	d.CreateSession("s", "code", "", "")
	d.CreateSession("other", "other", "", "")
	files := []ArtifactFile{{"index.html", "<h1>first</h1>"}, {"style.css", "h1 {color:red}"}}
	a, err := d.CreateArtifact("s", "project", "Tetris", "message:1", files)
	if err != nil {
		t.Fatal(err)
	}
	same, err := d.CreateArtifact("s", "duplicate", "Tetris", "message:1", files)
	if err != nil || same.ID != a.ID {
		t.Fatalf("import not idempotent: %+v %v", same, err)
	}
	if _, err = d.Artifact("other", a.ID, 0); !errors.Is(err, sql.ErrNoRows) {
		t.Fatalf("cross-session read: %v", err)
	}
	if _, err = d.EditArtifact("other", a.ID, 1, "", []ArtifactEdit{{Name: "index.html", Operation: "delete"}}, 0); !errors.Is(err, sql.ErrNoRows) {
		t.Fatalf("cross-session edit: %v", err)
	}
	edits := []ArtifactEdit{{Name: "index.html", Operation: "replace", Old: "first", New: "second"}, {Name: "script.js", Operation: "create", New: "let score = 0;"}}
	a, err = d.EditArtifact("s", a.ID, 1, "제목 수정 및 스크립트 추가", edits, 0)
	if err != nil || a.Version != 2 {
		t.Fatalf("edit: %+v %v", a, err)
	}
	if a.Files[1] != files[1] {
		t.Fatal("unmodified file changed")
	}
	var blobs int
	d.conn.QueryRow(`SELECT count(*) FROM code_blobs`).Scan(&blobs)
	if blobs != 4 {
		t.Fatalf("unchanged file was duplicated: %d blobs", blobs)
	}
	// The first valid edit must also roll back if a later replacement fails.
	bad := []ArtifactEdit{{Name: "index.html", Operation: "replace", Old: "second", New: "third"}, {Name: "style.css", Operation: "replace", Old: "absent", New: "blue"}}
	if _, err = d.EditArtifact("s", a.ID, 2, "bad", bad, 0); err == nil {
		t.Fatal("invalid patch succeeded")
	}
	current, _ := d.Artifact("s", a.ID, 0)
	if !reflect.DeepEqual(current, a) {
		t.Fatalf("partial mutation survived: %+v", current)
	}
	if _, err = d.EditArtifact("s", a.ID, 1, "stale", edits, 0); !errors.Is(err, ErrArtifactConflict) {
		t.Fatalf("stale patch: %v", err)
	}
	restored, err := d.EditArtifact("s", a.ID, 2, "", nil, 1)
	if err != nil || restored.Version != 3 || !reflect.DeepEqual(restored.Files, files) {
		t.Fatalf("restore: %+v %v", restored, err)
	}
	old, _ := d.Artifact("s", a.ID, 2)
	if !reflect.DeepEqual(old, a) {
		t.Fatal("history rewritten")
	}
	d.Close()
	d, err = Open(path)
	if err != nil {
		t.Fatal(err)
	}
	current, err = d.Artifact("s", a.ID, 0)
	if err != nil || !reflect.DeepEqual(current, restored) {
		t.Fatalf("restart: %+v %v", current, err)
	}
	revisions, _ := d.ArtifactVersions("s", a.ID)
	if len(revisions) != 3 || revisions[0].Version != 3 {
		t.Fatalf("history: %+v", revisions)
	}
	if err = d.DeleteSession("s"); err != nil {
		t.Fatal(err)
	}
	for _, table := range []string{"code_projects", "code_revisions", "code_blobs"} {
		var n int
		d.conn.QueryRow("SELECT count(*) FROM " + table).Scan(&n)
		if n != 0 {
			t.Fatalf("orphaned %s: %d", table, n)
		}
	}
}
func TestArtifactConcurrentEditsAndAmbiguousPatch(t *testing.T) {
	d, err := Open(filepath.Join(t.TempDir(), "code.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	d.CreateSession("s", "code", "", "")
	a, err := d.CreateArtifact("s", "p", "Code", "", []ArtifactFile{{"main.js", "red red"}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = d.EditArtifact("s", a.ID, 1, "", []ArtifactEdit{{Name: "main.js", Operation: "replace", Old: "red", New: "blue"}}, 0); err == nil {
		t.Fatal("ambiguous replacement succeeded")
	}
	// Overlapping occurrences are ambiguous too ("aa" occurs twice in "aaa").
	overlapping, err := d.CreateArtifact("s", "overlap", "Overlap", "", []ArtifactFile{{"main.js", "aaa"}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = d.EditArtifact("s", overlapping.ID, 1, "", []ArtifactEdit{{Name: "main.js", Operation: "replace", Old: "aa", New: "b"}}, 0); err == nil {
		t.Fatal("overlapping replacement succeeded")
	}
	var wg sync.WaitGroup
	errs := make(chan error, 2)
	for _, newText := range []string{"blue red", "green red"} {
		wg.Add(1)
		go func(text string) {
			defer wg.Done()
			_, e := d.EditArtifact("s", a.ID, 1, "change", []ArtifactEdit{{Name: "main.js", Operation: "replace", Old: "red red", New: text}}, 0)
			errs <- e
		}(newText)
	}
	wg.Wait()
	close(errs)
	success, conflict := 0, 0
	for e := range errs {
		if e == nil {
			success++
		} else if errors.Is(e, ErrArtifactConflict) {
			conflict++
		} else {
			t.Fatal(e)
		}
	}
	if success != 1 || conflict != 1 {
		t.Fatalf("concurrent writers: success %d conflicts %d", success, conflict)
	}
	if _, err = d.EditArtifact("s", a.ID, 2, "", []ArtifactEdit{{Name: "../escape", Operation: "create", New: "x"}}, 0); err == nil {
		t.Fatal("invalid name accepted")
	}
}

func TestArtifactWriteRepairsLegacyEmptyFile(t *testing.T) {
	d, err := Open(filepath.Join(t.TempDir(), "code.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	d.CreateSession("s", "test", "", "")
	_, err = d.CreateArtifact("s", "p", "Game", "", []ArtifactFile{{Name: "index.html", Source: "<h1>old</h1>"}, {Name: "style.css", Source: ""}})
	if err != nil {
		t.Fatal(err)
	}
	// Simulate the corrupt empty HTML saved by the old release.
	_, err = d.conn.Exec(`UPDATE code_blobs SET content='' WHERE project_id='p'`)
	if err != nil {
		t.Fatal(err)
	}
	a, err := d.EditArtifact("s", "p", 1, "recovery", []ArtifactEdit{{Name: "index.html", Operation: "write", New: "<h1>recovered</h1>"}}, 0)
	if err != nil || a.Version != 2 || a.Files[0].Source != "<h1>recovered</h1>" {
		t.Fatalf("cannot repair empty file: %+v %v", a, err)
	}
	if _, err = d.EditArtifact("s", "p", 2, "empty", []ArtifactEdit{{Name: "index.html", Operation: "write", New: ""}}, 0); err == nil {
		t.Fatal("empty HTML accepted")
	}
	if _, err = d.EditArtifact("s", "p", 2, "", nil, 1); err == nil {
		t.Fatal("corrupt empty version restored")
	}
	a, _ = d.Artifact("s", "p", 0)
	if a.Version != 2 || a.Files[1].Source != "" {
		t.Fatalf("unexpected modification: %+v", a)
	}
}
