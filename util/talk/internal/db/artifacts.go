package db

import (
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"time"
	"unicode/utf8"
)

var ErrArtifactConflict = errors.New("code project changed; read the latest version before editing")

const MaxArtifactBytes = 2 << 20

type ArtifactFile struct {
	Name   string `json:"name"`
	Source string `json:"source"`
}
type Artifact struct {
	ID        string         `json:"id"`
	SessionID string         `json:"session_id"`
	Title     string         `json:"title"`
	Version   int            `json:"version"`
	SourceKey string         `json:"source_key,omitempty"`
	Files     []ArtifactFile `json:"files,omitempty"`
	Summary   string         `json:"summary"`
	CreatedAt string         `json:"created_at"`
}
type ArtifactEdit struct {
	Name      string `json:"name"`
	Operation string `json:"operation"`
	Old       string `json:"old,omitempty"`
	New       string `json:"new,omitempty"`
}

func migrateArtifacts(conn *sql.DB) error {
	_, err := conn.Exec(`
 CREATE TABLE IF NOT EXISTS code_projects(id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, title TEXT NOT NULL, head INTEGER NOT NULL, source_key TEXT NOT NULL DEFAULT '', UNIQUE(session_id,source_key));
 CREATE TABLE IF NOT EXISTS code_revisions(project_id TEXT NOT NULL REFERENCES code_projects(id) ON DELETE CASCADE, version INTEGER NOT NULL, files TEXT NOT NULL, summary TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(project_id,version));
 CREATE TABLE IF NOT EXISTS code_blobs(project_id TEXT NOT NULL REFERENCES code_projects(id) ON DELETE CASCADE, hash TEXT NOT NULL, content TEXT NOT NULL, PRIMARY KEY(project_id,hash));
 CREATE TRIGGER IF NOT EXISTS code_project_delete AFTER DELETE ON code_projects BEGIN DELETE FROM code_revisions WHERE project_id=OLD.id; DELETE FROM code_blobs WHERE project_id=OLD.id; END;
 CREATE TRIGGER IF NOT EXISTS code_session_delete AFTER DELETE ON sessions BEGIN DELETE FROM code_projects WHERE session_id=OLD.id; END;
 `)
	return err
}
func validateArtifactFiles(files []ArtifactFile) error {
	if len(files) == 0 || len(files) > 32 {
		return fmt.Errorf("a code project needs 1 to 32 files")
	}
	total := 0
	names := map[string]bool{}
	for _, f := range files {
		if f.Name == "" || len(f.Name) > 160 || strings.ContainsAny(f.Name, "/\\\x00\r\n") || f.Name == "." || f.Name == ".." || strings.TrimSpace(f.Name) != f.Name || !utf8.ValidString(f.Name) {
			return fmt.Errorf("invalid flat file name: %q", f.Name)
		}
		if names[f.Name] {
			return fmt.Errorf("duplicate file: %s", f.Name)
		}
		names[f.Name] = true
		ext := strings.ToLower(f.Name)
		if (strings.HasSuffix(ext, ".html") || strings.HasSuffix(ext, ".htm") || strings.HasSuffix(ext, ".svg")) && strings.TrimSpace(f.Source) == "" {
			return fmt.Errorf("web entry %s cannot be empty; nothing saved, previous version retained", f.Name)
		}
		total += len(f.Source)
		if !utf8.ValidString(f.Source) || total > MaxArtifactBytes {
			return fmt.Errorf("code files must be UTF-8 text totalling at most 2 MiB")
		}
	}
	return nil
}
func (d *DB) Artifacts(session string) ([]Artifact, error) {
	rows, err := d.conn.Query(`SELECT p.id,p.session_id,p.title,p.head,p.source_key,r.summary,r.created_at FROM code_projects p JOIN code_revisions r ON r.project_id=p.id AND r.version=p.head WHERE p.session_id=? ORDER BY r.created_at,p.id`, session)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Artifact{}
	for rows.Next() {
		var a Artifact
		if err = rows.Scan(&a.ID, &a.SessionID, &a.Title, &a.Version, &a.SourceKey, &a.Summary, &a.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, a)
	}
	return out, rows.Err()
}
func artifactInTx(tx *sql.Tx, session, id string, version int) (Artifact, error) {
	var a Artifact
	err := tx.QueryRow(`SELECT id,session_id,title,head,source_key FROM code_projects WHERE id=? AND session_id=?`, id, session).Scan(&a.ID, &a.SessionID, &a.Title, &a.Version, &a.SourceKey)
	if err != nil {
		return a, err
	}
	if version > 0 {
		a.Version = version
	}
	var manifest string
	err = tx.QueryRow(`SELECT files,summary,created_at FROM code_revisions WHERE project_id=? AND version=?`, id, a.Version).Scan(&manifest, &a.Summary, &a.CreatedAt)
	if err != nil {
		return a, err
	}
	var refs []ArtifactFile
	if err = json.Unmarshal([]byte(manifest), &refs); err != nil {
		return a, err
	}
	for _, ref := range refs {
		var content string
		if err = tx.QueryRow(`SELECT content FROM code_blobs WHERE project_id=? AND hash=?`, id, ref.Source).Scan(&content); err != nil {
			return a, err
		}
		a.Files = append(a.Files, ArtifactFile{Name: ref.Name, Source: content})
	}
	return a, nil
}
func (d *DB) Artifact(session, id string, version int) (Artifact, error) {
	tx, err := d.conn.Begin()
	if err != nil {
		return Artifact{}, err
	}
	defer tx.Rollback()
	return artifactInTx(tx, session, id, version)
}
func saveArtifactRevision(tx *sql.Tx, a *Artifact) error {
	if err := validateArtifactFiles(a.Files); err != nil {
		return err
	}
	if len(a.Summary) > 2000 {
		return fmt.Errorf("summary too long")
	}
	refs := make([]ArtifactFile, 0, len(a.Files))
	for _, f := range a.Files {
		sum := sha256.Sum256([]byte(f.Source))
		hash := hex.EncodeToString(sum[:])
		if _, err := tx.Exec(`INSERT OR IGNORE INTO code_blobs(project_id,hash,content) VALUES(?,?,?)`, a.ID, hash, f.Source); err != nil {
			return err
		}
		refs = append(refs, ArtifactFile{Name: f.Name, Source: hash})
	}
	raw, _ := json.Marshal(refs)
	a.CreatedAt = time.Now().UTC().Format(time.RFC3339Nano)
	_, err := tx.Exec(`INSERT INTO code_revisions(project_id,version,files,summary,created_at) VALUES(?,?,?,?,?)`, a.ID, a.Version, string(raw), a.Summary, a.CreatedAt)
	return err
}
func (d *DB) CreateArtifact(session, id, title, sourceKey string, files []ArtifactFile) (Artifact, error) {
	d.artifactMu.Lock()
	defer d.artifactMu.Unlock()
	if strings.TrimSpace(title) == "" || utf8.RuneCountInString(title) > 120 {
		return Artifact{}, fmt.Errorf("title needs 1 to 120 characters")
	}
	if sourceKey == "" {
		sourceKey = "project:" + id
	}
	tx, err := d.conn.Begin()
	if err != nil {
		return Artifact{}, err
	}
	defer tx.Rollback()
	var existing string
	err = tx.QueryRow(`SELECT id FROM code_projects WHERE session_id=? AND source_key=?`, session, sourceKey).Scan(&existing)
	if err == nil {
		return artifactInTx(tx, session, existing, 0)
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return Artifact{}, err
	}
	var found string
	if err = tx.QueryRow(`SELECT id FROM sessions WHERE id=?`, session).Scan(&found); err != nil {
		return Artifact{}, err
	}
	a := Artifact{ID: id, SessionID: session, Title: strings.TrimSpace(title), Version: 1, SourceKey: sourceKey, Files: files, Summary: "최초 저장"}
	if _, err = tx.Exec(`INSERT INTO code_projects(id,session_id,title,head,source_key) VALUES(?,?,?,?,?)`, id, session, a.Title, 1, sourceKey); err != nil {
		return Artifact{}, err
	}
	if err = saveArtifactRevision(tx, &a); err != nil {
		return Artifact{}, err
	}
	return a, tx.Commit()
}
func (d *DB) ArtifactVersions(session, id string) ([]Artifact, error) {
	rows, err := d.conn.Query(`SELECT r.version,r.summary,r.created_at FROM code_revisions r JOIN code_projects p ON p.id=r.project_id WHERE p.session_id=? AND p.id=? ORDER BY r.version DESC`, session, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Artifact{}
	for rows.Next() {
		a := Artifact{ID: id, SessionID: session}
		if err = rows.Scan(&a.Version, &a.Summary, &a.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, a)
	}
	return out, rows.Err()
}
func (d *DB) EditArtifact(session, id string, base int, summary string, edits []ArtifactEdit, restore int) (Artifact, error) {
	d.artifactMu.Lock()
	defer d.artifactMu.Unlock()
	tx, err := d.conn.Begin()
	if err != nil {
		return Artifact{}, err
	}
	defer tx.Rollback()
	a, err := artifactInTx(tx, session, id, 0)
	if err != nil {
		return a, err
	}
	if base < 1 || base != a.Version {
		return a, ErrArtifactConflict
	}
	beforeFiles := append([]ArtifactFile(nil), a.Files...)
	if restore > 0 {
		if len(edits) > 0 {
			return a, fmt.Errorf("restore and edits cannot be combined")
		}
		old, e := artifactInTx(tx, session, id, restore)
		if e != nil {
			return a, e
		}
		a.Files = old.Files
		summary = fmt.Sprintf("버전 %d 복원", restore)
	} else {
		if len(edits) == 0 || len(edits) > 128 {
			return a, fmt.Errorf("provide 1 to 128 edits")
		}
		for _, edit := range edits {
			index := -1
			for i, f := range a.Files {
				if f.Name == edit.Name {
					index = i
					break
				}
			}
			switch edit.Operation {
			case "create":
				if index >= 0 {
					return a, fmt.Errorf("file already exists: %s", edit.Name)
				}
				a.Files = append(a.Files, ArtifactFile{Name: edit.Name, Source: edit.New})
			case "write":
				if index < 0 {
					return a, fmt.Errorf("file not found: %s; use create for new files", edit.Name)
				}
				a.Files[index].Source = edit.New
			case "delete":
				if index < 0 {
					return a, fmt.Errorf("file not found: %s", edit.Name)
				}
				a.Files = append(a.Files[:index], a.Files[index+1:]...)
			case "replace":
				if index < 0 || edit.Old == "" {
					return a, fmt.Errorf("replace requires an existing file and nonempty old text; use write with source for an empty file or full rewrite")
				}
				if first := strings.Index(a.Files[index].Source, edit.Old); first < 0 || first != strings.LastIndex(a.Files[index].Source, edit.Old) {
					return a, fmt.Errorf("old text must match exactly once in %s; read current code and include more context", edit.Name)
				}
				a.Files[index].Source = strings.Replace(a.Files[index].Source, edit.Old, edit.New, 1)
			default:
				return a, fmt.Errorf("operation must be create, write, replace or delete")
			}
		}
	}
	if reflect.DeepEqual(beforeFiles, a.Files) {
		return a, fmt.Errorf("no file content changed; no new version saved")
	}
	a.Version++
	a.Summary = summary
	if err = saveArtifactRevision(tx, &a); err != nil {
		return a, err
	}
	result, err := tx.Exec(`UPDATE code_projects SET head=? WHERE id=? AND head=?`, a.Version, id, base)
	if err != nil {
		return a, err
	}
	n, _ := result.RowsAffected()
	if n != 1 {
		return a, ErrArtifactConflict
	}
	return a, tx.Commit()
}
