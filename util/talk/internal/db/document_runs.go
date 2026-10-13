package db

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sparktalk/internal/workflows"
	"time"
)

func (d *DB) DocumentRun(session, id string) (workflows.Run, error) {
	var raw, status string
	var r workflows.Run
	e := d.conn.QueryRow(`SELECT data,status FROM workflow_runs WHERE id=? AND session_id=?`, id, session).Scan(&raw, &status)
	if e != nil {
		return r, e
	}
	if e = json.Unmarshal([]byte(raw), &r); e != nil {
		return r, e
	}
	if r.Document == nil {
		return r, fmt.Errorf("not a document task")
	}
	r.Status = status
	return r, nil
}
func (d *DB) DocumentUnits(session, id string) ([]workflows.DocumentUnit, error) {
	if _, e := d.DocumentRun(session, id); e != nil {
		return nil, e
	}
	rows, e := d.conn.Query(`SELECT data FROM workflow_units WHERE run_id=? ORDER BY ordinal`, id)
	if e != nil {
		return nil, e
	}
	defer rows.Close()
	out := []workflows.DocumentUnit{}
	for rows.Next() {
		var raw string
		var u workflows.DocumentUnit
		if e = rows.Scan(&raw); e != nil {
			return nil, e
		}
		if e = json.Unmarshal([]byte(raw), &u); e != nil {
			return nil, e
		}
		out = append(out, u)
	}
	return out, rows.Err()
}

// SaveDocumentProgress commits unit results and the run checkpoint atomically.
func (d *DB) SaveDocumentProgress(r *workflows.Run, units ...workflows.DocumentUnit) error {
	return d.saveDocumentProgress(r, false, units...)
}

// SaveDocumentRepair invalidates generated files in the same transaction as
// reopening a translation. Converted documents must never serve an older text.
func (d *DB) SaveDocumentRepair(r *workflows.Run, units ...workflows.DocumentUnit) error {
	return d.saveDocumentProgress(r, true, units...)
}

func (d *DB) saveDocumentProgress(r *workflows.Run, clearOutputs bool, units ...workflows.DocumentUnit) error {
	return d.saveDocumentCheckpoint(r, clearOutputs, nil, units...)
}

func (d *DB) saveDocumentCheckpoint(r *workflows.Run, clearOutputs bool, records *documentCheckpointRecords, units ...workflows.DocumentUnit) error {
	caller := r
	owned := *r
	if r.Document != nil {
		doc := *r.Document
		owned.Document = &doc
	}
	r = &owned
	r.Updated = time.Now().UTC().Format(time.RFC3339Nano)
	raw, e := json.Marshal(r)
	if e != nil {
		return e
	}
	tx, e := d.conn.Begin()
	if e != nil {
		return e
	}
	defer tx.Rollback()
	if e = checkDocumentAttemptInputs(tx, r.ID, records, units); e != nil {
		return e
	}
	query := `UPDATE workflow_runs SET status=?,data=?,updated=? WHERE id=? AND session_id=?`
	args := []any{r.Status, string(raw), r.Updated, r.ID, r.SessionID}
	if r.Document != nil && r.Document.Pipeline != "" {
		var savedRaw string
		if e = tx.QueryRow(`SELECT data FROM workflow_runs WHERE id=? AND session_id=?`, r.ID, r.SessionID).Scan(&savedRaw); e != nil {
			return e
		}
		var saved workflows.Run
		if e = json.Unmarshal([]byte(savedRaw), &saved); e != nil {
			return e
		}
		if saved.Document != nil && (saved.Document.Terminal && !r.Document.Terminal || saved.Document.Publication != nil && r.Document.Publication == nil || saved.Document.DraftPublication != nil && r.Document.DraftPublication == nil && r.Document.Publication == nil) {
			return fmt.Errorf("%w: completed revision cannot be reopened", ErrStaleDocumentExecution)
		}
		query += ` AND json_extract(data,'$.document.execution_id')=? AND (status!='canceled' OR ?='canceled')`
		args = append(args, r.Document.ExecutionID, r.Status)
		if clearOutputs {
			return fmt.Errorf("versioned document repair requires a new revision")
		}
	}
	result, e := tx.Exec(query, args...)
	if e != nil {
		return e
	}
	n, e := result.RowsAffected()
	if e != nil {
		return e
	}
	if n != 1 {
		if r.Document != nil && r.Document.Pipeline != "" {
			return ErrStaleDocumentExecution
		}
		return fmt.Errorf("document run no longer exists")
	}
	if clearOutputs {
		if _, e = tx.Exec(`DELETE FROM workflow_outputs WHERE run_id=?`, r.ID); e != nil {
			return e
		}
	}
	for _, u := range units {
		data, e := json.Marshal(u)
		if e != nil {
			return e
		}
		if _, e = tx.Exec(`INSERT INTO workflow_units(run_id,ordinal,data) VALUES(?,?,?) ON CONFLICT(run_id,ordinal) DO UPDATE SET data=excluded.data`, r.ID, u.Ordinal, string(data)); e != nil {
			return e
		}
	}
	if e = saveDocumentRecords(tx, r.ID, records); e != nil {
		return e
	}
	if r.Document != nil && r.Document.Pipeline != "" && records != nil {
		if e = syncDocumentAccounting(tx, r); e != nil {
			return e
		}
		updated, err := json.Marshal(r)
		if err != nil {
			return err
		}
		if _, e = tx.Exec(`UPDATE workflow_runs SET data=? WHERE id=? AND session_id=?`, string(updated), r.ID, r.SessionID); e != nil {
			return e
		}
	}
	if e = checkDocumentStorage(tx, r.SessionID, 0); e != nil {
		return e
	}
	if e = tx.Commit(); e != nil {
		return e
	}
	*caller = owned
	return nil
}
func (d *DB) ClaimDocumentRun(session, id string) (workflows.Run, error) {
	tx, e := d.conn.Begin()
	if e != nil {
		return workflows.Run{}, e
	}
	defer tx.Rollback()
	var raw, status string
	if e = tx.QueryRow(`SELECT data,status FROM workflow_runs WHERE id=? AND session_id=?`, id, session).Scan(&raw, &status); e != nil {
		return workflows.Run{}, e
	}
	var r workflows.Run
	if e = json.Unmarshal([]byte(raw), &r); e != nil {
		return r, e
	}
	r.Status = status
	if r.Document == nil || status == "running" || status == "canceled" || r.DocumentFinished() {
		return r, fmt.Errorf("task cannot be resumed (%s)", status)
	}
	if r.Document.Pipeline != "" {
		if r.Document.Pipeline != workflows.DocumentPipelineVersion {
			return r, fmt.Errorf("unsupported document pipeline %s", r.Document.Pipeline)
		}
		r.Document.ExecutionID = rand.Text()
	}
	r.Status = "running"
	rawBytes, e := json.Marshal(r)
	if e != nil {
		return r, e
	}
	result, e := tx.Exec(`UPDATE workflow_runs SET status='running',data=? WHERE id=? AND session_id=? AND status=?`, string(rawBytes), id, session, status)
	if e != nil {
		return r, e
	}
	n, _ := result.RowsAffected()
	if n != 1 {
		return r, fmt.Errorf("task already claimed")
	}
	return r, tx.Commit()
}
func (d *DB) SaveDocumentOutput(session, id, name, mime string, data []byte) error {
	if _, e := d.DocumentRun(session, id); e != nil {
		return e
	}
	if len(data) > 96<<20 {
		return fmt.Errorf("output exceeds 96 MiB")
	}
	tx, e := d.conn.Begin()
	if e != nil {
		return e
	}
	defer tx.Rollback()
	var latestRaw string
	if e = tx.QueryRow(`SELECT data FROM workflow_runs WHERE id=? AND session_id=?`, id, session).Scan(&latestRaw); e != nil {
		return e
	}
	var latest workflows.Run
	if e = json.Unmarshal([]byte(latestRaw), &latest); e != nil {
		return e
	}
	if latest.Document != nil && latest.Document.Pipeline != "" {
		manifest := latest.Document.Publication
		if manifest == nil {
			manifest = latest.Document.DraftPublication
		}
		if manifest != nil {
			for _, entry := range manifest.Files {
				if entry.Name != name {
					continue
				}
				hash := sha256.Sum256(data)
				if entry.Hash != hex.EncodeToString(hash[:]) || entry.MIME != mime {
					return fmt.Errorf("published document outputs are immutable; create a new revision")
				}
				return nil
			}
		}
	}
	var old int64
	if e = tx.QueryRow(`SELECT COALESCE((SELECT length(data) FROM workflow_outputs WHERE run_id=? AND name=?),0)`, id, name).Scan(&old); e != nil {
		return e
	}
	if e = checkDocumentStorage(tx, session, int64(len(data))-old); e != nil {
		return e
	}
	if _, e = tx.Exec(`INSERT INTO workflow_outputs(run_id,name,mime,data) VALUES(?,?,?,?) ON CONFLICT(run_id,name) DO UPDATE SET mime=excluded.mime,data=excluded.data`, id, name, mime, data); e != nil {
		return e
	}
	return tx.Commit()
}
func (d *DB) DocumentOutput(session, id, name string) ([]byte, string, error) {
	var data []byte
	var mime string
	e := d.conn.QueryRow(`SELECT o.data,o.mime FROM workflow_outputs o JOIN workflow_runs r ON r.id=o.run_id WHERE r.session_id=? AND o.run_id=? AND o.name=?`, session, id, name).Scan(&data, &mime)
	return data, mime, e
}
func (d *DB) ClearDocumentOutputs(session, id string) error {
	if run, err := d.DocumentRun(session, id); err != nil {
		return err
	} else if run.Document.Pipeline != "" {
		return fmt.Errorf("versioned document outputs require a new revision")
	}
	_, e := d.conn.Exec(`DELETE FROM workflow_outputs WHERE run_id IN(SELECT id FROM workflow_runs WHERE id=? AND session_id=?)`, id, session)
	return e
}

func (d *DB) DeleteDocumentRun(session, id string) error {
	if _, e := d.DocumentRun(session, id); e != nil {
		return e
	}
	_, e := d.conn.Exec(`DELETE FROM workflow_runs WHERE id=? AND session_id=?`, id, session)
	return e
}

func (d *DB) DocumentRequest(session, key string) (workflows.Run, error) {
	var id string
	e := d.conn.QueryRow(`SELECT id FROM workflow_runs WHERE session_id=? AND json_extract(data,'$.document.request_id')=?`, session, key).Scan(&id)
	if e != nil {
		return workflows.Run{}, e
	}
	return d.DocumentRun(session, id)
}
func (d *DB) CreateDocumentRun(r *workflows.Run, units []workflows.DocumentUnit) error {
	r.Updated = time.Now().UTC().Format(time.RFC3339Nano)
	raw, e := json.Marshal(r)
	if e != nil {
		return e
	}
	tx, e := d.conn.Begin()
	if e != nil {
		return e
	}
	defer tx.Rollback()
	for _, id := range r.Document.Sources {
		var n int
		if e = tx.QueryRow(`SELECT 1 FROM source_snapshots WHERE id=? AND session_id=?`, id, r.SessionID).Scan(&n); e != nil {
			return e
		}
	}
	if _, e = tx.Exec(`INSERT INTO workflow_runs(id,session_id,status,data,updated) VALUES(?,?,?,?,?)`, r.ID, r.SessionID, r.Status, string(raw), r.Updated); e != nil {
		return e
	}
	for _, u := range units {
		data, e := json.Marshal(u)
		if e != nil {
			return e
		}
		if _, e = tx.Exec(`INSERT INTO workflow_units(run_id,ordinal,data) VALUES(?,?,?)`, r.ID, u.Ordinal, string(data)); e != nil {
			return e
		}
	}
	if e = checkDocumentStorage(tx, r.SessionID, 0); e != nil {
		return e
	}
	return tx.Commit()
}
