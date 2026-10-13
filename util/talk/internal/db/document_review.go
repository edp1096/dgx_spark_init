package db

import (
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"sparktalk/internal/performance"
	"sparktalk/internal/workflows"
)

var ErrStaleDocumentExecution = errors.New("document checkpoint rejected: stale execution or canceled task")

func (d *DB) initDocumentReviewStorage() error {
	_, err := d.conn.Exec(`
CREATE TABLE IF NOT EXISTS document_attempts(run_id TEXT NOT NULL REFERENCES workflow_runs(id) ON DELETE CASCADE,id TEXT NOT NULL,data TEXT NOT NULL,PRIMARY KEY(run_id,id));
CREATE TABLE IF NOT EXISTS document_reviews(run_id TEXT NOT NULL REFERENCES workflow_runs(id) ON DELETE CASCADE,id TEXT NOT NULL,data TEXT NOT NULL,PRIMARY KEY(run_id,id));
CREATE TABLE IF NOT EXISTS document_entities(run_id TEXT NOT NULL REFERENCES workflow_runs(id) ON DELETE CASCADE,id TEXT NOT NULL,version INTEGER NOT NULL,data TEXT NOT NULL,PRIMARY KEY(run_id,id,version));`)
	return err
}

type documentCheckpointRecords struct {
	attempt  *workflows.DocumentRequestAttempt
	review   *workflows.DocumentReview
	entities []workflows.DocumentEntity
}

func checkDocumentAttemptInputs(tx *sql.Tx, run string, records *documentCheckpointRecords, units []workflows.DocumentUnit) error {
	if records == nil || records.attempt == nil {
		return nil
	}
	var exists int
	if err := tx.QueryRow(`SELECT COUNT(*) FROM document_attempts WHERE run_id=? AND id=?`, run, records.attempt.ID).Scan(&exists); err != nil {
		return err
	}
	// Identical request records may be committed again after a lost acknowledgement.
	// saveDocumentRecords still rejects a different payload under the same ID.
	if exists > 0 {
		for _, unit := range units {
			var saved string
			if err := tx.QueryRow(`SELECT data FROM workflow_units WHERE run_id=? AND ordinal=?`, run, unit.Ordinal).Scan(&saved); err != nil {
				return err
			}
			incoming, err := json.Marshal(unit)
			if err != nil {
				return err
			}
			if saved != string(incoming) {
				return fmt.Errorf("%w: repeated request would overwrite a newer checkpoint", ErrStaleDocumentExecution)
			}
		}
		return nil
	}
	for ordinal, expected := range records.attempt.ExpectedUnitOutputHashes {
		var raw string
		if err := tx.QueryRow(`SELECT data FROM workflow_units WHERE run_id=? AND ordinal=?`, run, ordinal).Scan(&raw); err != nil {
			return err
		}
		var unit workflows.DocumentUnit
		if err := json.Unmarshal([]byte(raw), &unit); err != nil {
			return err
		}
		output, err := json.Marshal(unit.Output)
		if err != nil {
			return err
		}
		hash := sha256.Sum256(output)
		if hex.EncodeToString(hash[:]) != expected {
			return fmt.Errorf("%w: paragraph output changed during model request", ErrStaleDocumentExecution)
		}
	}
	return nil
}

// Derive canonical usage from immutable request records, not from repeated
// checkpoint commits or one copy of a batch attempt per paragraph.
func syncDocumentAccounting(tx *sql.Tx, run *workflows.Run) error {
	rows, err := tx.Query(`SELECT json_extract(data,'$.stage'),json_extract(data,'$.measurement') FROM document_attempts WHERE run_id=? ORDER BY rowid`, run.ID)
	if err != nil {
		return err
	}
	defer rows.Close()
	run.Document.Calls, run.Document.OutputTokens = 0, 0
	run.Document.StageMetrics = map[string]workflows.DocumentStageMetric{}
	measurements := map[string][]*performance.Summary{}
	for rows.Next() {
		var stage, raw string
		if err = rows.Scan(&stage, &raw); err != nil {
			return err
		}
		var attempt workflows.DocumentAttempt
		if err = json.Unmarshal([]byte(raw), &attempt); err != nil {
			return err
		}
		metric := run.Document.StageMetrics[stage]
		metric.Calls++
		metric.Seconds += attempt.Seconds
		metric.PromptTokens += attempt.PromptTokens
		metric.OutputTokens += attempt.OutputTokens
		run.Document.StageMetrics[stage] = metric
		measurements[stage] = append(measurements[stage], attempt.Performance)
		run.Document.Calls++
		run.Document.OutputTokens += attempt.OutputTokens
	}
	for stage, samples := range measurements {
		metric := run.Document.StageMetrics[stage]
		metric.Performance = performance.Combine(samples)
		run.Document.StageMetrics[stage] = metric
	}
	return rows.Err()
}

func (d *DB) SaveDocumentReviewProgress(run *workflows.Run, attempt *workflows.DocumentRequestAttempt, review *workflows.DocumentReview, entities []workflows.DocumentEntity, units ...workflows.DocumentUnit) error {
	return d.saveDocumentCheckpoint(run, false, &documentCheckpointRecords{attempt, review, entities}, units...)
}

func saveDocumentRecords(tx *sql.Tx, run string, records *documentCheckpointRecords) error {
	if records == nil {
		return nil
	}
	insert := func(table, id string, value any) error {
		if id == "" {
			return fmt.Errorf("empty document record id")
		}
		raw, err := json.Marshal(value)
		if err != nil {
			return err
		}
		// IDs are immutable; a repeated identical commit is safe, a different
		// payload under the same ID must not rewrite usage or review history.
		var prior string
		err = tx.QueryRow("SELECT data FROM "+table+" WHERE run_id=? AND id=?", run, id).Scan(&prior)
		if err == nil {
			if prior != string(raw) {
				return fmt.Errorf("conflicting document record %s", id)
			}
			return nil
		}
		if err != sql.ErrNoRows {
			return err
		}
		_, err = tx.Exec("INSERT INTO "+table+"(run_id,id,data) VALUES(?,?,?)", run, id, string(raw))
		return err
	}
	if records.attempt != nil {
		if err := insert("document_attempts", records.attempt.ID, records.attempt); err != nil {
			return err
		}
	}
	if records.review != nil {
		if err := insert("document_reviews", records.review.ID, records.review); err != nil {
			return err
		}
	}
	for _, entity := range records.entities {
		if entity.ID == "" || entity.Version < 1 {
			return fmt.Errorf("invalid entity version")
		}
		raw, err := json.Marshal(entity)
		if err != nil {
			return err
		}
		var prior string
		err = tx.QueryRow(`SELECT data FROM document_entities WHERE run_id=? AND id=? AND version=?`, run, entity.ID, entity.Version).Scan(&prior)
		if err == nil {
			if prior != string(raw) {
				return fmt.Errorf("conflicting entity version")
			}
			continue
		}
		if err != sql.ErrNoRows {
			return err
		}
		if _, err = tx.Exec(`INSERT INTO document_entities(run_id,id,version,data) VALUES(?,?,?,?)`, run, entity.ID, entity.Version, string(raw)); err != nil {
			return err
		}
	}
	return nil
}

func (d *DB) DocumentEntities(session, run string) ([]workflows.DocumentEntity, error) {
	if _, err := d.DocumentRun(session, run); err != nil {
		return nil, err
	}
	rows, err := d.conn.Query(`SELECT e.data FROM document_entities e WHERE e.run_id=? AND e.version=(SELECT MAX(n.version) FROM document_entities n WHERE n.run_id=e.run_id AND n.id=e.id) ORDER BY e.id`, run)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []workflows.DocumentEntity
	for rows.Next() {
		var raw string
		if err = rows.Scan(&raw); err != nil {
			return nil, err
		}
		var entity workflows.DocumentEntity
		if err = json.Unmarshal([]byte(raw), &entity); err != nil {
			return nil, err
		}
		out = append(out, entity)
	}
	return out, rows.Err()
}

func (d *DB) DocumentReviews(session, run string) ([]workflows.DocumentReview, error) {
	if _, err := d.DocumentRun(session, run); err != nil {
		return nil, err
	}
	rows, err := d.conn.Query(`SELECT data FROM document_reviews WHERE run_id=? ORDER BY id`, run)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []workflows.DocumentReview
	for rows.Next() {
		var raw string
		if err = rows.Scan(&raw); err != nil {
			return nil, err
		}
		var review workflows.DocumentReview
		if err = json.Unmarshal([]byte(raw), &review); err != nil {
			return nil, err
		}
		out = append(out, review)
	}
	return out, rows.Err()
}

// Publish all requested files as one revision. Bytes may be prepared in earlier
// transactions, but readers only advertise this committed manifest.
func (d *DB) PublishDocumentRevision(run *workflows.Run, names []string, textHash string) error {
	return d.publishDocumentManifest(run, names, textHash, true)
}

// A verified base-file manifest is a downloadable draft while a requested
// conversion remains failed. It never makes the task terminal or complete.
func (d *DB) PublishDocumentDraft(run *workflows.Run, names []string, textHash string) error {
	return d.publishDocumentManifest(run, names, textHash, false)
}

func (d *DB) publishDocumentManifest(run *workflows.Run, names []string, textHash string, final bool) error {
	if run.Document == nil || run.Document.Pipeline == "" || run.Document.ExecutionID == "" {
		return fmt.Errorf("not a versioned document run")
	}
	tx, err := d.conn.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	publication := &workflows.DocumentPublication{TextHash: textHash, PublishedAt: time.Now().UTC().Format(time.RFC3339Nano)}
	seen := map[string]bool{}
	for _, name := range names {
		if seen[name] {
			continue
		}
		seen[name] = true
		var mime string
		var raw []byte
		if err = tx.QueryRow(`SELECT mime,data FROM workflow_outputs WHERE run_id=? AND name=?`, run.ID, name).Scan(&mime, &raw); err != nil {
			return err
		}
		if len(raw) == 0 {
			return fmt.Errorf("empty document output %s", name)
		}
		h := sha256.Sum256(raw)
		publication.Files = append(publication.Files, workflows.DocumentManifestEntry{Name: name, MIME: mime, Hash: hex.EncodeToString(h[:]), Bytes: len(raw)})
	}
	if len(publication.Files) == 0 {
		return fmt.Errorf("empty document manifest")
	}
	owned := *run
	doc := *run.Document
	owned.Document = &doc
	if final {
		doc.Publication = publication
		doc.DraftPublication = nil
	} else {
		if doc.Terminal || doc.Publication != nil {
			return fmt.Errorf("completed document cannot become a draft")
		}
		doc.DraftPublication = publication
	}
	owned.Updated = time.Now().UTC().Format(time.RFC3339Nano)
	raw, err := json.Marshal(owned)
	if err != nil {
		return err
	}
	result, err := tx.Exec(`UPDATE workflow_runs SET data=?,status=?,updated=? WHERE id=? AND session_id=? AND json_extract(data,'$.document.execution_id')=? AND status!='canceled'`, string(raw), owned.Status, owned.Updated, run.ID, run.SessionID, doc.ExecutionID)
	if err != nil {
		return err
	}
	n, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if n != 1 {
		return fmt.Errorf("stale or canceled document publisher")
	}
	if err = checkDocumentStorage(tx, run.SessionID, 0); err != nil {
		return err
	}
	if err = tx.Commit(); err != nil {
		return err
	}
	*run = owned
	return nil
}
