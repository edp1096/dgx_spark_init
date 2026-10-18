package db

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"

	"sparktalk/internal/workflows"
)

func documentContentHash(data []byte) string {
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:])
}

func (d *DB) initDocumentConversionStorage() error {
	_, err := d.conn.Exec(`CREATE TABLE IF NOT EXISTS document_conversions(
 id TEXT PRIMARY KEY,run_id TEXT NOT NULL REFERENCES workflow_runs(id) ON DELETE CASCADE,
 format TEXT NOT NULL,status TEXT NOT NULL,data TEXT NOT NULL,content BLOB);
 CREATE INDEX IF NOT EXISTS document_conversions_run ON document_conversions(run_id);
 CREATE UNIQUE INDEX IF NOT EXISTS document_conversion_active ON document_conversions((1)) WHERE status='running';
 UPDATE document_conversions SET status='failed',data=json_set(data,'$.status','failed','$.error','서버 재시작으로 변환이 중단됐습니다. 기존 파일은 보존되어 있습니다. 다시 변환해 주세요.','$.finished_at',strftime('%Y-%m-%dT%H:%M:%fZ','now')) WHERE status='running';
 DELETE FROM document_conversions
 WHERE rowid NOT IN (SELECT MAX(rowid) FROM document_conversions GROUP BY run_id,format)
 AND rowid NOT IN (SELECT MAX(rowid) FROM document_conversions WHERE status='completed' GROUP BY run_id,format);`)
	return err
}

func conversionSource(tx *sql.Tx, session, id string) ([]byte, error) {
	var raw, status string
	if err := tx.QueryRow(`SELECT data,status FROM workflow_runs WHERE id=? AND session_id=?`, id, session).Scan(&raw, &status); err != nil {
		return nil, err
	}
	var run workflows.Run
	if err := json.Unmarshal([]byte(raw), &run); err != nil {
		return nil, err
	}
	if run.Document == nil || run.Current < 2 || status == "running" || status == "canceled" {
		return nil, fmt.Errorf("완료된 본문이 있는 비활성 작업만 재변환할 수 있습니다")
	}
	if run.Document.Pipeline != "" && run.Document.Publication == nil {
		return nil, fmt.Errorf("문서 작업을 완료한 뒤 재변환해 주세요")
	}
	var body []byte
	err := tx.QueryRow(`SELECT data FROM workflow_outputs WHERE run_id=? AND name='output.txt'`, id).Scan(&body)
	return body, err
}

func (d *DB) BeginDocumentConversion(session, run, format, converter string) (workflows.DocumentConversion, []byte, error) {
	j := workflows.DocumentConversion{ID: rand.Text(), Format: format, Status: "running", StartedAt: time.Now().UTC().Format(time.RFC3339Nano), Converter: converter}
	if format != "pdf" && format != "docx" && format != "hwp" && format != "hwpx" {
		return j, nil, fmt.Errorf("unsupported document format")
	}
	tx, err := d.conn.Begin()
	if err != nil {
		return j, nil, err
	}
	defer tx.Rollback()
	body, err := conversionSource(tx, session, run)
	if err != nil {
		return j, nil, err
	}
	var exists int
	if err = tx.QueryRow(`SELECT COUNT(*) FROM workflow_outputs WHERE run_id=? AND name=?`, run, "output."+format).Scan(&exists); err != nil {
		return j, nil, err
	}
	if exists == 0 {
		return j, nil, fmt.Errorf("아직 생성하지 않은 형식입니다. 먼저 파일 만들기를 실행해 주세요")
	}
	j.InputHash = documentContentHash(body)
	raw, err := json.Marshal(j)
	if err != nil {
		return j, nil, err
	}
	// Retain only the usable file and the current attempt, never an attempt history.
	if _, err = tx.Exec(`DELETE FROM document_conversions WHERE run_id=? AND format=? AND status='failed'`, run, format); err != nil {
		return j, nil, err
	}
	if err = checkDocumentStorage(tx, session, int64(len(raw))); err != nil {
		return j, nil, err
	}
	if _, err = tx.Exec(`INSERT INTO document_conversions(id,run_id,format,status,data) VALUES(?,?,?,?,?)`, j.ID, run, format, j.Status, string(raw)); err != nil {
		return j, nil, err
	}
	return j, body, tx.Commit()
}

// Replace the current converted file and successful status together. Original outputs,
// translation metrics and publication hashes are never updated by conversion.
func (d *DB) CompleteDocumentConversion(session, run, id, mime string, content []byte) error {
	if len(content) == 0 || len(content) > 24<<20 {
		return fmt.Errorf("invalid converted document size")
	}
	tx, err := d.conn.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	var raw string
	if err = tx.QueryRow(`SELECT c.data FROM document_conversions c JOIN workflow_runs r ON r.id=c.run_id WHERE r.session_id=? AND c.run_id=? AND c.id=? AND c.status='running'`, session, run, id).Scan(&raw); err != nil {
		return err
	}
	var j workflows.DocumentConversion
	if err = json.Unmarshal([]byte(raw), &j); err != nil {
		return err
	}
	body, err := conversionSource(tx, session, run)
	if err != nil {
		return err
	}
	if j.InputHash != documentContentHash(body) {
		return fmt.Errorf("변환 도중 저장된 본문이 변경됐습니다. 기존 파일을 유지합니다")
	}
	j.Status = "completed"
	j.MIME = mime
	j.Bytes = len(content)
	j.Hash = documentContentHash(content)
	finishDocumentConversion(&j)
	updated, err := json.Marshal(j)
	if err != nil {
		return err
	}
	// Removal is part of the commit: a failed write restores the previous file.
	if _, err = tx.Exec(`DELETE FROM document_conversions WHERE run_id=? AND format=? AND id!=?`, run, j.Format, id); err != nil {
		return err
	}
	if err = checkDocumentStorage(tx, session, int64(len(content)+len(updated)-len(raw))); err != nil {
		return err
	}
	if _, err = tx.Exec(`UPDATE document_conversions SET status='completed',data=?,content=? WHERE id=? AND status='running'`, string(updated), content, id); err != nil {
		return err
	}
	return tx.Commit()
}

func finishDocumentConversion(j *workflows.DocumentConversion) {
	now := time.Now().UTC()
	j.FinishedAt = now.Format(time.RFC3339Nano)
	if started, err := time.Parse(time.RFC3339Nano, j.StartedAt); err == nil {
		j.Seconds = max(0, now.Sub(started).Seconds())
	}
}

func (d *DB) FailDocumentConversion(session, run, id string, failure error) error {
	tx, err := d.conn.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	var raw string
	if err = tx.QueryRow(`SELECT c.data FROM document_conversions c JOIN workflow_runs r ON r.id=c.run_id WHERE r.session_id=? AND c.run_id=? AND c.id=? AND c.status='running'`, session, run, id).Scan(&raw); err != nil {
		return err
	}
	var j workflows.DocumentConversion
	if err = json.Unmarshal([]byte(raw), &j); err != nil {
		return err
	}
	j.Status = "failed"
	j.Error = string([]rune(failure.Error())[:min(1500, len([]rune(failure.Error())))])
	finishDocumentConversion(&j)
	updated, err := json.Marshal(j)
	if err != nil {
		return err
	}
	if _, err = tx.Exec(`UPDATE document_conversions SET status='failed',data=? WHERE id=? AND status='running'`, string(updated), id); err != nil {
		return err
	}
	if _, err = tx.Exec(`DELETE FROM document_conversions WHERE run_id=? AND format=? AND status='failed' AND id!=?`, run, j.Format, id); err != nil {
		return err
	}
	return tx.Commit()
}

func (d *DB) DocumentConversions(session, run string) ([]workflows.DocumentConversion, error) {
	if _, err := d.DocumentRun(session, run); err != nil {
		return nil, err
	}
	body, _, err := d.DocumentOutput(session, run, "output.txt")
	if err != nil && err != sql.ErrNoRows {
		return nil, err
	}
	inputHash := documentContentHash(body)
	rows, err := d.conn.Query(`SELECT data FROM document_conversions WHERE run_id=? ORDER BY rowid DESC`, run)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	result := []workflows.DocumentConversion{}
	current := map[string]bool{}
	for rows.Next() {
		var raw string
		var j workflows.DocumentConversion
		if err = rows.Scan(&raw); err != nil {
			return nil, err
		}
		if err = json.Unmarshal([]byte(raw), &j); err != nil {
			return nil, err
		}
		j.Current = j.Status == "completed" && j.InputHash == inputHash && !current[j.Format]
		if j.Current {
			current[j.Format] = true
		}
		result = append(result, j)
	}
	return result, rows.Err()
}

// Resolve the current file in one read transaction. Replacement may prune the
// previous row concurrently, so listing IDs and fetching later is not sufficient.
func (d *DB) ConvertedDocumentOutput(session, run, name string) ([]byte, string, error) {
	tx, err := d.conn.BeginTx(context.Background(), &sql.TxOptions{ReadOnly: true})
	if err != nil {
		return nil, "", err
	}
	defer tx.Rollback()
	var body []byte
	err = tx.QueryRow(`SELECT o.data FROM workflow_outputs o JOIN workflow_runs r ON r.id=o.run_id WHERE r.session_id=? AND o.run_id=? AND o.name='output.txt'`, session, run).Scan(&body)
	if err != nil {
		return nil, "", err
	}
	var raw string
	var content []byte
	err = tx.QueryRow(`SELECT data,content FROM document_conversions WHERE run_id=? AND 'output.'||format=? AND status='completed' AND json_extract(data,'$.input_hash')=? ORDER BY rowid DESC LIMIT 1`, run, name, documentContentHash(body)).Scan(&raw, &content)
	if err == sql.ErrNoRows {
		var mime string
		err = tx.QueryRow(`SELECT data,mime FROM workflow_outputs WHERE run_id=? AND name=?`, run, name).Scan(&content, &mime)
		return content, mime, err
	}
	if err != nil {
		return nil, "", err
	}
	var j workflows.DocumentConversion
	if err = json.Unmarshal([]byte(raw), &j); err != nil {
		return nil, "", err
	}
	if j.Hash != documentContentHash(content) || j.Bytes != len(content) {
		return nil, "", fmt.Errorf("converted document integrity check failed")
	}
	return content, j.MIME, nil
}
