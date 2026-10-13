package db

import (
	"database/sql"
	"fmt"
)

// Limits include raw sources, structured snapshots, images, task checkpoints
// and generated files. Repeated tasks over one source also consume this budget.
func checkDocumentStorage(tx *sql.Tx, session string, added int64) error {
	var local, total int64
	query := `SELECT
 (SELECT COALESCE(SUM(length(raw)+length(CAST(data AS BLOB))),0) FROM source_snapshots WHERE session_id=?)+
 (SELECT COALESCE(SUM(length(a.data)),0) FROM source_assets a JOIN source_snapshots s ON s.id=a.source_id WHERE s.session_id=?)+
 (SELECT COALESCE(SUM(length(o.data)),0) FROM workflow_outputs o JOIN workflow_runs r ON r.id=o.run_id WHERE r.session_id=?)+
 (SELECT COALESCE(SUM(length(CAST(u.data AS BLOB))),0) FROM workflow_units u JOIN workflow_runs r ON r.id=u.run_id WHERE r.session_id=?)+
 (SELECT COALESCE(SUM(length(CAST(data AS BLOB))),0) FROM workflow_runs WHERE session_id=? AND json_extract(data,'$.document') IS NOT NULL)`
	if e := tx.QueryRow(query, session, session, session, session, session).Scan(&local); e != nil {
		return e
	}
	var reviewLocal int64
	if e := tx.QueryRow(`SELECT
 (SELECT COALESCE(SUM(length(CAST(a.data AS BLOB))),0) FROM document_attempts a JOIN workflow_runs r ON r.id=a.run_id WHERE r.session_id=?)+
 (SELECT COALESCE(SUM(length(CAST(a.data AS BLOB))),0) FROM document_reviews a JOIN workflow_runs r ON r.id=a.run_id WHERE r.session_id=?)+
 (SELECT COALESCE(SUM(length(CAST(a.data AS BLOB))),0) FROM document_entities a JOIN workflow_runs r ON r.id=a.run_id WHERE r.session_id=?)+
 (SELECT COALESCE(SUM(COALESCE(length(c.content),0)+length(CAST(c.data AS BLOB))),0) FROM document_conversions c JOIN workflow_runs r ON r.id=c.run_id WHERE r.session_id=?)`, session, session, session, session).Scan(&reviewLocal); e != nil {
		return e
	}
	local += reviewLocal
	if local+added > 512<<20 {
		return fmt.Errorf("conversation document storage exceeds 512 MiB")
	}
	if e := tx.QueryRow(`SELECT (SELECT COALESCE(SUM(length(raw)+length(CAST(data AS BLOB))),0) FROM source_snapshots)+(SELECT COALESCE(SUM(length(data)),0) FROM source_assets)+(SELECT COALESCE(SUM(length(data)),0) FROM workflow_outputs)+(SELECT COALESCE(SUM(length(CAST(data AS BLOB))),0) FROM workflow_units)+(SELECT COALESCE(SUM(length(CAST(data AS BLOB))),0) FROM workflow_runs WHERE json_extract(data,'$.document') IS NOT NULL)`).Scan(&total); e != nil {
		return e
	}
	var reviewTotal int64
	if e := tx.QueryRow(`SELECT (SELECT COALESCE(SUM(length(CAST(data AS BLOB))),0) FROM document_attempts)+(SELECT COALESCE(SUM(length(CAST(data AS BLOB))),0) FROM document_reviews)+(SELECT COALESCE(SUM(length(CAST(data AS BLOB))),0) FROM document_entities)+(SELECT COALESCE(SUM(COALESCE(length(content),0)+length(CAST(data AS BLOB))),0) FROM document_conversions)`).Scan(&reviewTotal); e != nil {
		return e
	}
	total += reviewTotal
	if total+added > 2<<30 {
		return fmt.Errorf("document storage exceeds 2 GiB; remove unused document tasks/sources")
	}
	return nil
}
