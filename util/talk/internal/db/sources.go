package db

import (
	"encoding/json"
	"fmt"
	"sparktalk/internal/sourcebook"
	"time"
)

func (d *DB) initSources() error {
	_, e := d.conn.Exec(`CREATE TABLE IF NOT EXISTS source_snapshots(id TEXT PRIMARY KEY,session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,data TEXT NOT NULL,raw BLOB,created INTEGER NOT NULL);
 CREATE UNIQUE INDEX IF NOT EXISTS workflow_document_request ON workflow_runs(session_id,json_extract(data,'$.document.request_id')) WHERE json_extract(data,'$.document.request_id') IS NOT NULL AND json_extract(data,'$.document.request_id')!='';
 CREATE INDEX IF NOT EXISTS source_session ON source_snapshots(session_id);
 CREATE TABLE IF NOT EXISTS source_assets(source_id TEXT NOT NULL REFERENCES source_snapshots(id) ON DELETE CASCADE,id TEXT NOT NULL,data BLOB NOT NULL,PRIMARY KEY(source_id,id));
 CREATE TABLE IF NOT EXISTS workflow_units(run_id TEXT NOT NULL REFERENCES workflow_runs(id) ON DELETE CASCADE,ordinal INTEGER NOT NULL,data TEXT NOT NULL,PRIMARY KEY(run_id,ordinal));
 CREATE TABLE IF NOT EXISTS workflow_outputs(run_id TEXT NOT NULL REFERENCES workflow_runs(id) ON DELETE CASCADE,name TEXT NOT NULL,mime TEXT NOT NULL,data BLOB NOT NULL,PRIMARY KEY(run_id,name));`)
	if e != nil {
		return e
	}
	if e = d.initDocumentReviewStorage(); e != nil {
		return e
	}
	return d.initDocumentConversionStorage()
}
func (d *DB) SaveSource(id, session string, b *sourcebook.Book, raw []byte) error {
	data, e := json.Marshal(b)
	if e != nil {
		return e
	}
	tx, e := d.conn.Begin()
	if e != nil {
		return e
	}
	defer tx.Rollback()
	var used int64
	if e = tx.QueryRow(`SELECT COALESCE(SUM(length(raw)+length(CAST(data AS BLOB))),0)+(SELECT COALESCE(SUM(length(a.data)),0) FROM source_assets a JOIN source_snapshots s ON s.id=a.source_id WHERE s.session_id=?) FROM source_snapshots WHERE session_id=?`, session, session).Scan(&used); e != nil {
		return e
	}
	size := len(raw) + len(data)
	for _, a := range b.Assets {
		size += len(a.Data)
	}
	if e = checkDocumentStorage(tx, session, int64(size)); e != nil {
		return e
	}
	if used+int64(size) > 256<<20 {
		return fmt.Errorf("conversation source storage exceeds 256 MiB")
	}
	if _, e = tx.Exec(`INSERT INTO source_snapshots(id,session_id,data,raw,created) VALUES(?,?,?,?,?)`, id, session, string(data), raw, time.Now().Unix()); e != nil {
		return e
	}
	for _, a := range b.Assets {
		if len(a.Data) > 0 {
			if _, e = tx.Exec(`INSERT INTO source_assets(source_id,id,data) VALUES(?,?,?)`, id, a.ID, a.Data); e != nil {
				return e
			}
		}
	}
	return tx.Commit()
}
func (d *DB) Source(session, id string, assets bool) (*sourcebook.Book, error) {
	var raw string
	if e := d.conn.QueryRow(`SELECT data FROM source_snapshots WHERE id=? AND session_id=?`, id, session).Scan(&raw); e != nil {
		return nil, e
	}
	var b sourcebook.Book
	if e := json.Unmarshal([]byte(raw), &b); e != nil {
		return nil, e
	}
	if b.Version < sourcebook.SnapshotVersion {
		var original []byte
		if e := d.conn.QueryRow(`SELECT raw FROM source_snapshots WHERE id=? AND session_id=?`, id, session).Scan(&original); e != nil {
			return nil, e
		}
		b.RestoreInlineStructure(original)
	}
	if assets {
		rows, e := d.conn.Query(`SELECT id,data FROM source_assets WHERE source_id=?`, id)
		if e != nil {
			return nil, e
		}
		defer rows.Close()
		for rows.Next() {
			var id string
			var data []byte
			if e = rows.Scan(&id, &data); e != nil {
				return nil, e
			}
			for i := range b.Assets {
				if b.Assets[i].ID == id {
					b.Assets[i].Data = data
					break
				}
			}
		}
		if e = rows.Err(); e != nil {
			return nil, e
		}
		b.VerifyAssets()
	}
	return &b, nil
}

func (d *DB) RawSource(session, id string) ([]byte, error) {
	var raw []byte
	err := d.conn.QueryRow(`SELECT raw FROM source_snapshots WHERE id=? AND session_id=?`, id, session).Scan(&raw)
	return raw, err
}
func (d *DB) DeleteSource(session, id string) error {
	result, e := d.conn.Exec(`DELETE FROM source_snapshots WHERE id=? AND session_id=? AND NOT EXISTS(SELECT 1 FROM workflow_runs w,json_each(json_extract(w.data,'$.document.sources')) refs WHERE w.session_id=source_snapshots.session_id AND refs.value=source_snapshots.id)`, id, session)
	if e != nil {
		return e
	}
	n, _ := result.RowsAffected()
	if n == 0 && d.SourceExists(session, id) {
		return fmt.Errorf("source is retained by a document task")
	}
	return nil
}

func (d *DB) CleanupSources() error {
	_, e := d.conn.Exec(`DELETE FROM source_snapshots WHERE created<? AND NOT EXISTS(SELECT 1 FROM workflow_runs w,json_each(json_extract(w.data,'$.document.sources')) refs WHERE w.session_id=source_snapshots.session_id AND refs.value=source_snapshots.id)`, time.Now().Add(-24*time.Hour).Unix())
	return e
}
func (d *DB) SourceExists(session, id string) bool {
	var n int
	e := d.conn.QueryRow(`SELECT 1 FROM source_snapshots WHERE id=? AND session_id=?`, id, session).Scan(&n)
	return e == nil
}

func (d *DB) SaveSourceAssets(session, id string, b *sourcebook.Book) error {
	data, e := json.Marshal(b)
	if e != nil {
		return e
	}
	tx, e := d.conn.Begin()
	if e != nil {
		return e
	}
	defer tx.Rollback()
	var old int64
	if e = tx.QueryRow(`SELECT length(CAST(data AS BLOB))+(SELECT COALESCE(SUM(length(data)),0) FROM source_assets WHERE source_id=?) FROM source_snapshots WHERE id=? AND session_id=?`, id, id, session).Scan(&old); e != nil {
		return e
	}
	size := len(data)
	for _, a := range b.Assets {
		size += len(a.Data)
	}
	if e = checkDocumentStorage(tx, session, int64(size)-old); e != nil {
		return e
	}
	result, e := tx.Exec(`UPDATE source_snapshots SET data=? WHERE id=? AND session_id=?`, string(data), id, session)
	if e != nil {
		return e
	}
	n, _ := result.RowsAffected()
	if n != 1 {
		return fmt.Errorf("source no longer exists")
	}
	for _, a := range b.Assets {
		if len(a.Data) > 0 {
			if _, e = tx.Exec(`INSERT INTO source_assets(source_id,id,data) VALUES(?,?,?) ON CONFLICT(source_id,id) DO UPDATE SET data=excluded.data`, id, a.ID, a.Data); e != nil {
				return e
			}
		}
	}
	return tx.Commit()
}
