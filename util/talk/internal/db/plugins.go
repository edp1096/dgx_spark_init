package db

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"sparktalk/internal/plugins"
)

func migratePlugins(c *sql.DB) error {
	_, err := c.Exec(`
 CREATE TABLE IF NOT EXISTS plugin_packages(id TEXT PRIMARY KEY,data TEXT NOT NULL);
 CREATE TABLE IF NOT EXISTS plugin_settings(id TEXT PRIMARY KEY,data TEXT NOT NULL);
 CREATE TABLE IF NOT EXISTS plugin_values(plugin_id TEXT NOT NULL,key TEXT NOT NULL,value TEXT NOT NULL,PRIMARY KEY(plugin_id,key));
 CREATE TABLE IF NOT EXISTS plugin_runs(id TEXT PRIMARY KEY,plugin_id TEXT NOT NULL,request_key TEXT NOT NULL,data TEXT NOT NULL,status TEXT NOT NULL,created_at TEXT NOT NULL,UNIQUE(plugin_id,request_key));
 CREATE INDEX IF NOT EXISTS plugin_runs_owner ON plugin_runs(plugin_id,created_at DESC);
 `)
	return err
}
func (d *DB) PluginSettings(ctx context.Context, id string) (plugins.Settings, error) {
	var raw string
	err := d.conn.QueryRowContext(ctx, `SELECT data FROM plugin_settings WHERE id=?`, id).Scan(&raw)
	if errors.Is(err, sql.ErrNoRows) {
		return plugins.Settings{}, nil
	}
	if err != nil {
		return plugins.Settings{}, err
	}
	var s plugins.Settings
	err = json.Unmarshal([]byte(raw), &s)
	return s, err
}
func (d *DB) SavePluginSettings(ctx context.Context, id string, s plugins.Settings) error {
	b, err := json.Marshal(s)
	if err != nil {
		return err
	}
	_, err = d.conn.ExecContext(ctx, `INSERT INTO plugin_settings(id,data) VALUES(?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data`, id, string(b))
	return err
}
func validPluginValue(key string, v json.RawMessage) error {
	if len(key) == 0 || len(key) > 128 || len(v) > plugins.MaxPayload || !json.Valid(v) {
		return fmt.Errorf("invalid plugin storage key/value")
	}
	return nil
}
func (d *DB) PluginValue(ctx context.Context, id, key string) (json.RawMessage, error) {
	var raw string
	err := d.conn.QueryRowContext(ctx, `SELECT value FROM plugin_values WHERE plugin_id=? AND key=?`, id, key).Scan(&raw)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, plugins.ErrNotFound
	}
	return json.RawMessage(raw), err
}
func (d *DB) PutPluginValue(ctx context.Context, id, key string, v json.RawMessage) error {
	if err := validPluginValue(key, v); err != nil {
		return err
	}
	tx, err := d.conn.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	var size, count int
	if err = tx.QueryRowContext(ctx, `SELECT COALESCE(SUM(length(CAST(value AS BLOB))),0),COUNT(*) FROM plugin_values WHERE plugin_id=? AND key<>?`, id, key).Scan(&size, &count); err != nil {
		return err
	}
	if size+len(v) > 1<<20 || count >= 128 {
		return fmt.Errorf("plugin storage quota exceeded")
	}
	if _, err = tx.ExecContext(ctx, `INSERT INTO plugin_values(plugin_id,key,value) VALUES(?,?,?) ON CONFLICT(plugin_id,key) DO UPDATE SET value=excluded.value`, id, key, string(v)); err != nil {
		return err
	}
	return tx.Commit()
}
func (d *DB) DeletePluginValue(ctx context.Context, id, key string) error {
	_, err := d.conn.ExecContext(ctx, `DELETE FROM plugin_values WHERE plugin_id=? AND key=?`, id, key)
	return err
}
func (d *DB) MigratePlugin(ctx context.Context, id string, from, to int, fn func(map[string]json.RawMessage) error) error {
	tx, err := d.conn.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	var raw string
	if err = tx.QueryRowContext(ctx, `SELECT data FROM plugin_settings WHERE id=?`, id).Scan(&raw); err != nil {
		return err
	}
	var s plugins.Settings
	if err = json.Unmarshal([]byte(raw), &s); err != nil {
		return err
	}
	if s.Enabled || s.DataVersion != from {
		return plugins.ErrConflict
	}
	rows, err := tx.QueryContext(ctx, `SELECT key,value FROM plugin_values WHERE plugin_id=?`, id)
	if err != nil {
		return err
	}
	values := map[string]json.RawMessage{}
	for rows.Next() {
		var k, v string
		if err = rows.Scan(&k, &v); err != nil {
			rows.Close()
			return err
		}
		values[k] = json.RawMessage(v)
	}
	err = rows.Err()
	rows.Close()
	if err != nil {
		return err
	}
	if err = fn(values); err != nil {
		return err
	}
	if len(values) > 128 {
		return fmt.Errorf("plugin storage quota exceeded")
	}
	total := 0
	for k, v := range values {
		if err = validPluginValue(k, v); err != nil {
			return err
		}
		total += len(v)
	}
	if total > 1<<20 {
		return fmt.Errorf("plugin storage quota exceeded")
	}
	if _, err = tx.ExecContext(ctx, `DELETE FROM plugin_values WHERE plugin_id=?`, id); err != nil {
		return err
	}
	for k, v := range values {
		if _, err = tx.ExecContext(ctx, `INSERT INTO plugin_values(plugin_id,key,value) VALUES(?,?,?)`, id, k, string(v)); err != nil {
			return err
		}
	}
	s.DataVersion = to
	b, err := json.Marshal(s)
	if err != nil {
		return err
	}
	if _, err = tx.ExecContext(ctx, `UPDATE plugin_settings SET data=? WHERE id=?`, string(b), id); err != nil {
		return err
	}
	return tx.Commit()
}
func (d *DB) CreatePluginRun(ctx context.Context, r plugins.Run) (plugins.Run, bool, error) {
	b, err := json.Marshal(r)
	if err != nil {
		return plugins.Run{}, false, err
	}
	result, err := d.conn.ExecContext(ctx, `INSERT INTO plugin_runs(id,plugin_id,request_key,data,status,created_at) VALUES(?,?,?,?,?,?) ON CONFLICT(plugin_id,request_key) DO NOTHING`, r.ID, r.PluginID, r.Key, string(b), r.Status, r.CreatedAt.Format(time.RFC3339Nano))
	if err != nil {
		return plugins.Run{}, false, err
	}
	n, err := result.RowsAffected()
	if err != nil {
		return plugins.Run{}, false, err
	}
	if n == 1 {
		return r, true, nil
	}
	var raw string
	if err = d.conn.QueryRowContext(ctx, `SELECT data FROM plugin_runs WHERE plugin_id=? AND request_key=?`, r.PluginID, r.Key).Scan(&raw); err != nil {
		return plugins.Run{}, false, err
	}
	var old plugins.Run
	if err = json.Unmarshal([]byte(raw), &old); err != nil {
		return old, false, err
	}
	if old.Operation != r.Operation || old.Request.SessionID != r.Request.SessionID || string(old.Request.Input) != string(r.Request.Input) {
		return plugins.Run{}, false, plugins.ErrConflict
	}
	return old, false, nil
}
func (d *DB) FinishPluginRun(ctx context.Context, r plugins.Run) error {
	b, err := json.Marshal(r)
	if err != nil {
		return err
	}
	_, err = d.conn.ExecContext(ctx, `UPDATE plugin_runs SET data=?,status=? WHERE id=? AND plugin_id=? AND status='running'`, string(b), r.Status, r.ID, r.PluginID)
	return err
}
func (d *DB) PluginRuns(ctx context.Context, id string) ([]plugins.Run, error) {
	rows, err := d.conn.QueryContext(ctx, `SELECT data FROM plugin_runs WHERE plugin_id=? ORDER BY created_at DESC LIMIT 100`, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []plugins.Run{}
	for rows.Next() {
		var raw string
		var r plugins.Run
		if err = rows.Scan(&raw); err != nil {
			return nil, err
		}
		if err = json.Unmarshal([]byte(raw), &r); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}
func (d *DB) RecoverPluginRuns(ctx context.Context) error {
	tx, err := d.conn.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	rows, err := tx.QueryContext(ctx, `SELECT data FROM plugin_runs WHERE status='running'`)
	if err != nil {
		return err
	}
	runs := []plugins.Run{}
	for rows.Next() {
		var raw string
		var r plugins.Run
		if err = rows.Scan(&raw); err != nil {
			rows.Close()
			return err
		}
		if err = json.Unmarshal([]byte(raw), &r); err != nil {
			rows.Close()
			return err
		}
		runs = append(runs, r)
	}
	err = rows.Err()
	rows.Close()
	if err != nil {
		return err
	}
	for _, r := range runs {
		r.Status = "interrupted"
		r.Error = "Talk stopped before the run completed; not replayed automatically"
		r.UpdatedAt = time.Now().UTC()
		b, _ := json.Marshal(r)
		if _, err = tx.ExecContext(ctx, `UPDATE plugin_runs SET status=?,data=? WHERE id=?`, r.Status, string(b), r.ID); err != nil {
			return err
		}
	}
	return tx.Commit()
}

func (d *DB) PluginRunByKey(ctx context.Context, id, key string) (plugins.Run, error) {
	var raw string
	err := d.conn.QueryRowContext(ctx, `SELECT data FROM plugin_runs WHERE plugin_id=? AND request_key=?`, id, key).Scan(&raw)
	if errors.Is(err, sql.ErrNoRows) {
		return plugins.Run{}, plugins.ErrNotFound
	}
	if err != nil {
		return plugins.Run{}, err
	}
	var r plugins.Run
	err = json.Unmarshal([]byte(raw), &r)
	return r, err
}
