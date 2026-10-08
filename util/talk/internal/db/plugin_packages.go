package db

import (
	"context"
	"encoding/json"
	"fmt"
	"sparktalk/internal/plugins"
)

func (d *DB) PluginPackages(ctx context.Context) ([]plugins.PackageRecord, error) {
	rows, err := d.conn.QueryContext(ctx, `SELECT data FROM plugin_packages ORDER BY id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []plugins.PackageRecord{}
	for rows.Next() {
		var raw string
		if err = rows.Scan(&raw); err != nil {
			return nil, err
		}
		var r plugins.PackageRecord
		if err = json.Unmarshal([]byte(raw), &r); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}
func (d *DB) PluginSnapshot(ctx context.Context, id string) (plugins.Settings, map[string]json.RawMessage, error) {
	s, err := d.PluginSettings(ctx, id)
	if err != nil {
		return s, nil, err
	}
	rows, err := d.conn.QueryContext(ctx, `SELECT key,value FROM plugin_values WHERE plugin_id=?`, id)
	if err != nil {
		return s, nil, err
	}
	defer rows.Close()
	values := map[string]json.RawMessage{}
	for rows.Next() {
		var key, raw string
		if err = rows.Scan(&key, &raw); err != nil {
			return s, nil, err
		}
		values[key] = json.RawMessage(raw)
	}
	return s, values, rows.Err()
}
func (d *DB) CommitPluginPackage(ctx context.Context, id string, r *plugins.PackageRecord, s plugins.Settings, values map[string]json.RawMessage, purge bool) error {
	total := 0
	if len(values) > 128 {
		return fmt.Errorf("too many storage keys")
	}
	for k, v := range values {
		if err := validPluginValue(k, v); err != nil {
			return err
		}
		total += len(v)
	}
	if total > 1<<20 {
		return fmt.Errorf("storage quota exceeded")
	}
	tx, err := d.conn.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if r == nil {
		_, err = tx.ExecContext(ctx, `DELETE FROM plugin_packages WHERE id=?`, id)
	} else {
		var b []byte
		b, err = json.Marshal(r)
		if err == nil {
			_, err = tx.ExecContext(ctx, `INSERT INTO plugin_packages(id,data) VALUES(?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data`, id, string(b))
		}
	}
	if err != nil {
		return err
	}
	if _, err = tx.ExecContext(ctx, `DELETE FROM plugin_values WHERE plugin_id=?`, id); err != nil {
		return err
	}
	if purge {
		if _, err = tx.ExecContext(ctx, `DELETE FROM plugin_settings WHERE id=?`, id); err != nil {
			return err
		}
		if _, err = tx.ExecContext(ctx, `DELETE FROM plugin_runs WHERE plugin_id=?`, id); err != nil {
			return err
		}
	} else {
		s.Enabled = false
		b, err := json.Marshal(s)
		if err != nil {
			return err
		}
		if _, err = tx.ExecContext(ctx, `INSERT INTO plugin_settings(id,data) VALUES(?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data`, id, string(b)); err != nil {
			return err
		}
		for k, v := range values {
			if _, err = tx.ExecContext(ctx, `INSERT INTO plugin_values(plugin_id,key,value) VALUES(?,?,?)`, id, k, string(v)); err != nil {
				return err
			}
		}
	}
	return tx.Commit()
}
