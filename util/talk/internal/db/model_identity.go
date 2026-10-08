package db

import "sparktalk/internal/modelidentity"

func (d *DB) migrateModelIdentity() error {
	tx, err := d.conn.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	for _, table := range []string{"sessions", "context_segments"} {
		if _, err = tx.Exec("UPDATE "+table+" SET model=? WHERE model IN (?,?)", modelidentity.Qwen38FNEXL3, modelidentity.LegacyQwenModel, modelidentity.LegacyQwenRuntime); err != nil {
			return err
		}
	}
	return tx.Commit()
}
