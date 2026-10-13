package db

import (
	"database/sql"
	"fmt"
	"time"
)

// Repeated model calls must reuse an existing unique name, not create duplicates.
func (d *DB) CreateGroupFromTool(id, name string) (Group, bool, error) {
	tx, err := d.conn.Begin()
	if err != nil {
		return Group{}, false, err
	}
	defer tx.Rollback()
	var count int
	if err = tx.QueryRow(`SELECT COUNT(*) FROM chat_groups WHERE name=?`, name).Scan(&count); err != nil {
		return Group{}, false, err
	}
	if count > 1 {
		return Group{}, false, fmt.Errorf("multiple folders already have that name; clarify the target folder")
	}
	if count == 1 {
		var g Group
		err = tx.QueryRow(`SELECT id,name,position,created_at,updated_at FROM chat_groups WHERE name=?`, name).Scan(&g.ID, &g.Name, &g.Position, &g.CreatedAt, &g.UpdatedAt)
		return g, false, err
	}
	var position int
	if err = tx.QueryRow(`SELECT COALESCE(MAX(position),-1)+1 FROM chat_groups`).Scan(&position); err != nil {
		return Group{}, false, err
	}
	now := time.Now()
	if _, err = tx.Exec(`INSERT INTO chat_groups(id,name,position,created_at,updated_at) VALUES(?,?,?,?,?)`, id, name, position, now, now); err != nil {
		return Group{}, false, err
	}
	return Group{ID: id, Name: name, Position: position, CreatedAt: now, UpdatedAt: now}, true, tx.Commit()
}

func (d *DB) RenameGroupFromTool(id, expectedName, name string) (bool, error) {
	tx, err := d.conn.Begin()
	if err != nil {
		return false, err
	}
	defer tx.Rollback()
	var current string
	if err = tx.QueryRow(`SELECT name FROM chat_groups WHERE id=?`, id).Scan(&current); err != nil {
		return false, err
	}
	if current != expectedName || current == name {
		return false, nil
	}
	var count int
	if err = tx.QueryRow(`SELECT COUNT(*) FROM chat_groups WHERE name=? AND id<>?`, name, id).Scan(&count); err != nil {
		return false, err
	}
	if count > 0 {
		return false, fmt.Errorf("another folder already has that name")
	}
	if _, err = tx.Exec(`UPDATE chat_groups SET name=?,updated_at=? WHERE id=? AND name=?`, name, time.Now(), id, expectedName); err != nil {
		return false, err
	}
	return true, tx.Commit()
}

// Folder deletion matches the UI: only remove the folder and ungroup its chats.
// Checking the observed name in the transaction protects concurrent UI edits.
func (d *DB) DeleteGroupFromTool(id, expectedName string) (bool, int64, error) {
	tx, err := d.conn.Begin()
	if err != nil {
		return false, 0, err
	}
	defer tx.Rollback()
	var current string
	if err = tx.QueryRow(`SELECT name FROM chat_groups WHERE id=?`, id).Scan(&current); err != nil {
		return false, 0, err
	}
	if current != expectedName {
		return false, 0, nil
	}
	result, err := tx.Exec(`UPDATE sessions SET group_id=NULL WHERE group_id=?`, id)
	if err != nil {
		return false, 0, err
	}
	count, err := result.RowsAffected()
	if err != nil {
		return false, 0, err
	}
	result, err = tx.Exec(`DELETE FROM chat_groups WHERE id=? AND name=?`, id, expectedName)
	if err != nil {
		return false, 0, err
	}
	if n, err := result.RowsAffected(); err != nil || n != 1 {
		if err == nil {
			err = sql.ErrNoRows
		}
		return false, 0, err
	}
	return true, count, tx.Commit()
}
