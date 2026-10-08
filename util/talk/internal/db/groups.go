package db

import (
	"database/sql"
	"fmt"
	"time"
)

func (d *DB) Groups() ([]Group, error) {
	rows, err := d.conn.Query(`SELECT id,name,position,created_at,updated_at FROM chat_groups ORDER BY position,id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	groups := []Group{}
	for rows.Next() {
		var group Group
		if err := rows.Scan(&group.ID, &group.Name, &group.Position, &group.CreatedAt, &group.UpdatedAt); err != nil {
			return nil, err
		}
		groups = append(groups, group)
	}
	return groups, rows.Err()
}

func (d *DB) CreateGroup(id, name string) (Group, error) {
	now := time.Now()
	var position int
	if err := d.conn.QueryRow(`SELECT COALESCE(MAX(position),-1)+1 FROM chat_groups`).Scan(&position); err != nil {
		return Group{}, err
	}
	_, err := d.conn.Exec(`INSERT INTO chat_groups(id,name,position,created_at,updated_at) VALUES(?,?,?,?,?)`, id, name, position, now, now)
	return Group{ID: id, Name: name, Position: position, CreatedAt: now, UpdatedAt: now}, err
}

func (d *DB) RenameGroup(id, name string) error {
	result, err := d.conn.Exec(`UPDATE chat_groups SET name=?,updated_at=? WHERE id=?`, name, time.Now(), id)
	return rowsAffected(result, err)
}

func (d *DB) MoveGroup(id, direction string) error {
	tx, err := d.conn.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	var position int
	if err := tx.QueryRow(`SELECT position FROM chat_groups WHERE id=?`, id).Scan(&position); err != nil {
		return err
	}
	operator, order := `<`, `DESC`
	if direction == "down" {
		operator, order = `>`, `ASC`
	}
	var otherID string
	var otherPosition int
	query := fmt.Sprintf(`SELECT id,position FROM chat_groups WHERE position %s ? ORDER BY position %s LIMIT 1`, operator, order)
	if err := tx.QueryRow(query, position).Scan(&otherID, &otherPosition); err != nil {
		if err == sql.ErrNoRows {
			return nil
		}
		return err
	}
	if _, err := tx.Exec(`UPDATE chat_groups SET position=?,updated_at=? WHERE id=?`, otherPosition, time.Now(), id); err != nil {
		return err
	}
	if _, err := tx.Exec(`UPDATE chat_groups SET position=?,updated_at=? WHERE id=?`, position, time.Now(), otherID); err != nil {
		return err
	}
	return tx.Commit()
}

func (d *DB) DeleteGroup(id string) error {
	tx, err := d.conn.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if _, err := tx.Exec(`UPDATE sessions SET group_id=NULL WHERE group_id=?`, id); err != nil {
		return err
	}
	result, err := tx.Exec(`DELETE FROM chat_groups WHERE id=?`, id)
	if err := rowsAffected(result, err); err != nil {
		return err
	}
	return tx.Commit()
}

func (d *DB) SetSessionGroup(sessionID, groupID string) error {
	if groupID != "" {
		var exists int
		if err := d.conn.QueryRow(`SELECT 1 FROM chat_groups WHERE id=?`, groupID).Scan(&exists); err != nil {
			return err
		}
	}
	var value any
	if groupID != "" {
		value = groupID
	}
	result, err := d.conn.Exec(`UPDATE sessions SET group_id=?,updated_at=? WHERE id=?`, value, time.Now(), sessionID)
	return rowsAffected(result, err)
}

func rowsAffected(result sql.Result, err error) error {
	if err != nil {
		return err
	}
	changed, err := result.RowsAffected()
	if err == nil && changed == 0 {
		return sql.ErrNoRows
	}
	return err
}
