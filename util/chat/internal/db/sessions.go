package db

import (
	"database/sql"
	"encoding/json"
	"time"
)

func (d *DB) CreateSession(id, title, model, reasoning string) (Session, error) {
	now := time.Now()
	_, err := d.conn.Exec(`INSERT INTO sessions(id,title,model,reasoning_effort,created_at,updated_at) VALUES(?,?,?,?,?,?)`, id, title, model, reasoning, now, now)
	return Session{ID: id, Title: title, Model: model, Reasoning: reasoning, CreatedAt: now, UpdatedAt: now}, err
}

func (d *DB) Sessions() ([]Session, error) {
	rows, err := d.conn.Query(`SELECT id,title,model,reasoning_effort,COALESCE(group_id,''),created_at,updated_at FROM sessions ORDER BY updated_at DESC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Session{}
	for rows.Next() {
		var item Session
		if err := rows.Scan(&item.ID, &item.Title, &item.Model, &item.Reasoning, &item.GroupID, &item.CreatedAt, &item.UpdatedAt); err != nil {
			return nil, err
		}
		out = append(out, item)
	}
	return out, rows.Err()
}

func (d *DB) DeleteSession(id string) error {
	_, err := d.conn.Exec(`DELETE FROM sessions WHERE id=?`, id)
	return err
}

func (d *DB) Messages(sessionID string) ([]Message, error) {
	rows, err := d.conn.Query(`SELECT id,session_id,role,content,reasoning_content,tool_trace,response_variants,created_at FROM messages WHERE session_id=? ORDER BY id`, sessionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Message{}
	for rows.Next() {
		var item Message
		var traceJSON, variantsJSON string
		if err := rows.Scan(&item.ID, &item.SessionID, &item.Role, &item.Content, &item.Reasoning, &traceJSON, &variantsJSON, &item.CreatedAt); err != nil {
			return nil, err
		}
		_ = json.Unmarshal([]byte(traceJSON), &item.ToolTrace)
		_ = json.Unmarshal([]byte(variantsJSON), &item.Variants)
		ensureCurrentVariant(&item)
		syncCurrentAttachments(&item)
		out = append(out, item)
	}
	return out, rows.Err()
}

func (d *DB) AddMessage(sessionID, role, content, reasoning string, toolTrace []ToolEvent, attachments []Attachment) (Message, error) {
	now := time.Now()
	traceJSON, _ := json.Marshal(toolTrace)
	variants := []ResponseVariant{{Content: content, Reasoning: reasoning, ToolTrace: toolTrace, Attachments: attachments, CreatedAt: now}}
	variantsJSON, _ := json.Marshal(variants)
	result, err := d.conn.Exec(`INSERT INTO messages(session_id,role,content,reasoning_content,tool_trace,response_variants,created_at) VALUES(?,?,?,?,?,?,?)`, sessionID, role, content, reasoning, string(traceJSON), string(variantsJSON), now)
	if err != nil {
		return Message{}, err
	}
	id, _ := result.LastInsertId()
	_, _ = d.conn.Exec(`UPDATE sessions SET updated_at=? WHERE id=?`, now, sessionID)
	return Message{ID: id, SessionID: sessionID, Role: role, Content: content, Reasoning: reasoning, ToolTrace: toolTrace, Attachments: attachments, Variants: variants, CreatedAt: now}, nil
}

func (d *DB) UpdateSession(id, title, model, reasoning string) error {
	_, err := d.conn.Exec(`UPDATE sessions SET title=CASE WHEN ?='' THEN title ELSE ? END, model=?, reasoning_effort=?, updated_at=? WHERE id=?`, title, title, model, reasoning, time.Now(), id)
	return err
}

func (d *DB) UpdateSessionTitle(id, title string) error {
	_, err := d.conn.Exec(`UPDATE sessions SET title=?, updated_at=? WHERE id=? AND title_manual=0`, title, time.Now(), id)
	return err
}

func (d *DB) RenameSession(id, title string) error {
	result, err := d.conn.Exec(`UPDATE sessions SET title=?, title_manual=1, updated_at=? WHERE id=?`, title, time.Now(), id)
	if err != nil {
		return err
	}
	changed, err := result.RowsAffected()
	if err == nil && changed == 0 {
		return sql.ErrNoRows
	}
	return err
}

func (d *DB) MessageCount(sessionID string) (int, error) {
	var count int
	err := d.conn.QueryRow(`SELECT COUNT(*) FROM messages WHERE session_id=?`, sessionID).Scan(&count)
	return count, err
}

// ReferencedAttachmentIDs returns every image referenced by any message
// variant, including variants that are not currently selected in the UI.
func (d *DB) ReferencedAttachmentIDs() (map[string]struct{}, error) {
	rows, err := d.conn.Query(`SELECT response_variants FROM messages`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	ids := make(map[string]struct{})
	for rows.Next() {
		var variantsJSON string
		if err := rows.Scan(&variantsJSON); err != nil {
			return nil, err
		}
		var variants []ResponseVariant
		if err := json.Unmarshal([]byte(variantsJSON), &variants); err != nil {
			continue
		}
		for _, variant := range variants {
			for _, attachment := range variant.Attachments {
				if attachment.ID != "" {
					ids[attachment.ID] = struct{}{}
				}
			}
		}
	}
	return ids, rows.Err()
}

// RetryContext returns the assistant message being retried and the conversation
// leading up to it. The existing response is not changed until replacement
// generation completes successfully.
