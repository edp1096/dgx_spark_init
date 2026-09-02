package db

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"time"
)

const (
	MessagePending   = "pending"
	MessageCompleted = "completed"
	MessageFailed    = "failed"
	MessageCancelled = "cancelled"
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

func (d *DB) Session(id string) (Session, error) {
	var item Session
	err := d.conn.QueryRow(`SELECT id,title,model,reasoning_effort,COALESCE(group_id,''),created_at,updated_at FROM sessions WHERE id=?`, id).
		Scan(&item.ID, &item.Title, &item.Model, &item.Reasoning, &item.GroupID, &item.CreatedAt, &item.UpdatedAt)
	return item, err
}

func (d *DB) DeleteSession(id string) error {
	tx, err := d.conn.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if _, err := tx.Exec(`DELETE FROM ssh_conversation_grants WHERE session_id=?`, id); err != nil {
		return err
	}
	if _, err := tx.Exec(`DELETE FROM tool_grants WHERE scope='conversation' AND session_id=?`, id); err != nil {
		return err
	}
	// Do not rely on SQLite's connection-local foreign_keys pragma here.
	// Explicit deletion also guarantees that FTS cleanup triggers run.
	if _, err := tx.Exec(`DELETE FROM context_segments WHERE session_id=?`, id); err != nil {
		return err
	}
	if _, err := tx.Exec(`DELETE FROM messages WHERE session_id=?`, id); err != nil {
		return err
	}
	if _, err := tx.Exec(`DELETE FROM tool_audit WHERE session_id=?`, id); err != nil {
		return err
	}
	if _, err := tx.Exec(`DELETE FROM sessions WHERE id=?`, id); err != nil {
		return err
	}
	return tx.Commit()
}

func (d *DB) Messages(sessionID string) ([]Message, error) {
	rows, err := d.conn.Query(`SELECT id,session_id,role,status,error,content,reasoning_content,tool_trace,response_variants,created_at FROM messages WHERE session_id=? ORDER BY id`, sessionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []Message{}
	for rows.Next() {
		var item Message
		var traceJSON, variantsJSON string
		if err := rows.Scan(&item.ID, &item.SessionID, &item.Role, &item.Status, &item.Error, &item.Content, &item.Reasoning, &traceJSON, &variantsJSON, &item.CreatedAt); err != nil {
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
	return d.addMessage(sessionID, role, MessageCompleted, "", content, reasoning, toolTrace, attachments)
}

func (d *DB) AddPendingMessage(sessionID, content string, attachments []Attachment) (Message, error) {
	return d.addMessage(sessionID, "user", MessagePending, "", content, "", nil, attachments)
}

// AppendMessageVariantAttachment persists media imported during a model tool
// call on the exact user-message variant that requested it.
func (d *DB) AppendMessageVariantAttachment(messageID int64, variantIndex int, attachment Attachment) error {
	return d.ReplaceMessageVariantAttachment(messageID, variantIndex, "", attachment)
}

// ReplaceMessageVariantAttachment appends an attachment, or atomically swaps
// a previous download when retrying the same media_import call.
func (d *DB) ReplaceMessageVariantAttachment(messageID int64, variantIndex int, replaceID string, attachment Attachment) error {
	tx, err := d.conn.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	var role, variantsJSON string
	if err := tx.QueryRow(`SELECT role,response_variants FROM messages WHERE id=?`, messageID).Scan(&role, &variantsJSON); err != nil {
		return err
	}
	if role != "user" {
		return fmt.Errorf("message %d is not a user message", messageID)
	}
	var variants []ResponseVariant
	if err := json.Unmarshal([]byte(variantsJSON), &variants); err != nil || len(variants) == 0 {
		return fmt.Errorf("message %d has no response variants", messageID)
	}
	if variantIndex < 0 {
		variantIndex = len(variants) - 1
	}
	if variantIndex < 0 || variantIndex >= len(variants) {
		return fmt.Errorf("invalid user variant %d", variantIndex)
	}
	attachments := variants[variantIndex].Attachments[:0]
	for _, existing := range variants[variantIndex].Attachments {
		if existing.ID == attachment.ID {
			return nil
		}
		if replaceID != "" && existing.ID == replaceID {
			continue
		}
		attachments = append(attachments, existing)
	}
	variants[variantIndex].Attachments = append(attachments, attachment)
	updated, err := json.Marshal(variants)
	if err != nil {
		return err
	}
	if _, err := tx.Exec(`UPDATE messages SET response_variants=? WHERE id=?`, string(updated), messageID); err != nil {
		return err
	}
	return tx.Commit()
}

func (d *DB) addMessage(sessionID, role, status, failure, content, reasoning string, toolTrace []ToolEvent, attachments []Attachment) (Message, error) {
	now := time.Now()
	traceJSON, _ := json.Marshal(toolTrace)
	variants := []ResponseVariant{{Content: content, Reasoning: reasoning, ToolTrace: toolTrace, Attachments: attachments, CreatedAt: now}}
	variantsJSON, _ := json.Marshal(variants)
	result, err := d.conn.Exec(`INSERT INTO messages(session_id,role,status,error,content,reasoning_content,tool_trace,response_variants,created_at) VALUES(?,?,?,?,?,?,?,?,?)`, sessionID, role, status, failure, content, reasoning, string(traceJSON), string(variantsJSON), now)
	if err != nil {
		return Message{}, err
	}
	id, _ := result.LastInsertId()
	_, _ = d.conn.Exec(`UPDATE sessions SET updated_at=? WHERE id=?`, now, sessionID)
	return Message{ID: id, SessionID: sessionID, Role: role, Status: status, Error: failure, Content: content, Reasoning: reasoning, ToolTrace: toolTrace, Attachments: attachments, Variants: variants, CreatedAt: now}, nil
}

// CompletePendingTurn atomically commits the pending user request and its
// assistant answer. A model-facing turn is never half-completed.
func (d *DB) CompletePendingTurn(userMessageID int64, content, reasoning string, toolTrace []ToolEvent) (Message, error) {
	return d.CompletePendingTurnWithAttachments(userMessageID, content, reasoning, toolTrace, nil)
}

func (d *DB) CompletePendingTurnWithAttachments(userMessageID int64, content, reasoning string, toolTrace []ToolEvent, attachments []Attachment) (Message, error) {
	tx, err := d.conn.Begin()
	if err != nil {
		return Message{}, err
	}
	defer tx.Rollback()
	var sessionID string
	if err := tx.QueryRow(`SELECT session_id FROM messages WHERE id=? AND role='user' AND status='pending'`, userMessageID).Scan(&sessionID); err != nil {
		return Message{}, err
	}
	now := time.Now()
	if _, err := tx.Exec(`UPDATE messages SET status='completed',error='' WHERE id=?`, userMessageID); err != nil {
		return Message{}, err
	}
	traceJSON, _ := json.Marshal(toolTrace)
	variants := []ResponseVariant{{Content: content, Reasoning: reasoning, ToolTrace: toolTrace, Attachments: attachments, CreatedAt: now}}
	variantsJSON, _ := json.Marshal(variants)
	result, err := tx.Exec(`INSERT INTO messages(session_id,role,status,error,content,reasoning_content,tool_trace,response_variants,created_at) VALUES(?,'assistant','completed','',?,?,?,?,?)`, sessionID, content, reasoning, string(traceJSON), string(variantsJSON), now)
	if err != nil {
		return Message{}, err
	}
	id, _ := result.LastInsertId()
	if _, err := tx.Exec(`UPDATE sessions SET updated_at=? WHERE id=?`, now, sessionID); err != nil {
		return Message{}, err
	}
	if err := tx.Commit(); err != nil {
		return Message{}, err
	}
	return Message{ID: id, SessionID: sessionID, Role: "assistant", Status: MessageCompleted, Content: content, Reasoning: reasoning, ToolTrace: toolTrace, Attachments: attachments, Variants: variants, CreatedAt: now}, nil
}

func (d *DB) FailPendingTurn(userMessageID int64, status, failure, partialContent, partialReasoning string, toolTrace []ToolEvent) error {
	if status != MessageCancelled {
		status = MessageFailed
	}
	tx, err := d.conn.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	var sessionID string
	if err := tx.QueryRow(`SELECT session_id FROM messages WHERE id=? AND role='user' AND status='pending'`, userMessageID).Scan(&sessionID); err != nil {
		return err
	}
	if _, err := tx.Exec(`UPDATE messages SET status=?,error=? WHERE id=?`, status, failure, userMessageID); err != nil {
		return err
	}
	if partialContent != "" || partialReasoning != "" || len(toolTrace) > 0 {
		now := time.Now()
		traceJSON, _ := json.Marshal(toolTrace)
		variants := []ResponseVariant{{Content: partialContent, Reasoning: partialReasoning, ToolTrace: toolTrace, CreatedAt: now}}
		variantsJSON, _ := json.Marshal(variants)
		if _, err := tx.Exec(`INSERT INTO messages(session_id,role,status,error,content,reasoning_content,tool_trace,response_variants,created_at) VALUES(?,'assistant',?,?, ?,?,?,?,?)`, sessionID, status, failure, partialContent, partialReasoning, string(traceJSON), string(variantsJSON), now); err != nil {
			return err
		}
	}
	return tx.Commit()
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

func (d *DB) CompletedUserMessageCount(sessionID string) (int, error) {
	var count int
	err := d.conn.QueryRow(`SELECT COUNT(*) FROM messages WHERE session_id=? AND role='user' AND status='completed'`, sessionID).Scan(&count)
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
