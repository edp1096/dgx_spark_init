package db

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"time"
)

func (d *DB) RetryContext(messageID int64, userVariant int) (Message, []Message, error) {
	var target Message
	var targetTrace, targetVariants string
	err := d.conn.QueryRow(`SELECT id,session_id,role,content,reasoning_content,tool_trace,response_variants,created_at FROM messages WHERE id=?`, messageID).
		Scan(&target.ID, &target.SessionID, &target.Role, &target.Content, &target.Reasoning, &targetTrace, &targetVariants, &target.CreatedAt)
	if err != nil {
		return Message{}, nil, err
	}
	if target.Role != "assistant" {
		return Message{}, nil, fmt.Errorf("message %d is not an assistant response", messageID)
	}
	_ = json.Unmarshal([]byte(targetTrace), &target.ToolTrace)
	_ = json.Unmarshal([]byte(targetVariants), &target.Variants)
	ensureCurrentVariant(&target)
	syncCurrentAttachments(&target)
	rows, err := d.conn.Query(`SELECT id,session_id,role,content,reasoning_content,tool_trace,response_variants,created_at FROM messages WHERE session_id=? AND id<? ORDER BY id`, target.SessionID, messageID)
	if err != nil {
		return Message{}, nil, err
	}
	defer rows.Close()
	history := []Message{}
	for rows.Next() {
		var item Message
		var traceJSON, variantsJSON string
		if err := rows.Scan(&item.ID, &item.SessionID, &item.Role, &item.Content, &item.Reasoning, &traceJSON, &variantsJSON, &item.CreatedAt); err != nil {
			return Message{}, nil, err
		}
		_ = json.Unmarshal([]byte(traceJSON), &item.ToolTrace)
		_ = json.Unmarshal([]byte(variantsJSON), &item.Variants)
		ensureCurrentVariant(&item)
		syncCurrentAttachments(&item)
		history = append(history, item)
	}
	if len(history) == 0 || history[len(history)-1].Role != "user" {
		return Message{}, nil, fmt.Errorf("assistant response has no preceding user message")
	}
	parent := &history[len(history)-1]
	if userVariant < 0 {
		userVariant = len(parent.Variants) - 1
	}
	if userVariant < 0 || userVariant >= len(parent.Variants) {
		return Message{}, nil, fmt.Errorf("invalid user variant %d", userVariant)
	}
	parent.Content = parent.Variants[userVariant].Content
	parent.Attachments = parent.Variants[userVariant].Attachments
	return target, history, rows.Err()
}

// EditContext returns a user message, the conversation before it, and the
// assistant response immediately following it when one exists.
func (d *DB) EditContext(messageID int64) (Message, *Message, []Message, error) {
	var target Message
	var traceJSON, variantsJSON string
	err := d.conn.QueryRow(`SELECT id,session_id,role,content,reasoning_content,tool_trace,response_variants,created_at FROM messages WHERE id=?`, messageID).
		Scan(&target.ID, &target.SessionID, &target.Role, &target.Content, &target.Reasoning, &traceJSON, &variantsJSON, &target.CreatedAt)
	if err != nil {
		return Message{}, nil, nil, err
	}
	if target.Role != "user" {
		return Message{}, nil, nil, fmt.Errorf("message %d is not a user request", messageID)
	}
	_ = json.Unmarshal([]byte(traceJSON), &target.ToolTrace)
	_ = json.Unmarshal([]byte(variantsJSON), &target.Variants)
	ensureCurrentVariant(&target)
	syncCurrentAttachments(&target)

	history, err := d.messagesBefore(target.SessionID, messageID)
	if err != nil {
		return Message{}, nil, nil, err
	}
	var next Message
	var nextTrace, nextVariants string
	err = d.conn.QueryRow(`SELECT id,session_id,role,content,reasoning_content,tool_trace,response_variants,created_at FROM messages WHERE session_id=? AND id>? ORDER BY id LIMIT 1`, target.SessionID, messageID).
		Scan(&next.ID, &next.SessionID, &next.Role, &next.Content, &next.Reasoning, &nextTrace, &nextVariants, &next.CreatedAt)
	if err == sql.ErrNoRows {
		return target, nil, history, nil
	}
	if err != nil {
		return Message{}, nil, nil, err
	}
	if next.Role != "assistant" {
		return target, nil, history, nil
	}
	_ = json.Unmarshal([]byte(nextTrace), &next.ToolTrace)
	_ = json.Unmarshal([]byte(nextVariants), &next.Variants)
	ensureCurrentVariant(&next)
	syncCurrentAttachments(&next)
	return target, &next, history, nil
}

func (d *DB) messagesBefore(sessionID string, messageID int64) ([]Message, error) {
	rows, err := d.conn.Query(`SELECT id,session_id,role,content,reasoning_content,tool_trace,response_variants,created_at FROM messages WHERE session_id=? AND id<? ORDER BY id`, sessionID, messageID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	history := []Message{}
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
		history = append(history, item)
	}
	return history, rows.Err()
}

// AppendEditedBranch commits a revised user request and its generated answer
// together. Existing request and response variants remain available.
func (d *DB) AppendEditedBranch(userMessageID int64, userContent string, userAttachments []Attachment, answer, reasoning string, toolTrace []ToolEvent) error {
	tx, err := d.conn.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	var sessionID, oldContent, variantsJSON string
	var oldCreatedAt time.Time
	if err := tx.QueryRow(`SELECT session_id,content,response_variants,created_at FROM messages WHERE id=? AND role='user'`, userMessageID).
		Scan(&sessionID, &oldContent, &variantsJSON, &oldCreatedAt); err != nil {
		return err
	}
	var userVariants []ResponseVariant
	_ = json.Unmarshal([]byte(variantsJSON), &userVariants)
	if len(userVariants) == 0 {
		userVariants = append(userVariants, ResponseVariant{Content: oldContent, CreatedAt: oldCreatedAt})
	}
	now := time.Now()
	userVariants = append(userVariants, ResponseVariant{Content: userContent, Attachments: userAttachments, CreatedAt: now})
	parentVariant := len(userVariants) - 1
	userVariantsJSON, _ := json.Marshal(userVariants)
	if _, err := tx.Exec(`UPDATE messages SET content=?,response_variants=?,created_at=? WHERE id=?`, userContent, string(userVariantsJSON), now, userMessageID); err != nil {
		return err
	}

	var assistantID int64
	var nextRole string
	var assistantContent, assistantReasoning, assistantTraceJSON, assistantVariantsJSON string
	var assistantCreatedAt time.Time
	err = tx.QueryRow(`SELECT id,role,content,reasoning_content,tool_trace,response_variants,created_at FROM messages WHERE session_id=? AND id>? ORDER BY id LIMIT 1`, sessionID, userMessageID).
		Scan(&assistantID, &nextRole, &assistantContent, &assistantReasoning, &assistantTraceJSON, &assistantVariantsJSON, &assistantCreatedAt)
	if err == nil && nextRole != "assistant" {
		err = sql.ErrNoRows
	}
	traceJSON, _ := json.Marshal(toolTrace)
	if err == sql.ErrNoRows {
		if _, deleteErr := tx.Exec(`DELETE FROM messages WHERE session_id=? AND id>?`, sessionID, userMessageID); deleteErr != nil {
			return deleteErr
		}
		answerVariants := []ResponseVariant{{Content: answer, Reasoning: reasoning, ToolTrace: toolTrace, ParentVariant: parentVariant, CreatedAt: now}}
		answerVariantsJSON, _ := json.Marshal(answerVariants)
		result, insertErr := tx.Exec(`INSERT INTO messages(session_id,role,content,reasoning_content,tool_trace,response_variants,created_at) VALUES(?,'assistant',?,?,?,?,?)`, sessionID, answer, reasoning, string(traceJSON), string(answerVariantsJSON), now)
		if insertErr != nil {
			return insertErr
		}
		assistantID, _ = result.LastInsertId()
	} else if err != nil {
		return err
	} else {
		var answerVariants []ResponseVariant
		_ = json.Unmarshal([]byte(assistantVariantsJSON), &answerVariants)
		if len(answerVariants) == 0 {
			var oldTrace []ToolEvent
			_ = json.Unmarshal([]byte(assistantTraceJSON), &oldTrace)
			answerVariants = append(answerVariants, ResponseVariant{Content: assistantContent, Reasoning: assistantReasoning, ToolTrace: oldTrace, CreatedAt: assistantCreatedAt})
		}
		answerVariants = append(answerVariants, ResponseVariant{Content: answer, Reasoning: reasoning, ToolTrace: toolTrace, ParentVariant: parentVariant, CreatedAt: now})
		answerVariantsJSON, _ := json.Marshal(answerVariants)
		if _, err := tx.Exec(`UPDATE messages SET content=?,reasoning_content=?,tool_trace=?,response_variants=?,created_at=? WHERE id=?`, answer, reasoning, string(traceJSON), string(answerVariantsJSON), now, assistantID); err != nil {
			return err
		}
	}
	if _, err := tx.Exec(`DELETE FROM messages WHERE session_id=? AND id>?`, sessionID, assistantID); err != nil {
		return err
	}
	if _, err := tx.Exec(`UPDATE sessions SET updated_at=? WHERE id=?`, now, sessionID); err != nil {
		return err
	}
	return tx.Commit()
}

// ReplaceAssistant atomically appends a regenerated response as a variant,
// selects it as the current response, and truncates the later branch. If
// generation fails this method is never called, preserving the conversation.
func (d *DB) ReplaceAssistant(messageID int64, content, reasoning string, toolTrace []ToolEvent, parentVariant int) error {
	tx, err := d.conn.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	var sessionID, oldContent, oldReasoning, oldTraceJSON, variantsJSON string
	var oldCreatedAt time.Time
	if err := tx.QueryRow(`SELECT session_id,content,reasoning_content,tool_trace,response_variants,created_at FROM messages WHERE id=? AND role='assistant'`, messageID).
		Scan(&sessionID, &oldContent, &oldReasoning, &oldTraceJSON, &variantsJSON, &oldCreatedAt); err != nil {
		return err
	}
	var variants []ResponseVariant
	_ = json.Unmarshal([]byte(variantsJSON), &variants)
	if len(variants) == 0 {
		var oldTrace []ToolEvent
		_ = json.Unmarshal([]byte(oldTraceJSON), &oldTrace)
		variants = append(variants, ResponseVariant{Content: oldContent, Reasoning: oldReasoning, ToolTrace: oldTrace, CreatedAt: oldCreatedAt})
	}
	now := time.Now()
	if parentVariant < 0 {
		parentVariant = 0
		var parentVariantsJSON string
		if err := tx.QueryRow(`SELECT response_variants FROM messages WHERE session_id=? AND role='user' AND id<? ORDER BY id DESC LIMIT 1`, sessionID, messageID).Scan(&parentVariantsJSON); err == nil {
			var parentVariants []ResponseVariant
			_ = json.Unmarshal([]byte(parentVariantsJSON), &parentVariants)
			if len(parentVariants) > 0 {
				parentVariant = len(parentVariants) - 1
			}
		}
	}
	variants = append(variants, ResponseVariant{Content: content, Reasoning: reasoning, ToolTrace: toolTrace, ParentVariant: parentVariant, CreatedAt: now})
	variantsJSONBytes, _ := json.Marshal(variants)
	traceJSON, _ := json.Marshal(toolTrace)
	if _, err := tx.Exec(`UPDATE messages SET content=?, reasoning_content=?, tool_trace=?, response_variants=?, created_at=? WHERE id=?`, content, reasoning, string(traceJSON), string(variantsJSONBytes), now, messageID); err != nil {
		return err
	}
	if _, err := tx.Exec(`DELETE FROM messages WHERE session_id=? AND id>?`, sessionID, messageID); err != nil {
		return err
	}
	if _, err := tx.Exec(`UPDATE sessions SET updated_at=? WHERE id=?`, time.Now(), sessionID); err != nil {
		return err
	}
	return tx.Commit()
}

func ensureCurrentVariant(message *Message) {
	if len(message.Variants) > 0 {
		return
	}
	message.Variants = []ResponseVariant{{
		Content: message.Content, Reasoning: message.Reasoning, ToolTrace: message.ToolTrace, Attachments: message.Attachments, CreatedAt: message.CreatedAt,
	}}
}

func syncCurrentAttachments(message *Message) {
	if len(message.Variants) > 0 {
		message.Attachments = message.Variants[len(message.Variants)-1].Attachments
	}
}
