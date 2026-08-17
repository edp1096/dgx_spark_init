package db

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	_ "modernc.org/sqlite"
)

type DB struct{ conn *sql.DB }

type Session struct {
	ID        string    `json:"id"`
	Title     string    `json:"title"`
	Model     string    `json:"model"`
	Reasoning string    `json:"reasoning_effort"`
	GroupID   string    `json:"group_id"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

type Group struct {
	ID        string    `json:"id"`
	Name      string    `json:"name"`
	Position  int       `json:"position"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

type Message struct {
	ID          int64             `json:"id"`
	SessionID   string            `json:"session_id"`
	Role        string            `json:"role"`
	Content     string            `json:"content"`
	Reasoning   string            `json:"reasoning_content,omitempty"`
	ToolTrace   []ToolEvent       `json:"tool_trace,omitempty"`
	Attachments []Attachment      `json:"attachments,omitempty"`
	Variants    []ResponseVariant `json:"variants,omitempty"`
	CreatedAt   time.Time         `json:"created_at"`
}

type ToolEvent struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
	Result    string `json:"result"`
	Error     string `json:"error,omitempty"`
}

type Attachment struct {
	ID   string `json:"id"`
	Name string `json:"name"`
	MIME string `json:"mime"`
	Size int64  `json:"size"`
	URL  string `json:"url"`
}

type ResponseVariant struct {
	Content       string       `json:"content"`
	Reasoning     string       `json:"reasoning_content,omitempty"`
	ToolTrace     []ToolEvent  `json:"tool_trace,omitempty"`
	Attachments   []Attachment `json:"attachments,omitempty"`
	ParentVariant int          `json:"parent_variant,omitempty"`
	CreatedAt     time.Time    `json:"created_at"`
}

func Open(path string) (*DB, error) {
	conn, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}
	d := &DB{conn: conn}
	if _, err = conn.Exec(`
		PRAGMA foreign_keys=ON;
		PRAGMA journal_mode=WAL;
		PRAGMA busy_timeout=5000;
		CREATE TABLE IF NOT EXISTS sessions (
			id TEXT PRIMARY KEY, title TEXT NOT NULL, model TEXT NOT NULL DEFAULT '',
			reasoning_effort TEXT NOT NULL DEFAULT '',
			title_manual INTEGER NOT NULL DEFAULT 0,
			created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL
		);
		CREATE TABLE IF NOT EXISTS chat_groups (
			id TEXT PRIMARY KEY, name TEXT NOT NULL, position INTEGER NOT NULL,
			created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL
		);
		CREATE TABLE IF NOT EXISTS messages (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
			role TEXT NOT NULL, content TEXT NOT NULL,
			reasoning_content TEXT NOT NULL DEFAULT '', tool_trace TEXT NOT NULL DEFAULT '[]',
			response_variants TEXT NOT NULL DEFAULT '[]',
			created_at DATETIME NOT NULL
		);
		CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
	`); err != nil {
		conn.Close()
		return nil, fmt.Errorf("migrate: %w", err)
	}
	// Migrations from the initial MVP schema. Duplicate-column errors are safe.
	_, _ = conn.Exec(`ALTER TABLE sessions ADD COLUMN model TEXT NOT NULL DEFAULT ''`)
	_, _ = conn.Exec(`ALTER TABLE sessions ADD COLUMN reasoning_effort TEXT NOT NULL DEFAULT ''`)
	_, _ = conn.Exec(`ALTER TABLE sessions ADD COLUMN title_manual INTEGER NOT NULL DEFAULT 0`)
	_, _ = conn.Exec(`ALTER TABLE sessions ADD COLUMN group_id TEXT`)
	_, _ = conn.Exec(`ALTER TABLE messages ADD COLUMN reasoning_content TEXT NOT NULL DEFAULT ''`)
	_, _ = conn.Exec(`ALTER TABLE messages ADD COLUMN tool_trace TEXT NOT NULL DEFAULT '[]'`)
	_, _ = conn.Exec(`ALTER TABLE messages ADD COLUMN response_variants TEXT NOT NULL DEFAULT '[]'`)
	return d, nil
}

func (d *DB) Close() error { return d.conn.Close() }

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
