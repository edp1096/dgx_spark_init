package db

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"time"
)

func (d *DB) initVoicePublications() error {
	_, err := d.conn.Exec(`CREATE TABLE IF NOT EXISTS voice_publications (
 job_id TEXT NOT NULL, version INTEGER NOT NULL, session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
 message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE, PRIMARY KEY(job_id,version))`)
	return err
}

func (d *DB) VoicePublication(session, job string, version int) (*Message, error) {
	var m Message
	var variants string
	err := d.conn.QueryRow(`SELECT m.id,m.session_id,m.content,m.response_variants,m.created_at FROM voice_publications p JOIN messages m ON m.id=p.message_id WHERE p.job_id=? AND p.version=? AND p.session_id=?`, job, version, session).Scan(&m.ID, &m.SessionID, &m.Content, &variants, &m.CreatedAt)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	m.Role = "assistant"
	m.Status = "completed"
	if err = json.Unmarshal([]byte(variants), &m.Variants); err != nil {
		return nil, err
	}
	syncCurrentAttachments(&m)
	return &m, nil
}

// The message and its idempotency key commit together; a lost callback response
// or worker restart must never duplicate a delivery.
func (d *DB) PublishVoice(session, job string, version int, content string, attachments []Attachment) (*Message, error) {
	if version < 1 || session == "" || job == "" {
		return nil, fmt.Errorf("invalid voice publication")
	}
	tx, err := d.conn.Begin()
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()
	var owner string
	err = tx.QueryRow(`SELECT session_id FROM voice_publications WHERE job_id=? AND version=?`, job, version).Scan(&owner)
	if err == nil {
		if owner != session {
			return nil, fmt.Errorf("voice publication belongs to another conversation")
		}
		if err = tx.Commit(); err != nil {
			return nil, err
		}
		return d.VoicePublication(session, job, version)
	}
	if err != sql.ErrNoRows {
		return nil, err
	}
	now := time.Now()
	variants := []ResponseVariant{{Content: content, Attachments: attachments, CreatedAt: now}}
	raw, _ := json.Marshal(variants)
	result, err := tx.Exec(`INSERT INTO messages(session_id,role,status,error,content,reasoning_content,tool_trace,response_variants,created_at) VALUES(?, 'assistant','completed','',?,'','[]',?,?)`, session, content, string(raw), now)
	if err != nil {
		return nil, err
	}
	id, err := result.LastInsertId()
	if err != nil {
		return nil, err
	}
	if _, err = tx.Exec(`INSERT INTO voice_publications(job_id,version,session_id,message_id) VALUES(?,?,?,?)`, job, version, session, id); err != nil {
		return nil, err
	}
	if _, err = tx.Exec(`UPDATE sessions SET updated_at=? WHERE id=?`, now, session); err != nil {
		return nil, err
	}
	if err = tx.Commit(); err != nil {
		return nil, err
	}
	return &Message{ID: id, SessionID: session, Role: "assistant", Status: "completed", Content: content, Attachments: attachments, Variants: variants, CreatedAt: now}, nil
}
