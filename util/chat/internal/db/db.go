package db

import (
	"database/sql"
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
