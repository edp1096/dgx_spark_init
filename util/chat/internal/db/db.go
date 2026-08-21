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
	Status      string            `json:"status"`
	Error       string            `json:"error,omitempty"`
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
	ID        string `json:"id"`
	Name      string `json:"name"`
	MIME      string `json:"mime"`
	Size      int64  `json:"size"`
	URL       string `json:"url"`
	SourceURL string `json:"source_url,omitempty"`
}

type ResponseVariant struct {
	Content       string       `json:"content"`
	Reasoning     string       `json:"reasoning_content,omitempty"`
	ToolTrace     []ToolEvent  `json:"tool_trace,omitempty"`
	Attachments   []Attachment `json:"attachments,omitempty"`
	ParentVariant int          `json:"parent_variant,omitempty"`
	CreatedAt     time.Time    `json:"created_at"`
}

type ContextSegment struct {
	ID              int64     `json:"id"`
	SessionID       string    `json:"session_id"`
	StartMessageID  int64     `json:"start_message_id"`
	EndMessageID    int64     `json:"end_message_id"`
	Summary         string    `json:"summary"`
	Checkpoint      string    `json:"-"`
	EstimatedTokens int       `json:"estimated_tokens"`
	Model           string    `json:"model"`
	CreatedAt       time.Time `json:"created_at"`
}

type SSHHost struct {
	ID             string    `json:"id"`
	Alias          string    `json:"alias"`
	Name           string    `json:"name"`
	Hostname       string    `json:"hostname"`
	Port           int       `json:"port"`
	Username       string    `json:"username"`
	KeyID          string    `json:"key_id"`
	TimeoutSeconds int       `json:"timeout_seconds"`
	CreatedAt      time.Time `json:"created_at"`
	UpdatedAt      time.Time `json:"updated_at"`
}

type SSHConversationGrant struct {
	SessionID string    `json:"session_id"`
	HostID    string    `json:"host_id"`
	HostAlias string    `json:"host_alias"`
	HostName  string    `json:"host_name"`
	CreatedAt time.Time `json:"created_at"`
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
			status TEXT NOT NULL DEFAULT 'completed', error TEXT NOT NULL DEFAULT '',
			reasoning_content TEXT NOT NULL DEFAULT '', tool_trace TEXT NOT NULL DEFAULT '[]',
			response_variants TEXT NOT NULL DEFAULT '[]',
			created_at DATETIME NOT NULL
		);
		CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
		CREATE TABLE IF NOT EXISTS context_segments (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
			start_message_id INTEGER NOT NULL,
			end_message_id INTEGER NOT NULL,
			summary TEXT NOT NULL,
			checkpoint TEXT NOT NULL,
			estimated_tokens INTEGER NOT NULL DEFAULT 0,
			model TEXT NOT NULL DEFAULT '',
			created_at DATETIME NOT NULL,
			UNIQUE(session_id, end_message_id)
		);
		CREATE INDEX IF NOT EXISTS idx_context_segments_session ON context_segments(session_id, end_message_id);
		CREATE TABLE IF NOT EXISTS ssh_hosts (
			id TEXT PRIMARY KEY,
			alias TEXT NOT NULL UNIQUE,
			name TEXT NOT NULL,
			hostname TEXT NOT NULL,
			port INTEGER NOT NULL DEFAULT 22,
			username TEXT NOT NULL,
			key_id TEXT NOT NULL,
			timeout_seconds INTEGER NOT NULL DEFAULT 60,
			created_at DATETIME NOT NULL,
			updated_at DATETIME NOT NULL
		);
		CREATE TABLE IF NOT EXISTS ssh_conversation_grants (
			session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
			host_id TEXT NOT NULL REFERENCES ssh_hosts(id) ON DELETE CASCADE,
			created_at DATETIME NOT NULL,
			PRIMARY KEY(session_id, host_id)
		);
		CREATE INDEX IF NOT EXISTS idx_ssh_conversation_grants_host ON ssh_conversation_grants(host_id);
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
	_, _ = conn.Exec(`ALTER TABLE messages ADD COLUMN status TEXT NOT NULL DEFAULT 'completed'`)
	_, _ = conn.Exec(`ALTER TABLE messages ADD COLUMN error TEXT NOT NULL DEFAULT ''`)
	// A process restart turns abandoned in-flight requests into failures. Older
	// databases had no status column; a user request without an immediately
	// following assistant response is a request that never completed.
	_, _ = conn.Exec(`UPDATE messages SET status='failed', error=CASE WHEN error='' THEN '이 요청은 완료되지 않았습니다.' ELSE error END WHERE status='pending'`)
	_, _ = conn.Exec(`
		UPDATE messages AS user_message
		SET status='failed', error=CASE WHEN error='' THEN '이전 모델 요청이 완료되지 않았습니다.' ELSE error END
		WHERE role='user' AND status='completed'
		  AND NOT EXISTS (
			SELECT 1 FROM messages AS answer
			WHERE answer.id=(SELECT MIN(next_message.id) FROM messages AS next_message WHERE next_message.session_id=user_message.session_id AND next_message.id>user_message.id)
			  AND answer.role='assistant' AND answer.status='completed'
		  )
	`)
	return d, nil
}

func (d *DB) Close() error { return d.conn.Close() }
