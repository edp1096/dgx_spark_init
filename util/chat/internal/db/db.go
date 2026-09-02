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

type Memory struct {
	ID              int64     `json:"id"`
	Kind            string    `json:"kind"`
	Title           string    `json:"title"`
	Content         string    `json:"content"`
	Enabled         bool      `json:"enabled"`
	SourceSessionID string    `json:"source_session_id,omitempty"`
	SourceMessageID int64     `json:"source_message_id,omitempty"`
	CreatedAt       time.Time `json:"created_at"`
	UpdatedAt       time.Time `json:"updated_at"`
}

type RecallItem struct {
	Kind      string    `json:"kind"`
	Title     string    `json:"title"`
	Role      string    `json:"role,omitempty"`
	Content   string    `json:"content"`
	SessionID string    `json:"session_id,omitempty"`
	MessageID int64     `json:"message_id,omitempty"`
	CreatedAt time.Time `json:"created_at"`
}

type ConversationSearchOptions struct {
	Limit      int
	Sort       string
	Scope      string
	DateFrom   string
	DateTo     string
	CursorID   int64
	CursorRank float64
}

type ConversationSearchCursor struct {
	MessageID int64   `json:"message_id"`
	Rank      float64 `json:"rank,omitempty"`
}

type ToolAudit struct {
	ID        int64     `json:"id"`
	SessionID string    `json:"session_id,omitempty"`
	ToolName  string    `json:"tool_name"`
	Resource  string    `json:"resource,omitempty"`
	Action    string    `json:"action"`
	Decision  string    `json:"decision"`
	Detail    string    `json:"detail,omitempty"`
	CreatedAt time.Time `json:"created_at"`
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
		CREATE TABLE IF NOT EXISTS memories (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			kind TEXT NOT NULL CHECK(kind IN ('user','memory')),
			title TEXT NOT NULL DEFAULT '',
			content TEXT NOT NULL,
			enabled INTEGER NOT NULL DEFAULT 1,
			source_session_id TEXT NOT NULL DEFAULT '',
			source_message_id INTEGER NOT NULL DEFAULT 0,
			created_at DATETIME NOT NULL,
			updated_at DATETIME NOT NULL
		);
		CREATE INDEX IF NOT EXISTS idx_memories_kind_updated ON memories(kind, enabled, updated_at DESC);
		CREATE VIRTUAL TABLE IF NOT EXISTS memory_search USING fts5(
			kind UNINDEXED, title, content, tokenize='trigram'
		);
		CREATE VIRTUAL TABLE IF NOT EXISTS message_search USING fts5(
			session_id UNINDEXED, message_id UNINDEXED, title, role UNINDEXED,
			content, created_at UNINDEXED, tokenize='trigram'
		);
		CREATE TRIGGER IF NOT EXISTS memories_search_insert AFTER INSERT ON memories
		WHEN NEW.enabled=1 AND length(trim(NEW.content))>0 BEGIN
			INSERT INTO memory_search(rowid,kind,title,content) VALUES(NEW.id,NEW.kind,NEW.title,NEW.content);
		END;
		CREATE TRIGGER IF NOT EXISTS memories_search_update AFTER UPDATE ON memories BEGIN
			DELETE FROM memory_search WHERE rowid=OLD.id;
			INSERT INTO memory_search(rowid,kind,title,content)
				SELECT NEW.id,NEW.kind,NEW.title,NEW.content WHERE NEW.enabled=1 AND length(trim(NEW.content))>0;
		END;
		CREATE TRIGGER IF NOT EXISTS memories_search_delete AFTER DELETE ON memories BEGIN
			DELETE FROM memory_search WHERE rowid=OLD.id;
		END;
		CREATE TRIGGER IF NOT EXISTS messages_search_insert AFTER INSERT ON messages
		WHEN NEW.status='completed' AND length(trim(NEW.content))>0 BEGIN
			INSERT INTO message_search(rowid,session_id,message_id,title,role,content,created_at)
				SELECT NEW.id,NEW.session_id,NEW.id,s.title,NEW.role,NEW.content,NEW.created_at
				FROM sessions AS s WHERE s.id=NEW.session_id;
		END;
		CREATE TRIGGER IF NOT EXISTS messages_search_update AFTER UPDATE OF content,status ON messages BEGIN
			DELETE FROM message_search WHERE rowid=OLD.id;
			INSERT INTO message_search(rowid,session_id,message_id,title,role,content,created_at)
				SELECT NEW.id,NEW.session_id,NEW.id,s.title,NEW.role,NEW.content,NEW.created_at
				FROM sessions AS s WHERE s.id=NEW.session_id AND NEW.status='completed' AND length(trim(NEW.content))>0;
		END;
		CREATE TRIGGER IF NOT EXISTS messages_search_delete AFTER DELETE ON messages BEGIN
			DELETE FROM message_search WHERE rowid=OLD.id;
		END;
		CREATE TRIGGER IF NOT EXISTS sessions_search_title AFTER UPDATE OF title ON sessions BEGIN
			UPDATE message_search SET title=NEW.title WHERE session_id=NEW.id;
		END;
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
		CREATE TABLE IF NOT EXISTS tool_grants (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			scope TEXT NOT NULL CHECK(scope IN ('conversation','always')),
			session_id TEXT NOT NULL DEFAULT '',
			tool_name TEXT NOT NULL,
			resource TEXT NOT NULL DEFAULT '',
			action TEXT NOT NULL DEFAULT '*',
			created_at DATETIME NOT NULL,
			UNIQUE(scope,session_id,tool_name,resource,action)
		);
		CREATE INDEX IF NOT EXISTS idx_tool_grants_lookup ON tool_grants(tool_name,resource,action,scope,session_id);
		CREATE TABLE IF NOT EXISTS tool_audit (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			session_id TEXT NOT NULL DEFAULT '',
			tool_name TEXT NOT NULL,
			resource TEXT NOT NULL DEFAULT '',
			action TEXT NOT NULL DEFAULT '',
			decision TEXT NOT NULL,
			detail TEXT NOT NULL DEFAULT '',
			created_at DATETIME NOT NULL
		);
		CREATE INDEX IF NOT EXISTS idx_tool_audit_created ON tool_audit(created_at DESC);
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
	_, _ = conn.Exec(`
		INSERT INTO memory_search(rowid,kind,title,content)
		SELECT memory_row.id,memory_row.kind,memory_row.title,memory_row.content
		FROM memories AS memory_row
		WHERE memory_row.enabled=1 AND length(trim(memory_row.content))>0
		  AND NOT EXISTS(SELECT 1 FROM memory_search WHERE rowid=memory_row.id);
		INSERT INTO message_search(rowid,session_id,message_id,title,role,content,created_at)
		SELECT message_row.id,message_row.session_id,message_row.id,session_row.title,message_row.role,message_row.content,message_row.created_at
		FROM messages AS message_row JOIN sessions AS session_row ON session_row.id=message_row.session_id
		WHERE message_row.status='completed' AND length(trim(message_row.content))>0
		  AND NOT EXISTS(SELECT 1 FROM message_search WHERE rowid=message_row.id);
	`)
	_, _ = conn.Exec(`
		INSERT INTO tool_grants(scope,session_id,tool_name,resource,action,created_at)
		SELECT 'conversation',session_id,'ssh_exec',host_id,'execute',created_at
		FROM ssh_conversation_grants
		WHERE 1=1
		ON CONFLICT(scope,session_id,tool_name,resource,action) DO NOTHING;
		DELETE FROM ssh_conversation_grants;
	`)
	// Older builds relied on the connection-local foreign_keys pragma when a
	// conversation was deleted. Pooled SQLite connections could therefore
	// leave invisible messages and checkpoints behind. Remove only records
	// whose parent conversation is already gone.
	_, _ = conn.Exec(`
		DELETE FROM context_segments
		WHERE NOT EXISTS(SELECT 1 FROM sessions WHERE sessions.id=context_segments.session_id);
		DELETE FROM messages
		WHERE NOT EXISTS(SELECT 1 FROM sessions WHERE sessions.id=messages.session_id);
		DELETE FROM tool_grants
		WHERE scope='conversation' AND NOT EXISTS(SELECT 1 FROM sessions WHERE sessions.id=tool_grants.session_id);
		DELETE FROM tool_audit
		WHERE session_id<>'' AND NOT EXISTS(SELECT 1 FROM sessions WHERE sessions.id=tool_audit.session_id);
	`)
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
