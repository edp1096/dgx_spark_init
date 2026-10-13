package db

import (
	"database/sql"
	"fmt"
	"strings"
	"time"
)

type ConversationSummary struct {
	ID            string    `json:"session_id"`
	Title         string    `json:"title"`
	GroupID       string    `json:"group_id"`
	GroupName     string    `json:"group_name"`
	UpdatedAt     time.Time `json:"updated_at"`
	LastMessageID int64     `json:"last_message_id"`
	MessageCount  int       `json:"message_count"`
	Snippet       string    `json:"snippet"`
	DocumentKinds string    `json:"document_kinds"`
	rowID         int64
}

const conversationSummarySelect = `SELECT s.rowid,s.id,s.title,COALESCE(s.group_id,''),COALESCE(g.name,''),s.updated_at,
 COALESCE((SELECT MAX(id) FROM messages WHERE session_id=s.id),0),
 (SELECT COUNT(*) FROM messages WHERE session_id=s.id),
 COALESCE((SELECT substr(content,1,400) FROM messages WHERE session_id=s.id AND role='user' ORDER BY id LIMIT 1),''),
 COALESCE((SELECT group_concat(DISTINCT json_extract(data,'$.document.kind')) FROM workflow_runs WHERE session_id=s.id),'')
 FROM sessions s LEFT JOIN chat_groups g ON g.id=s.group_id`

type conversationScanner interface{ Scan(...any) error }

func scanConversation(row conversationScanner) (ConversationSummary, error) {
	var s ConversationSummary
	err := row.Scan(&s.rowID, &s.ID, &s.Title, &s.GroupID, &s.GroupName, &s.UpdatedAt, &s.LastMessageID, &s.MessageCount, &s.Snippet, &s.DocumentKinds)
	return s, err
}

func (d *DB) ConversationSummary(id string) (ConversationSummary, error) {
	return scanConversation(d.conn.QueryRow(conversationSummarySelect+` WHERE s.id=?`, id))
}

// Page distinct conversations, not messages, so one long book cannot consume
// every search result. Short search terms retain the existing substring fallback.
func (d *DB) FindConversations(query, kind, exclude string, before int64, limit int) ([]ConversationSummary, int64, error) {
	limit = max(1, min(limit, 50))
	where := []string{"s.id<>?"}
	criteria := []string{}
	args := []any{exclude}
	if before > 0 {
		where = append(where, "s.rowid<?")
		args = append(args, before)
	}
	if kind != "" {
		criteria = append(criteria, `EXISTS(SELECT 1 FROM workflow_runs w WHERE w.session_id=s.id AND json_extract(w.data,'$.document.kind')=?)`)
		args = append(args, kind)
	}
	if query != "" {
		match := ftsMatchQuery(manualSearchTerms(query))
		if match != "" {
			criteria = append(criteria, `(instr(lower(s.title),lower(?))>0 OR s.id IN (SELECT m.session_id FROM message_search JOIN messages m ON m.id=message_search.rowid WHERE message_search MATCH ? AND m.status='completed'))`)
			args = append(args, query, "content : ("+match+")")
		} else {
			criteria = append(criteria, `(instr(lower(s.title),lower(?))>0 OR EXISTS(SELECT 1 FROM messages m WHERE m.session_id=s.id AND m.status='completed' AND instr(lower(m.content),lower(?))>0))`)
			args = append(args, query, query)
		}
	}
	if len(criteria) > 0 {
		where = append(where, "("+strings.Join(criteria, " OR ")+")")
	}
	args = append(args, limit+1)
	rows, err := d.conn.Query(conversationSummarySelect+" WHERE "+strings.Join(where, " AND ")+" ORDER BY s.rowid DESC LIMIT ?", args...)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()
	items := []ConversationSummary{}
	for rows.Next() {
		item, err := scanConversation(rows)
		if err != nil {
			return nil, 0, err
		}
		items = append(items, item)
	}
	if err = rows.Err(); err != nil {
		return nil, 0, err
	}
	var next int64
	if len(items) > limit {
		items = items[:limit]
		next = items[len(items)-1].rowID
	}
	return items, next, nil
}

type ConversationMessage struct {
	ID         int64  `json:"message_id"`
	Role       string `json:"role"`
	Text       string `json:"text"`
	Characters int    `json:"total_chars"`
	NextOffset int    `json:"next_offset,omitempty"`
}

// Read bounded excerpts without allocating full book-sized messages, reasoning,
// attachments or tool logs. A single long message can be paged by character.
func (d *DB) ConversationMessages(id string, after, message int64, offset, limit int) ([]ConversationMessage, int64, error) {
	limit = max(1, min(limit, 6))
	offset = max(0, offset)
	where := `session_id=? AND role IN ('user','assistant') AND status='completed'`
	args := []any{id}
	if message > 0 {
		where += " AND id=?"
		args = append(args, message)
		limit = 1
	} else {
		where += " AND id>?"
		args = append(args, after)
	}
	args = append([]any{offset + 1}, args...)
	args = append(args, limit+1)
	rows, err := d.conn.Query(`SELECT id,role,substr(content,?,1600),length(content) FROM messages WHERE `+where+` ORDER BY id LIMIT ?`, args...)
	if err != nil {
		return nil, 0, err
	}
	defer rows.Close()
	items := []ConversationMessage{}
	for rows.Next() {
		var m ConversationMessage
		if err = rows.Scan(&m.ID, &m.Role, &m.Text, &m.Characters); err != nil {
			return nil, 0, err
		}
		if m.Characters > offset+1600 {
			m.NextOffset = offset + 1600
		}
		items = append(items, m)
	}
	if err = rows.Err(); err != nil {
		return nil, 0, err
	}
	var next int64
	if len(items) > limit {
		items = items[:limit]
		next = items[len(items)-1].ID
	}
	return items, next, nil
}

type ConversationMoveResult struct {
	ID      string `json:"session_id"`
	Title   string `json:"title"`
	Status  string `json:"status"`
	GroupID string `json:"group_id"`
}

// Move only the conversations whose observed metadata/content version still
// matches. Conflicts are reported individually; database errors roll back all.
func (d *DB) MoveConversationsFromTool(targets []ConversationSummary, destination, expectedName string) ([]ConversationMoveResult, error) {
	if len(targets) < 1 || len(targets) > 50 {
		return nil, fmt.Errorf("select 1–50 conversations")
	}
	tx, err := d.conn.Begin()
	if err != nil {
		return nil, err
	}
	defer tx.Rollback()
	if destination != "" {
		var name string
		if err = tx.QueryRow(`SELECT name FROM chat_groups WHERE id=?`, destination).Scan(&name); err != nil {
			return nil, err
		}
		if name != expectedName {
			return nil, fmt.Errorf("destination folder changed; list folders again")
		}
	}
	seen := map[string]bool{}
	results := []ConversationMoveResult{}
	for _, target := range targets {
		if target.ID == "" || seen[target.ID] {
			return nil, fmt.Errorf("invalid or duplicate conversation ID")
		}
		seen[target.ID] = true
		current, err := scanConversation(tx.QueryRow(conversationSummarySelect+` WHERE s.id=?`, target.ID))
		if err == sql.ErrNoRows {
			results = append(results, ConversationMoveResult{ID: target.ID, Title: target.Title, Status: "not_found"})
			continue
		}
		if err != nil {
			return nil, err
		}
		result := ConversationMoveResult{ID: current.ID, Title: current.Title, GroupID: current.GroupID, Status: "unchanged"}
		if current.GroupID != destination {
			if current.Title != target.Title || current.GroupID != target.GroupID || current.LastMessageID != target.LastMessageID || !current.UpdatedAt.Equal(target.UpdatedAt) {
				result.Status = "conflict"
			} else {
				var group any
				if destination != "" {
					group = destination
				}
				if _, err = tx.Exec(`UPDATE sessions SET group_id=?,updated_at=? WHERE id=?`, group, time.Now(), target.ID); err != nil {
					return nil, err
				}
				result.Status = "moved"
				result.GroupID = destination
			}
		}
		results = append(results, result)
	}
	return results, tx.Commit()
}
