package db

import (
	"database/sql"
	"errors"
	"time"
)

func (d *DB) ContextSegments(sessionID string) ([]ContextSegment, error) {
	rows, err := d.conn.Query(`SELECT id,session_id,start_message_id,end_message_id,summary,checkpoint,estimated_tokens,model,created_at FROM context_segments WHERE session_id=? ORDER BY end_message_id`, sessionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	segments := []ContextSegment{}
	for rows.Next() {
		var item ContextSegment
		if err := rows.Scan(&item.ID, &item.SessionID, &item.StartMessageID, &item.EndMessageID, &item.Summary, &item.Checkpoint, &item.EstimatedTokens, &item.Model, &item.CreatedAt); err != nil {
			return nil, err
		}
		segments = append(segments, item)
	}
	return segments, rows.Err()
}

func (d *DB) LatestContextSegment(sessionID string) (ContextSegment, bool, error) {
	var item ContextSegment
	err := d.conn.QueryRow(`SELECT id,session_id,start_message_id,end_message_id,summary,checkpoint,estimated_tokens,model,created_at FROM context_segments WHERE session_id=? ORDER BY end_message_id DESC LIMIT 1`, sessionID).
		Scan(&item.ID, &item.SessionID, &item.StartMessageID, &item.EndMessageID, &item.Summary, &item.Checkpoint, &item.EstimatedTokens, &item.Model, &item.CreatedAt)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return ContextSegment{}, false, nil
		}
		return ContextSegment{}, false, err
	}
	return item, true, nil
}

func (d *DB) AddContextSegment(sessionID string, startID, endID int64, summary, checkpoint, model string, estimatedTokens int) (ContextSegment, error) {
	now := time.Now()
	result, err := d.conn.Exec(`INSERT INTO context_segments(session_id,start_message_id,end_message_id,summary,checkpoint,estimated_tokens,model,created_at) VALUES(?,?,?,?,?,?,?,?)`,
		sessionID, startID, endID, summary, checkpoint, estimatedTokens, model, now)
	if err != nil {
		return ContextSegment{}, err
	}
	id, _ := result.LastInsertId()
	return ContextSegment{ID: id, SessionID: sessionID, StartMessageID: startID, EndMessageID: endID, Summary: summary, Checkpoint: checkpoint, EstimatedTokens: estimatedTokens, Model: model, CreatedAt: now}, nil
}

func (d *DB) DeleteContextSegmentsFrom(sessionID string, messageID int64) error {
	_, err := d.conn.Exec(`DELETE FROM context_segments WHERE session_id=? AND end_message_id>=?`, sessionID, messageID)
	return err
}

func (d *DB) ClearContextSegments(sessionID string) error {
	_, err := d.conn.Exec(`DELETE FROM context_segments WHERE session_id=?`, sessionID)
	return err
}
