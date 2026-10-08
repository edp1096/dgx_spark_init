package db

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
)

type TurnInput struct {
	ID      string `json:"id"`
	Content string `json:"content"`
}

func inputBase(content string) string {
	h := sha256.Sum256([]byte(content))
	return hex.EncodeToString(h[:])
}

// Keep follow-ups separate from the original text and its edit variants.
func (d *DB) SaveTurnInput(session string, messageID int64, base string, input TurnInput) (bool, error) {
	r, err := d.conn.Exec(`INSERT OR IGNORE INTO turn_inputs(message_id,base_hash,input_id,content)
 SELECT id,?,?,? FROM messages WHERE id=? AND session_id=? AND role='user'`, inputBase(base), input.ID, input.Content, messageID, session)
	if err != nil {
		return false, err
	}
	n, err := r.RowsAffected()
	if err != nil {
		return false, err
	}
	if n == 0 {
		var content string
		err = d.conn.QueryRow(`SELECT t.content FROM turn_inputs t JOIN messages m ON m.id=t.message_id WHERE t.message_id=? AND t.base_hash=? AND t.input_id=? AND m.session_id=?`, messageID, inputBase(base), input.ID, session).Scan(&content)
		if err != nil {
			return false, err
		}
		if content != input.Content {
			return false, fmt.Errorf("additional input ID was reused with different text")
		}
	}
	return n > 0, nil
}

func (d *DB) TurnInputs(messageID int64, base string) ([]TurnInput, error) {
	rows, err := d.conn.Query(`SELECT input_id,content FROM turn_inputs WHERE message_id=? AND base_hash=? ORDER BY seq`, messageID, inputBase(base))
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []TurnInput
	for rows.Next() {
		var v TurnInput
		if err := rows.Scan(&v.ID, &v.Content); err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	return out, rows.Err()
}

func (d *DB) loadTurnInputs(items []Message) error {
	for i := range items {
		m := &items[i]
		if m.Role != "user" {
			continue
		}
		var err error
		m.TurnInputs, err = d.TurnInputs(m.ID, m.Content)
		if err != nil {
			return err
		}
		for j := range m.Variants {
			m.Variants[j].TurnInputs, err = d.TurnInputs(m.ID, m.Variants[j].Content)
			if err != nil {
				return err
			}
		}
	}
	return nil
}
