package db

// Archives are scoped to a conversation and cascade with its deletion.
func (d *DB) ArchiveContextTool(sessionID, name, content string, messageIDs ...int64) (int64, error) {
	var messageID any
	if len(messageIDs) > 0 && messageIDs[0] > 0 {
		messageID = messageIDs[0]
	}
	r, err := d.conn.Exec(`INSERT INTO context_tool_archive(session_id,name,content,message_id) VALUES(?,?,?,?)`, sessionID, name, content, messageID)
	if err != nil {
		return 0, err
	}
	return r.LastInsertId()
}
func (d *DB) ReadContextTool(sessionID string, id int64) (string, error) {
	var content string
	err := d.conn.QueryRow(`SELECT content FROM context_tool_archive WHERE session_id=? AND id=?`, sessionID, id).Scan(&content)
	return content, err
}

func (d *DB) ReadContextMessage(sessionID string, id int64) (string, error) {
	var text string
	err := d.conn.QueryRow(`SELECT content FROM messages WHERE session_id=? AND id=? AND status='completed'`, sessionID, id).Scan(&text)
	return text, err
}
