package db

import "time"

func (d *DB) HasSSHConversationGrant(sessionID, hostID string) (bool, error) {
	var exists int
	err := d.conn.QueryRow(`SELECT EXISTS(SELECT 1 FROM ssh_conversation_grants WHERE session_id=? AND host_id=?)`, sessionID, hostID).Scan(&exists)
	return exists != 0, err
}

func (d *DB) GrantSSHConversation(sessionID, hostID string) error {
	_, err := d.conn.Exec(`INSERT INTO ssh_conversation_grants(session_id,host_id,created_at) VALUES(?,?,?) ON CONFLICT(session_id,host_id) DO NOTHING`, sessionID, hostID, time.Now())
	return err
}

func (d *DB) SSHConversationGrants(sessionID string) ([]SSHConversationGrant, error) {
	rows, err := d.conn.Query(`
		SELECT grant_row.session_id,grant_row.host_id,host.alias,host.name,grant_row.created_at
		FROM ssh_conversation_grants AS grant_row
		JOIN ssh_hosts AS host ON host.id=grant_row.host_id
		WHERE grant_row.session_id=?
		ORDER BY host.name COLLATE NOCASE,host.alias`, sessionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	grants := []SSHConversationGrant{}
	for rows.Next() {
		var grant SSHConversationGrant
		if err := rows.Scan(&grant.SessionID, &grant.HostID, &grant.HostAlias, &grant.HostName, &grant.CreatedAt); err != nil {
			return nil, err
		}
		grants = append(grants, grant)
	}
	return grants, rows.Err()
}

func (d *DB) DeleteSSHConversationGrant(sessionID, hostID string) error {
	_, err := d.conn.Exec(`DELETE FROM ssh_conversation_grants WHERE session_id=? AND host_id=?`, sessionID, hostID)
	return err
}

func (d *DB) DeleteSSHConversationGrants(sessionID string) error {
	_, err := d.conn.Exec(`DELETE FROM ssh_conversation_grants WHERE session_id=?`, sessionID)
	return err
}

func (d *DB) DeleteSSHHostGrants(hostID string) error {
	_, err := d.conn.Exec(`DELETE FROM ssh_conversation_grants WHERE host_id=?`, hostID)
	return err
}
