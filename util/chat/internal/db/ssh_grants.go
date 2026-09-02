package db

import "time"

func (d *DB) HasSSHConversationGrant(sessionID, hostID string) (bool, error) {
	return d.HasToolGrant(sessionID, "ssh_exec", hostID, "execute")
}

func (d *DB) GrantSSHConversation(sessionID, hostID string) error {
	return d.GrantToolConversation(sessionID, "ssh_exec", hostID, "execute")
}

func (d *DB) SSHConversationGrants(sessionID string) ([]SSHConversationGrant, error) {
	rows, err := d.conn.Query(`
		SELECT grant_row.session_id,grant_row.resource,host.alias,host.name,grant_row.created_at
		FROM tool_grants AS grant_row
		JOIN ssh_hosts AS host ON host.id=grant_row.resource
		WHERE grant_row.scope='conversation' AND grant_row.session_id=? AND grant_row.tool_name='ssh_exec' AND grant_row.action='execute'
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
	_, err := d.conn.Exec(`DELETE FROM tool_grants WHERE scope='conversation' AND session_id=? AND tool_name='ssh_exec' AND resource=?`, sessionID, hostID)
	return err
}

func (d *DB) DeleteSSHConversationGrants(sessionID string) error {
	_, err := d.conn.Exec(`DELETE FROM tool_grants WHERE scope='conversation' AND session_id=? AND tool_name='ssh_exec'`, sessionID)
	return err
}

func (d *DB) DeleteSSHHostGrants(hostID string) error {
	_, err := d.conn.Exec(`DELETE FROM tool_grants WHERE tool_name='ssh_exec' AND resource=?`, hostID)
	return err
}

func (d *DB) HasToolGrant(sessionID, toolName, resource, action string) (bool, error) {
	var exists int
	err := d.conn.QueryRow(`SELECT EXISTS(
		SELECT 1 FROM tool_grants
		WHERE tool_name=? AND resource=? AND action=?
		  AND ((scope='conversation' AND session_id=?) OR scope='always')
	)`, toolName, resource, action, sessionID).Scan(&exists)
	return exists != 0, err
}

func (d *DB) GrantToolConversation(sessionID, toolName, resource, action string) error {
	_, err := d.conn.Exec(`INSERT INTO tool_grants(scope,session_id,tool_name,resource,action,created_at) VALUES('conversation',?,?,?,?,?) ON CONFLICT(scope,session_id,tool_name,resource,action) DO NOTHING`,
		sessionID, toolName, resource, action, time.Now())
	return err
}

func (d *DB) AddToolAudit(sessionID, toolName, resource, action, decision, detail string) error {
	_, err := d.conn.Exec(`INSERT INTO tool_audit(session_id,tool_name,resource,action,decision,detail,created_at) VALUES(?,?,?,?,?,?,?)`,
		sessionID, toolName, resource, action, decision, detail, time.Now())
	return err
}

func (d *DB) ToolAudits(limit int) ([]ToolAudit, error) {
	if limit < 1 {
		limit = 20
	}
	if limit > 100 {
		limit = 100
	}
	rows, err := d.conn.Query(`SELECT id,session_id,tool_name,resource,action,decision,detail,created_at FROM tool_audit ORDER BY id DESC LIMIT ?`, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	items := []ToolAudit{}
	for rows.Next() {
		var item ToolAudit
		if err := rows.Scan(&item.ID, &item.SessionID, &item.ToolName, &item.Resource, &item.Action, &item.Decision, &item.Detail, &item.CreatedAt); err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	return items, rows.Err()
}
