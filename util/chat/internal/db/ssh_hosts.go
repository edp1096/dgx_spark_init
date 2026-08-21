package db

import (
	"database/sql"
	"errors"
	"time"
)

func (d *DB) SSHHosts() ([]SSHHost, error) {
	rows, err := d.conn.Query(`SELECT id,alias,name,hostname,port,username,key_id,timeout_seconds,created_at,updated_at FROM ssh_hosts ORDER BY name COLLATE NOCASE,alias`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	hosts := []SSHHost{}
	for rows.Next() {
		var host SSHHost
		if err := rows.Scan(&host.ID, &host.Alias, &host.Name, &host.Hostname, &host.Port, &host.Username, &host.KeyID, &host.TimeoutSeconds, &host.CreatedAt, &host.UpdatedAt); err != nil {
			return nil, err
		}
		hosts = append(hosts, host)
	}
	return hosts, rows.Err()
}

func (d *DB) SSHHost(id string) (SSHHost, error) {
	return scanSSHHost(d.conn.QueryRow(`SELECT id,alias,name,hostname,port,username,key_id,timeout_seconds,created_at,updated_at FROM ssh_hosts WHERE id=?`, id))
}

func (d *DB) SSHHostByAlias(alias string) (SSHHost, error) {
	return scanSSHHost(d.conn.QueryRow(`SELECT id,alias,name,hostname,port,username,key_id,timeout_seconds,created_at,updated_at FROM ssh_hosts WHERE alias=?`, alias))
}

func scanSSHHost(row *sql.Row) (SSHHost, error) {
	var host SSHHost
	err := row.Scan(&host.ID, &host.Alias, &host.Name, &host.Hostname, &host.Port, &host.Username, &host.KeyID, &host.TimeoutSeconds, &host.CreatedAt, &host.UpdatedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return SSHHost{}, errors.New("SSH server not found")
	}
	return host, err
}

func (d *DB) CreateSSHHost(host SSHHost) (SSHHost, error) {
	now := time.Now()
	host.CreatedAt, host.UpdatedAt = now, now
	_, err := d.conn.Exec(`INSERT INTO ssh_hosts(id,alias,name,hostname,port,username,key_id,timeout_seconds,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)`,
		host.ID, host.Alias, host.Name, host.Hostname, host.Port, host.Username, host.KeyID, host.TimeoutSeconds, now, now)
	return host, err
}

func (d *DB) UpdateSSHHost(host SSHHost) (SSHHost, error) {
	host.UpdatedAt = time.Now()
	result, err := d.conn.Exec(`UPDATE ssh_hosts SET alias=?,name=?,hostname=?,port=?,username=?,key_id=?,timeout_seconds=?,updated_at=? WHERE id=?`,
		host.Alias, host.Name, host.Hostname, host.Port, host.Username, host.KeyID, host.TimeoutSeconds, host.UpdatedAt, host.ID)
	if err != nil {
		return SSHHost{}, err
	}
	if changed, _ := result.RowsAffected(); changed == 0 {
		return SSHHost{}, errors.New("SSH server not found")
	}
	return d.SSHHost(host.ID)
}

func (d *DB) DeleteSSHHost(id string) error {
	tx, err := d.conn.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if _, err := tx.Exec(`DELETE FROM ssh_conversation_grants WHERE host_id=?`, id); err != nil {
		return err
	}
	result, err := tx.Exec(`DELETE FROM ssh_hosts WHERE id=?`, id)
	if err != nil {
		return err
	}
	if changed, _ := result.RowsAffected(); changed == 0 {
		return errors.New("SSH server not found")
	}
	return tx.Commit()
}
