package db

import (
	"encoding/json"
	"fmt"
	"sparktalk/internal/workflows"
	"time"
)

func (d *DB) Workflows() ([]workflows.Definition, error) {
	rows, e := d.conn.Query(`SELECT data FROM workflows ORDER BY name`)
	if e != nil {
		return nil, e
	}
	defer rows.Close()
	out := []workflows.Definition{}
	for rows.Next() {
		var raw string
		var x workflows.Definition
		if e = rows.Scan(&raw); e != nil {
			return nil, e
		}
		if e = json.Unmarshal([]byte(raw), &x); e != nil {
			return nil, e
		}
		out = append(out, x)
	}
	return out, rows.Err()
}
func (d *DB) SaveWorkflow(x workflows.Definition) error {
	b, e := json.Marshal(x)
	if e != nil {
		return e
	}
	_, e = d.conn.Exec(`INSERT INTO workflows(name,data) VALUES(?,?) ON CONFLICT(name) DO UPDATE SET data=excluded.data`, x.Name, string(b))
	return e
}
func (d *DB) DeleteWorkflow(name string) error {
	_, e := d.conn.Exec(`DELETE FROM workflows WHERE name=?`, name)
	return e
}
func (d *DB) SaveWorkflowRun(x *workflows.Run) error {
	x.Updated = time.Now().UTC().Format(time.RFC3339Nano)
	b, e := json.Marshal(x)
	if e != nil {
		return e
	}
	_, e = d.conn.Exec(`INSERT INTO workflow_runs(id,session_id,status,data,updated) VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status,data=excluded.data,updated=excluded.updated`, x.ID, x.SessionID, x.Status, string(b), x.Updated)
	return e
}
func (d *DB) WorkflowRuns(session string) ([]workflows.Run, error) {
	rows, e := d.conn.Query(`SELECT data,status FROM workflow_runs WHERE session_id=? ORDER BY updated DESC LIMIT 20`, session)
	if e != nil {
		return nil, e
	}
	defer rows.Close()
	out := []workflows.Run{}
	for rows.Next() {
		var raw, status string
		var x workflows.Run
		if e = rows.Scan(&raw, &status); e != nil {
			return nil, e
		}
		if e = json.Unmarshal([]byte(raw), &x); e != nil {
			return nil, e
		}
		x.Status = status
		if status == "paused" {
			for i := range x.Steps {
				if x.Steps[i].Status == "running" {
					x.Steps[i].Status = "paused"
				}
			}
		}
		out = append(out, x)
	}
	return out, rows.Err()
}
func (d *DB) ClaimWorkflowRun(id, session string) (workflows.Run, error) {
	tx, e := d.conn.Begin()
	if e != nil {
		return workflows.Run{}, e
	}
	defer tx.Rollback()
	var raw, status string
	if e = tx.QueryRow(`SELECT data,status FROM workflow_runs WHERE id=? AND session_id=?`, id, session).Scan(&raw, &status); e != nil {
		return workflows.Run{}, e
	}
	if status == "running" || status == "completed" {
		return workflows.Run{}, fmt.Errorf("진행 중이거나 완료된 작업입니다")
	}
	var x workflows.Run
	if e = json.Unmarshal([]byte(raw), &x); e != nil {
		return x, e
	}
	x.Status = "running"
	if _, e = tx.Exec(`UPDATE workflow_runs SET status='running' WHERE id=?`, id); e != nil {
		return x, e
	}
	return x, tx.Commit()
}
