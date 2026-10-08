package db

import (
	"encoding/json"
	"sparktalk/internal/skills"
)

func (d *DB) Skills() ([]skills.Skill, error) {
	rows, err := d.conn.Query(`SELECT name,description,instructions,toolsets,enabled FROM skills ORDER BY name`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	items := []skills.Skill{}
	for rows.Next() {
		var item skills.Skill
		var toolsets string
		if err := rows.Scan(&item.Name, &item.Description, &item.Instructions, &toolsets, &item.Enabled); err != nil {
			return nil, err
		}
		if err := json.Unmarshal([]byte(toolsets), &item.Toolsets); err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	return items, rows.Err()
}
func (d *DB) SaveSkill(item skills.Skill) error {
	toolsets, err := json.Marshal(item.Toolsets)
	if err != nil {
		return err
	}
	_, err = d.conn.Exec(`INSERT INTO skills(name,description,instructions,toolsets,enabled) VALUES(?,?,?,?,?)
 ON CONFLICT(name) DO UPDATE SET description=excluded.description,instructions=excluded.instructions,toolsets=excluded.toolsets,enabled=excluded.enabled,updated_at=CURRENT_TIMESTAMP`, item.Name, item.Description, item.Instructions, string(toolsets), item.Enabled)
	return err
}
func (d *DB) DeleteSkill(name string) error {
	_, err := d.conn.Exec(`DELETE FROM skills WHERE name=?`, name)
	return err
}
func (d *DB) BuiltinSkillSettings() (map[string]bool, error) {
	rows, err := d.conn.Query(`SELECT name,enabled FROM builtin_skill_settings`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	settings := map[string]bool{}
	for rows.Next() {
		var name string
		var enabled bool
		if err := rows.Scan(&name, &enabled); err != nil {
			return nil, err
		}
		settings[name] = enabled
	}
	return settings, rows.Err()
}
func (d *DB) SetBuiltinSkillEnabled(name string, enabled bool) error {
	_, err := d.conn.Exec(`INSERT INTO builtin_skill_settings(name,enabled) VALUES(?,?) ON CONFLICT(name) DO UPDATE SET enabled=excluded.enabled`, name, enabled)
	return err
}
