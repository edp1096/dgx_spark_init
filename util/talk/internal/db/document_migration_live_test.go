package db

import (
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Run only against an explicitly prepared, disposable snapshot. Open performs
// recovery writes, so this test must never take the live production database.
func TestDocumentMigrationPreservesProductionSnapshot(t *testing.T) {
	path := os.Getenv("SPARKTALK_DOCUMENT_MIGRATION_COPY")
	if path == "" {
		t.Skip("opt-in copied database migration")
	}
	abs, err := filepath.Abs(path)
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(abs) != "production-migration-check.db" || !strings.Contains(filepath.Dir(abs), "document-correction-") {
		t.Fatal("refusing a non-evaluation database")
	}
	queries := map[string]string{
		"outputs":  "SELECT run_id,name,mime,data FROM workflow_outputs ORDER BY run_id,name",
		"units":    "SELECT run_id,ordinal,data FROM workflow_units ORDER BY run_id,ordinal",
		"runs":     "SELECT id,session_id,status,data,updated FROM workflow_runs ORDER BY id",
		"sources":  "SELECT id,data,raw FROM source_snapshots ORDER BY id",
		"messages": "SELECT id,session_id,role,status,content FROM messages ORDER BY id",
	}
	type receipt struct {
		Rows int    `json:"rows"`
		Hash string `json:"sha256"`
	}
	snapshot := func() map[string]receipt {
		c, e := sql.Open("sqlite", "file:"+filepath.ToSlash(abs)+"?mode=ro")
		if e != nil {
			t.Fatal(e)
		}
		defer c.Close()
		out := map[string]receipt{}
		for key, q := range queries {
			rows, e := c.Query(q)
			if e != nil {
				t.Fatal(e)
			}
			columns, _ := rows.Columns()
			h := sha256.New()
			n := 0
			for rows.Next() {
				values := make([]any, len(columns))
				ptrs := make([]any, len(columns))
				for i := range values {
					ptrs[i] = &values[i]
				}
				if e = rows.Scan(ptrs...); e != nil {
					t.Fatal(e)
				}
				b, e := json.Marshal(values)
				if e != nil {
					t.Fatal(e)
				}
				h.Write(b)
				h.Write([]byte{'\n'})
				n++
			}
			if e = rows.Err(); e != nil {
				t.Fatal(e)
			}
			rows.Close()
			out[key] = receipt{n, hex.EncodeToString(h.Sum(nil))}
		}
		return out
	}
	before := snapshot()
	database, err := Open(abs)
	if err != nil {
		t.Fatal(err)
	}
	for _, table := range []string{"document_attempts", "document_reviews", "document_entities"} {
		var n int
		if err = database.conn.QueryRow("SELECT count(*) FROM " + table).Scan(&n); err != nil {
			t.Fatal(err)
		}
	}
	database.Close()
	after := snapshot()
	for key, a := range before {
		if a != after[key] {
			t.Errorf("migration changed %s: before=%+v after=%+v", key, a, after[key])
		}
	}
	raw, _ := json.MarshalIndent(map[string]any{"before": before, "after": after, "passed": !t.Failed()}, "", "  ")
	if err = os.WriteFile(filepath.Join(filepath.Dir(abs), "production-migration-check.json"), raw, 0600); err != nil {
		t.Fatal(err)
	}
}
