package orchestrator

import (
	"context"
	"database/sql"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	_ "modernc.org/sqlite"
)

// Opt-in operational check: use the managed cache/admission/load sequence,
// rather than treating an embedding HTTP health response as resident weights.
func TestLiveDocumentResidentEmbedding(t *testing.T) {
	bundle := os.Getenv("SPARKTALK_RESIDENT_BUNDLE")
	if bundle == "" {
		t.Skip("explicit document resident-bundle evaluation")
	}
	report := os.Getenv("SPARKTALK_RESIDENT_REPORT")
	model := os.Getenv("SPARKTALK_RESIDENT_MODEL")
	if report == "" || model == "" {
		t.Fatal("report path and expected model required")
	}
	db, err := sql.Open("sqlite", "file:/home/edp1096/workspace/dgx_spark_init/util/talk/dist/sparktalk.db?mode=ro")
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	var active int
	if err := db.QueryRow(`SELECT (SELECT count(*) FROM messages WHERE status NOT IN ('completed','cancelled','failed'))+(SELECT count(*) FROM workflow_runs WHERE status='running')`).Scan(&active); err != nil || active != 0 {
		t.Fatal("production must be idle", active, err)
	}
	client := http.Client{Timeout: 5 * time.Second}
	response, err := client.Get("http://127.0.0.1:8585/api/config")
	if err != nil {
		t.Fatal(err)
	}
	var cfg struct {
		Model struct {
			ID string `json:"default_model"`
		} `json:"model"`
	}
	err = json.NewDecoder(response.Body).Decode(&cfg)
	response.Body.Close()
	if err != nil || cfg.Model.ID != model {
		t.Fatal("active model changed", cfg.Model.ID, err)
	}
	c, err := NewController()
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	c.ConfigurePaths("/home/edp1096/.local/share/sparktalk", "/home/edp1096/.cache/huggingface")
	x, ok := c.Catalog().ResolveComponent(bundle, "extra-embedding")
	if !ok || !x.KeepResident {
		t.Fatal("no resident embedding in bundle", bundle)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	started := time.Now()
	err = c.prepareResidentEmbedding(ctx, x)
	data := map[string]any{"bundle": bundle, "model": model, "seconds": time.Since(started).Seconds(), "passed": err == nil}
	if err != nil {
		data["error"] = err.Error()
	}
	raw, _ := json.MarshalIndent(data, "", "  ")
	if e := os.WriteFile(filepath.Join(report, "embedding-managed.json"), raw, 0600); e != nil {
		t.Fatal(e)
	}
	if err != nil {
		t.Fatal(err)
	}
}
