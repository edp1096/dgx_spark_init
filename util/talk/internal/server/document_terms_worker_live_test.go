package server

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
)

func TestDocumentTermsWorkerLive(t *testing.T) {
	dir := os.Getenv("SPARKTALK_TERMS_WORKER_LIVE")
	if dir == "" {
		t.Skip("explicit isolated end-to-end term repair")
	}
	raw, e := os.ReadFile(filepath.Join(dir, "before.db"))
	if e != nil {
		t.Fatal(e)
	}
	dest := filepath.Join(dir, "isolated.db")
	if e = os.WriteFile(dest, raw, 0600); e != nil {
		t.Fatal(e)
	}
	database, e := db.Open(dest)
	if e != nil {
		t.Fatal(e)
	}
	defer database.Close()
	prod, e := sql.Open("sqlite", "file:/home/edp1096/workspace/dgx_spark_init/util/talk/dist/sparktalk.db?mode=ro")
	if e != nil {
		t.Fatal(e)
	}
	defer prod.Close()
	idle := func() bool {
		var n int
		return prod.QueryRow(`SELECT (SELECT count(*) FROM messages WHERE status NOT IN ('completed','cancelled','failed'))+(SELECT count(*) FROM workflow_runs WHERE status='running')`).Scan(&n) == nil && n == 0
	}
	if !idle() {
		t.Fatal("production work active")
	}
	resp, e := http.Get("http://127.0.0.1:8585/api/config")
	if e != nil {
		t.Fatal(e)
	}
	var cfg config.Config
	e = json.NewDecoder(resp.Body).Decode(&cfg)
	resp.Body.Close()
	if e != nil {
		t.Fatal(e)
	}
	s := &Server{db: database, cfg: cfg, llm: llm.New(cfg.Model.Endpoint, cfg.Model.DefaultModel, cfg.Model.APIKey, cfg.Model.ModelType)}
	defer func() {
		s.tasks.Stop()
		ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
		defer cancel()
		_ = s.tasks.Wait(ctx)
	}()
	var fixture struct {
		Session   string      `json:"session"`
		Run       string      `json:"run"`
		Forbidden []string    `json:"forbidden"`
		Preserve  [][2]string `json:"preserve"`
	}
	fixture.Session = "chat-1791909912810453934"
	fixture.Run = "chat-1791909951704209505"
	if data, err := os.ReadFile(filepath.Join(dir, "fixture.json")); err == nil {
		if err = json.Unmarshal(data, &fixture); err != nil {
			t.Fatal(err)
		}
	}
	parent, e := database.DocumentRun(fixture.Session, fixture.Run)
	if e != nil {
		t.Fatal(e)
	}
	original, _, e := database.DocumentOutput(parent.SessionID, parent.ID, "output.txt")
	if e != nil {
		t.Fatal(e)
	}
	if only := os.Getenv("SPARKTALK_TERMS_ONLY"); only != "" {
		parent.Document.Glossary = map[string]string{only: parent.Document.Glossary[only]}
	}
	child, e := s.createDocumentTermsRevision(parent, true)
	if e != nil {
		t.Fatal(e)
	}
	child, e = s.documentAction(context.Background(), parent.SessionID, documentStart{Action: "resume", RunID: child.ID})
	if e != nil {
		t.Fatal(e)
	}
	began := time.Now()
	deadline := began.Add(45 * time.Minute)
	memory := []map[string]any{}
	lastLog := time.Now()
	for time.Now().Before(deadline) {
		if !idle() {
			_, _ = s.documentAction(context.Background(), parent.SessionID, documentStart{Action: "pause", RunID: child.ID})
			t.Fatal("user work became active")
		}
		child, e = database.DocumentRun(parent.SessionID, child.ID)
		if e != nil {
			t.Fatal(e)
		}
		raw, _ := os.ReadFile("/proc/meminfo")
		memory = append(memory, map[string]any{"seconds": time.Since(began).Seconds(), "meminfo": string(raw)})
		if time.Since(lastLog) > 10*time.Second {
			report, _ := json.MarshalIndent(child, "", "  ")
			os.WriteFile(filepath.Join(dir, "progress.json"), report, 0600)
			t.Log(child.Document.Phase, child.Document.Calls, "calls", child.Document.Terms.Checked, "terms")
			lastLog = time.Now()
		}
		if child.Status != "running" {
			break
		}
		time.Sleep(time.Second)
	}
	raw, _ = json.MarshalIndent(child, "", "  ")
	os.WriteFile(filepath.Join(dir, "run.json"), raw, 0600)
	finalUnits, e := database.DocumentUnits(child.SessionID, child.ID)
	if e != nil {
		t.Fatal(e)
	}
	raw, _ = json.MarshalIndent(finalUnits, "", "  ")
	os.WriteFile(filepath.Join(dir, "units.json"), raw, 0600)
	raw, _ = json.MarshalIndent(memory, "", "  ")
	os.WriteFile(filepath.Join(dir, "memory.json"), raw, 0600)
	if !child.DocumentFinished() {
		t.Fatalf("not finished: %s %s", child.Status, child.Document.Error)
	}
	for _, format := range []string{"txt", "md", "html", "zip"} {
		out, _, err := database.DocumentOutput(child.SessionID, child.ID, "output."+format)
		if err != nil {
			t.Fatal(err)
		}
		os.WriteFile(filepath.Join(dir, "output."+format), out, 0600)
	}
	saved, _, _ := database.DocumentOutput(parent.SessionID, parent.ID, "output.txt")
	if !bytes.Equal(saved, original) {
		t.Fatal("parent changed")
	}
	output, _, _ := database.DocumentOutput(child.SessionID, child.ID, "output.txt")
	if only := os.Getenv("SPARKTALK_TERMS_ONLY"); only == "Dinah" && strings.Contains(string(output), "디나단") {
		t.Fatal("known incorrect name remains")
	}
	for _, word := range fixture.Forbidden {
		if strings.Contains(string(output), word) {
			t.Errorf("known fault remains: %s", word)
		}
	}
	for _, pair := range fixture.Preserve {
		if !strings.Contains(string(output), pair[1]) {
			t.Errorf("normal control altered: %s", pair[1])
		}
	}
	t.Log(fmt.Sprintf("finished in %.1fs: %d calls, %d changed, %d unresolved", time.Since(began).Seconds(), child.Document.Calls, child.Document.Terms.Changed, child.Document.Terms.Unresolved))
}
