package server

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/sourcebook"
	"sparktalk/internal/workflows"
)

// Same frozen excerpt and medium effort, legacy translation versus automatic
// consistency processing. All generated records stay in a private database.
func TestDocumentTermsNewTranslationLive(t *testing.T) {
	dir := os.Getenv("SPARKTALK_TERMS_NEW_LIVE")
	if dir == "" {
		t.Skip("explicit isolated fresh translation comparison")
	}
	prod, err := sql.Open("sqlite", "file:/home/edp1096/workspace/dgx_spark_init/util/talk/dist/sparktalk.db?mode=ro")
	if err != nil {
		t.Fatal(err)
	}
	defer prod.Close()
	idle := func() bool {
		var n int
		return prod.QueryRow(`SELECT (SELECT count(*) FROM messages WHERE status NOT IN ('completed','cancelled','failed'))+(SELECT count(*) FROM workflow_runs WHERE status='running')`).Scan(&n) == nil && n == 0
	}
	if !idle() {
		t.Fatal("production work active")
	}
	response, err := http.Get("http://127.0.0.1:8585/api/config")
	if err != nil {
		t.Fatal(err)
	}
	var cfg config.Config
	err = json.NewDecoder(response.Body).Decode(&cfg)
	response.Body.Close()
	if err != nil {
		t.Fatal(err)
	}
	cfg.Model.ReasoningEffort = "medium"
	database, err := db.Open(filepath.Join(dir, "isolated.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer database.Close()
	s := &Server{db: database, cfg: cfg, llm: llm.New(cfg.Model.Endpoint, cfg.Model.DefaultModel, cfg.Model.APIKey, cfg.Model.ModelType)}
	defer func() {
		s.tasks.Stop()
		ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
		defer cancel()
		_ = s.tasks.Wait(ctx)
	}()
	if _, err = database.CreateSession("sample", "Fresh translation comparison", cfg.Model.DefaultModel, "medium"); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "source.html"))
	if err != nil {
		t.Fatal(err)
	}
	book, err := sourcebook.Parse(raw, "text/html", "https://www.gutenberg.org/cache/epub/2776/pg2776-images.html", nil)
	if err != nil {
		t.Fatal(err)
	}
	if err = database.SaveSource("sample-source", "sample", book, raw); err != nil {
		t.Fatal(err)
	}
	for _, mode := range []string{"legacy", workflows.DocumentTermsVersion} {
		if !idle() {
			t.Fatal("user work active")
		}
		pipeline := mode
		if mode == "legacy" {
			pipeline = ""
		}
		began := time.Now()
		run, err := s.documentAction(context.Background(), "sample", documentStart{Action: "start", Kind: "translate", Sources: []string{"sample-source"}, Effort: "medium", Instructions: "원문의 내용을 빠짐없이 자연스러운 한국어로 번역한다. 이름, 수치와 문단 구조를 보존한다.", Pipeline: pipeline})
		if err != nil {
			t.Fatal(err)
		}
		memory := []map[string]any{}
		for time.Since(began) < 15*time.Minute {
			if !idle() {
				_, _ = s.documentAction(context.Background(), "sample", documentStart{Action: "pause", RunID: run.ID})
				t.Fatal("user work became active")
			}
			run, err = database.DocumentRun("sample", run.ID)
			if err != nil {
				t.Fatal(err)
			}
			mem, _ := os.ReadFile("/proc/meminfo")
			memory = append(memory, map[string]any{"seconds": time.Since(began).Seconds(), "meminfo": string(mem)})
			if run.Status != "running" {
				break
			}
			time.Sleep(time.Second)
		}
		units, err := database.DocumentUnits("sample", run.ID)
		if err != nil {
			t.Fatal(err)
		}
		report := map[string]any{"run": run, "units": units, "memory": memory, "seconds": time.Since(began).Seconds(), "source_sha256": documentTextHash(string(raw))}
		data, _ := json.MarshalIndent(report, "", "  ")
		if err = os.WriteFile(filepath.Join(dir, mode+".json"), data, 0600); err != nil {
			t.Fatal(err)
		}
		if !run.DocumentFinished() {
			t.Fatal(mode, run.Status, run.Document.Error)
		}
		for _, format := range []string{"txt", "html"} {
			out, _, err := database.DocumentOutput("sample", run.ID, "output."+format)
			if err != nil {
				t.Fatal(err)
			}
			os.WriteFile(filepath.Join(dir, mode+"."+format), out, 0600)
		}
		t.Log(fmt.Sprintf("%s: %.1fs, %d calls, %d tokens", mode, time.Since(began).Seconds(), run.Document.Calls, run.Document.OutputTokens))
	}
}
