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

// Exercise the actual durable worker and exporters using an isolated DB. The
// production database is read only and only supplies the user-work watchdog.
func TestDocumentReviewedEndToEndLive(t *testing.T) {
	dir := os.Getenv("SPARKTALK_DOCUMENT_REVIEWED_LIVE")
	if dir == "" {
		t.Skip("opt-in full pipeline evaluation")
	}
	var fixture struct {
		Temperature       *float64 `json:"temperature,omitempty"`
		Database          string   `json:"database"`
		ConfigURL         string   `json:"config_url"`
		Model             string   `json:"model"`
		Effort            string   `json:"effort"`
		DocumentsEndpoint string   `json:"documents_endpoint"`
		Formats           []string `json:"formats"`
		Cases             []struct {
			Name  string `json:"name"`
			Path  string `json:"path"`
			Hash  string `json:"sha256"`
			URL   string `json:"url"`
			MIME  string `json:"mime"`
			Title string `json:"title"`
		} `json:"cases"`
	}
	raw, err := os.ReadFile(filepath.Join(dir, "fixture.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	production, err := sql.Open("sqlite", "file:"+filepath.ToSlash(fixture.Database)+"?mode=ro")
	if err != nil {
		t.Fatal(err)
	}
	defer production.Close()
	idle := func() error {
		var n int
		err := production.QueryRow(`SELECT (SELECT count(*) FROM messages WHERE status NOT IN ('completed','cancelled','failed'))+(SELECT count(*) FROM workflow_runs WHERE status='running')`).Scan(&n)
		if err != nil {
			return err
		}
		if n != 0 {
			return fmt.Errorf("user work became active")
		}
		return nil
	}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			if tc.Name == "" || filepath.Base(tc.Name) != tc.Name {
				t.Fatal("invalid case name")
			}
			reportPath := filepath.Join(dir, tc.Name+".json")
			if data, e := os.ReadFile(reportPath); e == nil {
				var prior map[string]any
				_ = json.Unmarshal(data, &prior)
				if prior["finished"] == true && prior["formats_complete"] == true {
					t.Log("saved complete case exists")
					return
				}
			}
			if err = idle(); err != nil {
				t.Fatal(err)
			}
			httpClient := http.Client{Timeout: 5 * time.Second}
			response, e := httpClient.Get(fixture.ConfigURL)
			if e != nil {
				t.Fatal(e)
			}
			var cfg config.Config
			e = json.NewDecoder(response.Body).Decode(&cfg)
			response.Body.Close()
			if e != nil {
				t.Fatal(e)
			}
			if fixture.Model != "" && cfg.Model.DefaultModel != fixture.Model {
				t.Fatal("active model changed")
			}
			cfg.Model.ReasoningEffort = fixture.Effort
			cfg.Extra.DocumentsEnabled = fixture.DocumentsEndpoint != ""
			cfg.Extra.DocumentsEndpoint = fixture.DocumentsEndpoint
			localPath := filepath.Join(dir, tc.Name+".db")
			if filepath.Clean(localPath) == filepath.Clean(fixture.Database) {
				t.Fatal("refusing production DB")
			}
			database, e := db.Open(localPath)
			if e != nil {
				t.Fatal(e)
			}
			defer database.Close()
			s := &Server{db: database, cfg: cfg, llm: llm.New(cfg.Model.Endpoint, cfg.Model.DefaultModel, cfg.Model.APIKey, cfg.Model.ModelType)}
			defer func() {
				s.tasks.Stop()
				ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
				defer cancel()
				if e := s.tasks.Wait(ctx); e != nil {
					t.Error(e)
				}
			}()
			if _, e = database.Session("evaluation"); e != nil {
				if _, e = database.CreateSession("evaluation", tc.Name, cfg.Model.DefaultModel, fixture.Effort); e != nil {
					t.Fatal(e)
				}
			}
			source, e := os.ReadFile(tc.Path)
			if e != nil {
				t.Fatal(e)
			}
			if documentTextHash(string(source)) != tc.Hash {
				t.Fatal("frozen source changed")
			}
			const sourceID = "frozen-source"
			if !database.SourceExists("evaluation", sourceID) {
				book, e := sourcebook.Parse(source, tc.MIME, tc.URL, nil)
				if e != nil {
					t.Fatal(e)
				}
				if book.Title == "" {
					book.Title = tc.Title
				}
				if e = database.SaveSource(sourceID, "evaluation", book, source); e != nil {
					t.Fatal(e)
				}
			}
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Hour)
			defer cancel()
			watchDone := make(chan struct{})
			go func() {
				defer close(watchDone)
				ticker := time.NewTicker(time.Second)
				defer ticker.Stop()
				for {
					select {
					case <-ctx.Done():
						return
					case <-ticker.C:
						if idle() != nil {
							cancel()
							return
						}
					}
				}
			}()
			defer func() { cancel(); <-watchDone }()
			var run workflows.Run
			if saved, e := os.ReadFile(filepath.Join(dir, tc.Name+".run-id")); e == nil {
				run, e = database.DocumentRun("evaluation", string(saved))
				if e != nil {
					t.Fatal(e)
				}
				if !run.DocumentFinished() {
					run, e = s.documentAction(ctx, "evaluation", documentStart{Action: "resume", RunID: run.ID})
					if e != nil {
						t.Fatal(e)
					}
				}
			} else {
				profiles := documentProfiles(cfg, fixture.Effort)
				if fixture.Temperature != nil {
					profiles.Translate.Temperature = *fixture.Temperature
				}
				run, e = s.startDocument(ctx, "evaluation", documentStart{Profiles: profiles, Kind: "translate", Sources: []string{sourceID}, Instructions: "본문·제목·각주·서지 정보를 내용 누락 없이 자연스러운 한국어로 번역한다. 삽화와 원문의 문단 구조를 보존한다.", Effort: fixture.Effort, Pipeline: workflows.DocumentPipelineVersion})
				if e != nil {
					t.Fatal(e)
				}
				if e = os.WriteFile(filepath.Join(dir, tc.Name+".run-id"), []byte(run.ID), 0600); e != nil {
					t.Fatal(e)
				}
			}
			started := time.Now()
			for run.Status == "running" {
				if ctx.Err() != nil {
					_, _ = s.documentAction(context.Background(), "evaluation", documentStart{Action: "pause", RunID: run.ID})
					t.Fatal("evaluation stopped", ctx.Err())
				}
				time.Sleep(time.Second)
				run, e = database.DocumentRun("evaluation", run.ID)
				if e != nil {
					t.Fatal(e)
				}
				progress, _ := json.Marshal(map[string]any{"case": tc.Name, "seconds": time.Since(started).Seconds(), "status": run.Status, "document": run.Document})
				_ = os.WriteFile(filepath.Join(dir, "progress.json"), progress, 0600)
			}
			formatErrors := []string{}
			for _, format := range fixture.Formats {
				if !run.DocumentFinished() {
					break
				}
				run, e = s.documentAction(ctx, "evaluation", documentStart{Action: "export", RunID: run.ID, Format: format})
				if e != nil {
					t.Error("format", format, e)
					formatErrors = append(formatErrors, format+": "+e.Error())
					continue
				}
			}
			units, e := database.DocumentUnits("evaluation", run.ID)
			if e != nil {
				t.Fatal(e)
			}
			reviews, e := database.DocumentReviews("evaluation", run.ID)
			if e != nil {
				t.Fatal(e)
			}
			entities, e := database.DocumentEntities("evaluation", run.ID)
			if e != nil {
				t.Fatal(e)
			}
			report := map[string]any{"run": run, "units": units, "reviews": reviews, "entities": entities, "finished": run.DocumentFinished(), "formats_complete": len(formatErrors) == 0, "format_errors": formatErrors, "elapsed_wall_seconds": time.Since(started).Seconds(), "fixture_sha256": documentTextHash(string(raw)), "source_sha256": tc.Hash}
			data, _ := json.MarshalIndent(report, "", "  ")
			if e = os.WriteFile(reportPath, data, 0600); e != nil {
				t.Fatal(e)
			}
			if run.Document.Publication != nil {
				for _, file := range run.Document.Publication.Files {
					data, _, e := database.DocumentOutput("evaluation", run.ID, file.Name)
					if e != nil {
						t.Fatal(e)
					}
					if documentTextHash(string(data)) != file.Hash {
						t.Fatal("published file hash changed")
					}
					if e = os.WriteFile(filepath.Join(dir, tc.Name+"-"+file.Name), data, 0600); e != nil {
						t.Fatal(e)
					}
				}
			}
			t.Logf("%s status=%s calls=%d elapsed=%ds reviewed=%d/%d unresolved=%d", tc.Name, run.Status, run.Document.Calls, run.Document.ElapsedSeconds, run.Document.Reviewed, run.Document.ReviewTotal, run.Document.Unresolved)
			if !run.DocumentFinished() {
				t.Fatal(run.Status, run.Document.Error)
			}
		})
		if t.Failed() {
			return
		}
	}
}
