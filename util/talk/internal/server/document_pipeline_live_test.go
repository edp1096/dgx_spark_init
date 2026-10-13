package server

import (
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
	"unicode/utf8"

	"sparktalk/internal/config"
	"sparktalk/internal/llm"
	"sparktalk/internal/performance"
	"sparktalk/internal/sourcebook"
	"sparktalk/internal/workflows"
)

// This opt-in comparison never creates production chats or document jobs.
// Both legacy and improved binaries consume the same frozen source fixtures.
func TestDocumentPipelineLive(t *testing.T) {
	dir := os.Getenv("SPARKTALK_DOCUMENT_PIPELINE_LIVE")
	if dir == "" {
		t.Skip("opt-in translation pipeline comparison")
	}
	var fixture struct {
		NativeBudget *int   `json:"native_budget,omitempty"`
		Database     string `json:"database"`
		ConfigURL    string `json:"config_url"`
		Endpoint     string `json:"endpoint,omitempty"`
		Window       int    `json:"window"`
		Effort       string `json:"effort"`
		Cases        []struct {
			Effort      string                      `json:"effort"`
			Temperature *float64                    `json:"temperature"`
			Name        string                      `json:"name"`
			Run         workflows.Run               `json:"run"`
			Units       []workflows.DocumentUnit    `json:"units"`
			Books       map[string]*sourcebook.Book `json:"books"`
			RawSources  map[string][]byte           `json:"raw_sources"`
		} `json:"cases"`
	}
	raw, err := os.ReadFile(filepath.Join(dir, "fixture.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	database, err := sql.Open("sqlite", "file:"+filepath.ToSlash(fixture.Database)+"?mode=ro")
	if err != nil {
		t.Fatal(err)
	}
	defer database.Close()
	idle := func() error {
		var active int
		err := database.QueryRow(`SELECT (SELECT count(*) FROM messages WHERE status NOT IN ('completed','cancelled','failed'))+(SELECT count(*) FROM workflow_runs WHERE status='running')`).Scan(&active)
		if err != nil {
			return err
		}
		if active != 0 {
			return fmt.Errorf("user work is active; benchmark request cancelled")
		}
		return nil
	}
	httpClient := &http.Client{Timeout: 5 * time.Second}
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			path := filepath.Join(dir, tc.Name+".json")
			if _, err := os.Stat(path); err == nil {
				t.Log("saved case exists")
				return
			}
			if err := idle(); err != nil {
				t.Fatal(err)
			}
			response, err := httpClient.Get(fixture.ConfigURL)
			if err != nil {
				t.Fatal(err)
			}
			var cfg config.Config
			err = json.NewDecoder(response.Body).Decode(&cfg)
			response.Body.Close()
			if err != nil {
				t.Fatal(err)
			}
			if expected := os.Getenv("SPARKTALK_DOCUMENT_PIPELINE_MODEL"); expected != "" && cfg.Model.DefaultModel != expected {
				t.Fatal("active model changed", cfg.Model.DefaultModel)
			}
			if fixture.Endpoint != "" {
				cfg.Model.Endpoint = fixture.Endpoint
			}
			run := tc.Run
			run.Document.Model = cfg.Model.DefaultModel
			run.Document.ModelType = cfg.Model.ModelType
			run.Document.Endpoint = cfg.Model.Endpoint
			run.Document.Effort = fixture.Effort
			if tc.Effort != "" {
				run.Document.Effort = tc.Effort
			}
			temperature := 0.7
			if tc.Temperature != nil {
				temperature = *tc.Temperature
			}
			run.Document.ReasoningBudgetTokens = 0
			if fixture.NativeBudget != nil {
				run.Document.ReasoningBudgetTokens = *fixture.NativeBudget
			}
			units := tc.Units
			for source, book := range tc.Books {
				book.RestoreInlineStructure(tc.RawSources[source])
				decorateDocumentInputs(units, source, book)
			}
			caseCtx, caseCancel := context.WithCancel(context.Background())
			watchDone := make(chan struct{})
			go func() {
				defer close(watchDone)
				ticker := time.NewTicker(time.Second)
				defer ticker.Stop()
				for {
					select {
					case <-caseCtx.Done():
						return
					case <-ticker.C:
						if idle() != nil {
							caseCancel()
							return
						}
					}
				}
			}()
			defer func() { caseCancel(); <-watchDone }()
			client := llm.New(cfg.Model.Endpoint, cfg.Model.DefaultModel, cfg.Model.APIKey, cfg.Model.ModelType).WithBackendCancellation(caseCtx)
			report := map[string]any{"case": tc.Name, "model": cfg.Model.DefaultModel, "model_type": cfg.Model.ModelType, "effort": run.Document.Effort, "protocol": documentOutputProtocol, "native_reasoning_budget": documentReasoningBudget(run), "started_at": time.Now().Format(time.RFC3339Nano), "temperature": temperature, "sampling": documentSampling(run)}
			calls := []map[string]any{}
			output := map[string]string{}
			startedCase := time.Now()
			write := func() {
				report["calls"] = calls
				report["output"] = output
				report["seconds"] = time.Since(startedCase).Seconds()
				data, _ := json.MarshalIndent(report, "", "  ")
				if err := os.WriteFile(path, data, 0600); err != nil {
					t.Fatal(err)
				}
			}
			for index := 0; index < len(units); {
				if units[index].Status == "completed" {
					index++
					continue
				}
				if err := idle(); err != nil {
					t.Fatal(err)
				}
				u, indices := documentRequestBatch(run, units, index, fixture.Window)
				ctx, cancel := context.WithTimeout(caseCtx, 12*time.Minute)
				ctx = llm.WithTemperature(ctx, temperature)
				var metrics performance.Accumulator
				ctx = llm.WithPerformanceObserver(ctx, metrics.Start)
				started := time.Now()
				heartbeat := started
				firstAnswer := 0.0
				reasoningChars, answerChars := 0, 0
				result, requestErr := requestDocumentUnit(ctx, client, run, u, fixture.Window, func(kind, text string) error {
					if kind == "reasoning" {
						reasoningChars += utf8.RuneCountInString(text)
					}
					if kind == "delta" {
						answerChars += utf8.RuneCountInString(text)
						if firstAnswer == 0 {
							firstAnswer = time.Since(started).Seconds()
						}
					}
					if time.Since(heartbeat) > 2*time.Second {
						if err := idle(); err != nil {
							return err
						}
						progress, _ := json.Marshal(map[string]any{"case": tc.Name, "batch": indices, "model": cfg.Model.DefaultModel, "seconds": time.Since(started).Seconds(), "reasoning_chars": reasoningChars, "answer_chars": answerChars, "performance": metrics.Summary(true)})
						if err := os.WriteFile(filepath.Join(dir, "progress.json"), progress, 0600); err != nil {
							return err
						}
						heartbeat = time.Now()
					}
					return nil
				})
				cancel()
				out, validationErr := validateDocumentOutput(u, "translate", result)
				u.Output = out
				problem := requestErr
				if problem == nil {
					problem = validationErr
				}
				if problem == nil && len(shortDocumentOutputs(u)) > 0 {
					problem = fmt.Errorf("unexpectedly short paragraph output")
				}
				call := map[string]any{"indices": indices, "seconds": time.Since(started).Seconds(), "first_answer_seconds": firstAnswer, "reasoning_chars": utf8.RuneCountInString(result.Reasoning), "answer_chars": utf8.RuneCountInString(result.Content), "usage": result.Usage, "performance": metrics.Summary(false), "finish_reason": result.FinishReason, "raw_content": result.Content, "valid": problem == nil}
				if problem != nil {
					call["error"] = problem.Error()
				}
				calls = append(calls, call)
				if problem != nil {
					report["valid"] = false
					write()
					t.Fatal(problem)
				}
				for id, text := range out {
					output[id] = text
				}
				for _, i := range indices {
					units[i].Status = "completed"
				}
				run.Document.ContextTail = documentContextTail(u)
				if run.Document.Pipeline == "" {
					updateDocumentGlossary(run.Document, result)
				}
				write()
				t.Logf("%s model=%s batch=%v seconds=%.2f", tc.Name, cfg.Model.DefaultModel, indices, time.Since(started).Seconds())
			}
			report["valid"] = true
			report["finished_at"] = time.Now().Format(time.RFC3339Nano)
			write()
			var prose strings.Builder
			for _, unit := range units {
				for _, group := range documentTextGroups(unit) {
					for _, in := range group.Inputs {
						prose.WriteString(output[in.ID])
					}
					prose.WriteString("\n\n")
				}
			}
			if err := os.WriteFile(filepath.Join(dir, tc.Name+".txt"), []byte(prose.String()), 0600); err != nil {
				t.Fatal(err)
			}
		})
	}
}
