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
	"sparktalk/internal/workflows"
)

// Opt-in measurements against an already running backend. Fixtures and outputs
// are private files; the production DB is opened read-only, never via db.Open.
func TestDocumentEffortLive(t *testing.T) {
	dir := os.Getenv("SPARKTALK_DOCUMENT_EFFORT_LIVE")
	if dir == "" {
		t.Skip("opt-in document effort comparison")
	}
	var fixture struct {
		Database  string
		ConfigURL string
		Window    int
		Orders    [][]string
		Cases     []struct {
			Name string
			Run  workflows.Run
			Unit workflows.DocumentUnit
		}
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
	clientHTTP := &http.Client{Timeout: 5 * time.Second}
	orders := [][]string{{"off", "low", "medium"}, {"low", "medium", "off"}, {"medium", "off", "low"}}
	if len(fixture.Orders) > 0 {
		orders = fixture.Orders
	}
	for index, tc := range fixture.Cases {
		for _, effort := range orders[index%len(orders)] {
			t.Run(tc.Name+"/"+effort, func(t *testing.T) {
				resultPath := filepath.Join(dir, tc.Name+"-"+effort+".json")
				if _, err := os.Stat(resultPath); err == nil {
					t.Log("saved result already exists")
					return
				}
				var active int
				if err := database.QueryRow(`SELECT (SELECT count(*) FROM messages WHERE status='pending')+(SELECT count(*) FROM workflow_runs WHERE json_extract(data,'$.status')='running')`).Scan(&active); err != nil {
					t.Fatal(err)
				}
				if active != 0 {
					t.Fatal("user work active; benchmark stopped before submitting another request")
				}
				resp, err := clientHTTP.Get(fixture.ConfigURL)
				if err != nil {
					t.Fatal(err)
				}
				var cfg config.Config
				err = json.NewDecoder(resp.Body).Decode(&cfg)
				resp.Body.Close()
				if err != nil {
					t.Fatal(err)
				}
				if cfg.Model.DefaultModel != tc.Run.Document.Model || cfg.Model.Endpoint != tc.Run.Document.Endpoint || cfg.Model.ModelType != tc.Run.Document.ModelType {
					t.Fatal("active model differs from saved model")
				}
				encoded, _ := json.Marshal(tc.Run)
				var run workflows.Run
				if err = json.Unmarshal(encoded, &run); err != nil {
					t.Fatal(err)
				}
				run.Document.Effort = effort
				// The persisted last-used budget overrides effort in the worker.
				// Clear it so this comparison actually measures each requested mode.
				run.Document.ReasoningBudgetTokens = 0
				u := tc.Unit
				u.Inputs = documentRequestInputs(u)
				u.Attempts = nil
				u.RetryBase = 0
				u.OutputBudget = 0
				u.Output = nil
				u.Error = ""
				u.Warnings = nil
				client := llm.New(cfg.Model.Endpoint, cfg.Model.DefaultModel, cfg.Model.APIKey, cfg.Model.ModelType)
				report := map[string]any{"case": tc.Name, "effort": effort, "reasoning_budget_tokens": documentReasoningBudget(run), "output_budget_tokens": documentOutputBudget(run, u), "input_count": len(u.Inputs), "started_at": time.Now().Format(time.RFC3339Nano), "sampling_temperature": 0.7, "protocol": documentOutputProtocol, "checkpoint_ordinal": u.Ordinal}
				attempts := []map[string]any{}
				total := time.Now()
				for retry := 0; retry < 2; retry++ {
					ctx, cancel := context.WithTimeout(context.Background(), 6*time.Minute)
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
						if time.Since(heartbeat) >= 20*time.Second {
							progress, _ := json.Marshal(map[string]any{"case": tc.Name, "effort": effort, "attempt": retry + 1, "seconds": time.Since(started).Seconds(), "reasoning_chars": reasoningChars, "answer_chars": answerChars, "performance": metrics.Summary(true)})
							if err := os.WriteFile(filepath.Join(dir, "progress.json"), progress, 0600); err != nil {
								return err
							}
							heartbeat = time.Now()
						}
						return nil
					})
					cancel()
					seconds := time.Since(started).Seconds()
					output, validationErr := validateDocumentOutput(u, "translate", result)
					u.Output = output
					short := shortDocumentOutputs(u)
					problem := validationErr
					if problem == nil && len(short) > 0 {
						problem = fmt.Errorf("short translation: %v", short)
					}
					if requestErr != nil {
						problem = requestErr
					}
					if effort == "off" && result.Reasoning != "" {
						problem = fmt.Errorf("off unexpectedly generated reasoning")
					}
					attempt := map[string]any{"seconds": seconds, "first_answer_seconds": firstAnswer, "reasoning_chars": utf8.RuneCountInString(result.Reasoning), "content_chars": utf8.RuneCountInString(result.Content), "finish_reason": result.FinishReason, "usage": result.Usage, "performance": metrics.Summary(false), "raw_content": result.Content, "output": output, "short_outputs": short}
					if problem != nil {
						attempt["error"] = problem.Error()
					}
					attempts = append(attempts, attempt)
					report["attempts"] = attempts
					report["total_seconds"] = time.Since(total).Seconds()
					report["valid"] = problem == nil
					report["finished_at"] = time.Now().Format(time.RFC3339Nano)
					data, _ := json.MarshalIndent(report, "", "  ")
					if err := os.WriteFile(resultPath, data, 0600); err != nil {
						t.Fatal(err)
					}
					t.Logf("%s %s attempt %d: %.2fs; reasoning=%d answer=%d; valid=%v; %v", tc.Name, effort, retry+1, seconds, utf8.RuneCountInString(result.Reasoning), utf8.RuneCountInString(result.Content), problem == nil, problem)
					if problem == nil {
						var text strings.Builder
						for _, in := range u.Inputs {
							text.WriteString(output[in.ID])
						}
						if err := os.WriteFile(filepath.Join(dir, tc.Name+"-"+effort+".txt"), []byte(text.String()), 0600); err != nil {
							t.Fatal(err)
						}
						break
					}
					u.Attempts = append(u.Attempts, workflows.DocumentAttempt{Protocol: documentOutputProtocol, Error: problem.Error()})
					if requestErr != nil {
						break
					}
				}
			})
		}
	}
}
