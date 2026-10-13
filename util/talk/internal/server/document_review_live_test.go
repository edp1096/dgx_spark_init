package server

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"testing"
	"time"
	"unicode/utf8"

	"sparktalk/internal/config"
	"sparktalk/internal/llm"
	"sparktalk/internal/performance"
	"sparktalk/internal/workflows"
)

// Uses fixed source/target pairs; no production chat/job/DB writes. An
// independent watchdog cancels even during prefill if user work starts.
func TestDocumentReviewLive(t *testing.T) {
	dir := os.Getenv("SPARKTALK_DOCUMENT_REVIEW_LIVE")
	if dir == "" {
		t.Skip("opt-in reviewer calibration")
	}
	var fixture struct {
		NativeBudget   *int              `json:"native_budget,omitempty"`
		Pipeline       string            `json:"pipeline"`
		Database       string            `json:"database"`
		ConfigURL      string            `json:"config_url"`
		Efforts        []string          `json:"efforts"`
		SharedContexts []string          `json:"shared_contexts"`
		SourceTitles   map[string]string `json:"source_titles"`
		Batches        [][]struct {
			ID      string `json:"id"`
			Source  string `json:"source"`
			Target  string `json:"target"`
			Context string `json:"context,omitempty"`
		} `json:"batches"`
	}
	data, err := os.ReadFile(filepath.Join(dir, "fixture.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	database, err := sql.Open("sqlite", "file:"+filepath.ToSlash(fixture.Database)+"?mode=ro")
	if err != nil {
		t.Fatal(err)
	}
	defer database.Close()
	idle := func() error {
		var n int
		err := database.QueryRow(`SELECT (SELECT count(*) FROM messages WHERE status NOT IN ('completed','cancelled','failed'))+(SELECT count(*) FROM workflow_runs WHERE status='running')`).Scan(&n)
		if err != nil {
			return err
		}
		if n > 0 {
			return fmt.Errorf("user work is active")
		}
		return nil
	}
	httpClient := &http.Client{Timeout: 5 * time.Second}
	for _, effort := range fixture.Efforts {
		for index, batch := range fixture.Batches {
			name := fmt.Sprintf("%s-%03d", effort, index)
			t.Run(name, func(t *testing.T) {
				path := filepath.Join(dir, name+".json")
				if _, err := os.Stat(path); err == nil {
					t.Log("saved result exists")
					return
				}
				if err := idle(); err != nil {
					t.Fatal(err)
				}
				resp, err := httpClient.Get(fixture.ConfigURL)
				if err != nil {
					t.Fatal(err)
				}
				var cfg config.Config
				err = json.NewDecoder(resp.Body).Decode(&cfg)
				resp.Body.Close()
				if err != nil {
					t.Fatal(err)
				}
				if expected := os.Getenv("SPARKTALK_DOCUMENT_REVIEW_MODEL"); expected != "" && cfg.Model.DefaultModel != expected {
					t.Fatal("active model changed")
				}
				run := workflows.Run{Document: &workflows.DocumentTask{Pipeline: fixture.Pipeline, Kind: "translate", Instructions: "원문의 의미와 내용을 빠짐없이 한국어로 번역한다.", Model: cfg.Model.DefaultModel, ModelType: cfg.Model.ModelType, Endpoint: cfg.Model.Endpoint, Effort: effort}}
				run.Document.SourceTitles = fixture.SourceTitles
				if fixture.NativeBudget != nil {
					run.Document.ReasoningBudgetTokens = *fixture.NativeBudget
				}
				if index < len(fixture.SharedContexts) {
					run.Document.SourceContext = "Shared nearby source (reference only):\n" + fixture.SharedContexts[index] + "\n"
				}
				u := workflows.DocumentUnit{Output: map[string]string{}}
				ids := []string{}
				for i, pair := range batch {
					id := documentLocalID(i)
					ids = append(ids, pair.ID)
					u.Inputs = append(u.Inputs, workflows.DocumentInput{ID: id, Source: "fixture", Block: id, Group: id, Plain: true, Text: pair.Source})
					u.Output[id] = pair.Target
					if pair.Context != "" {
						run.Document.SourceContext += fmt.Sprintf("Context for %s: %s\n", id, pair.Context)
					}
				}
				ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
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
				client := llm.New(cfg.Model.Endpoint, cfg.Model.DefaultModel, cfg.Model.APIKey, cfg.Model.ModelType).WithBackendCancellation(ctx)
				var metrics performance.Accumulator
				ctx = llm.WithPerformanceObserver(ctx, metrics.Start)
				ctx = llm.WithTemperature(ctx, 0.2)
				started := time.Now()
				lastProgress := started
				var reasoningText, answerText string
				result, requestErr := requestDocumentReview(ctx, client, run, u, cfg.Context.WindowTokens, func(kind, text string) error {
					if kind == "reasoning" {
						reasoningText += text
					}
					if kind == "delta" {
						answerText += text
					}
					if time.Since(lastProgress) > 2*time.Second {
						progress, _ := json.Marshal(map[string]any{"case": name, "seconds": time.Since(started).Seconds(), "reasoning_chars": utf8.RuneCountInString(reasoningText), "content_chars": utf8.RuneCountInString(answerText), "performance": metrics.Summary(true)})
						if err := os.WriteFile(filepath.Join(dir, "progress.json"), progress, 0600); err != nil {
							return err
						}
						lastProgress = time.Now()
					}
					return nil
				})
				cancel()
				<-watchDone
				review, validationErr := validateDocumentReview(u, result)
				report := map[string]any{"protocol": review.Protocol, "model": cfg.Model.DefaultModel, "model_type": cfg.Model.ModelType, "effort": effort, "temperature": 0.2, "sampling": documentSampling(run), "native_reasoning_budget": documentReasoningBudget(run), "window": cfg.Context.WindowTokens, "ids": ids, "seconds": time.Since(started).Seconds(), "performance": metrics.Summary(false), "usage": result.Usage, "finish_reason": result.FinishReason, "raw_content": result.Content, "review": review, "valid": requestErr == nil && validationErr == nil}
				report["fixture_sha256"] = documentTextHash(string(data))
				report["reasoning_chars"] = utf8.RuneCountInString(result.Reasoning)
				report["raw_reasoning"] = result.Reasoning
				report["review_reasoning_reserve"] = documentReviewReasoningReserve(run)
				report["content_chars"] = utf8.RuneCountInString(result.Content)
				if requestErr != nil {
					report["request_error"] = requestErr.Error()
				}
				if validationErr != nil {
					report["validation_error"] = validationErr.Error()
				}
				out, _ := json.MarshalIndent(report, "", "  ")
				if err = os.WriteFile(path, out, 0600); err != nil {
					t.Fatal(err)
				}
				t.Logf("%s %.2fs valid=%v", name, time.Since(started).Seconds(), report["valid"])
				if requestErr != nil && !errors.Is(requestErr, llm.ErrOutputLimit) {
					t.Fatal(requestErr)
				}
			})
			if t.Failed() {
				return
			}
		}
	}
}
