package server

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"
	"unicode/utf8"

	"sparktalk/internal/llm"
	"sparktalk/internal/workflows"
)

// Opt-in replay of one saved failed unit, without opening or writing the user's DB.
func TestDocumentRecoveryLive(t *testing.T) {
	dir := os.Getenv("SPARKTALK_DOCUMENT_RECOVERY_LIVE")
	if dir == "" {
		t.Skip("opt-in single-unit recovery verification")
	}
	var fixture struct {
		Run    workflows.Run
		Unit   workflows.DocumentUnit
		Window int
	}
	raw, err := os.ReadFile(filepath.Join(dir, "fixture.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if budget, err := strconv.Atoi(os.Getenv("SPARKTALK_DOCUMENT_RECOVERY_BUDGET")); err == nil && budget > 0 {
		fixture.Run.Document.ReasoningBudgetTokens = budget
	}
	client := llm.New(fixture.Run.Document.Endpoint, fixture.Run.Document.Model, "", fixture.Run.Document.ModelType)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	start := time.Now()
	result, err := requestDocumentUnit(ctx, client, fixture.Run, fixture.Unit, fixture.Window)
	output, validationErr := validateDocumentOutput(fixture.Unit, fixture.Run.Document.Kind, result)
	report := map[string]any{"seconds": time.Since(start).Seconds(), "reasoning_budget_tokens": documentReasoningBudget(fixture.Run), "reasoning_chars": utf8.RuneCountInString(result.Reasoning), "content_chars": utf8.RuneCountInString(result.Content), "finish_reason": result.FinishReason, "usage": result.Usage, "input_count": len(fixture.Unit.Inputs), "output_count": len(output), "output": output}
	if err != nil {
		report["error"] = err.Error()
	}
	if validationErr != nil {
		report["validation_error"] = validationErr.Error()
	}
	encoded, _ := json.MarshalIndent(report, "", "  ")
	os.WriteFile(filepath.Join(dir, "result.json"), encoded, 0600)
	if err != nil {
		t.Fatal(err)
	}
	if validationErr != nil {
		t.Fatal(validationErr)
	}
	if len(shortDocumentOutputs(workflows.DocumentUnit{Inputs: fixture.Unit.Inputs, Output: output})) > 0 {
		t.Fatal("short translation")
	}
	t.Logf("failed unit recovered in %.2fs; reasoning %d chars, answer %d chars; all %d IDs validated", time.Since(start).Seconds(), utf8.RuneCountInString(result.Reasoning), utf8.RuneCountInString(result.Content), len(output))
}
