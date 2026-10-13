package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"

	"sparktalk/internal/llm"
	"sparktalk/internal/workflows"
)

func TestDocumentBatchPlannerRespectsRetrySizeAndContextLimits(t *testing.T) {
	run := workflows.Run{Document: &workflows.DocumentTask{Kind: "translate", Model: "qwen38fn_exl3", ModelType: "qwen38fn_exl3", Effort: "xhigh"}}
	units := make([]workflows.DocumentUnit, 5)
	for i := range units {
		units[i] = workflows.DocumentUnit{Ordinal: i, Stage: "process", Status: "pending", Inputs: []workflows.DocumentInput{{ID: string(rune('a' + i)), Text: strings.Repeat("x", 2000)}}}
	}
	batch, indices := documentRequestBatch(run, units, 0, 1048576)
	if len(indices) != 4 || len(batch.Inputs) != 4 {
		t.Fatal("bounded adjacent batch not planned", indices)
	}
	_, small := documentRequestBatch(run, units, 0, 12000)
	if len(small) >= len(indices) {
		t.Fatal("context allowance ignored")
	}
	units[0].Attempts = []workflows.DocumentAttempt{{Error: "failed batch"}}
	_, retry := documentRequestBatch(run, units, 0, 1048576)
	if len(retry) != 1 {
		t.Fatal("batch failure did not fall back to one checkpoint")
	}
	units[0].RetryBase = 1
	units[1].Status = "completed"
	_, completed := documentRequestBatch(run, units, 0, 1048576)
	if len(completed) != 1 {
		t.Fatal("crossed a completed checkpoint")
	}
}

func TestDocumentBatchesCommitSeparateCheckpointsAndFallbackAfterFailure(t *testing.T) {
	for _, failBatch := range []bool{false, true} {
		t.Run(map[bool]string{false: "success", true: "failed_batch_falls_back"}[failBatch], func(t *testing.T) {
			var calls atomic.Int32
			s, backend := documentFixture(t, func(w http.ResponseWriter, r *http.Request) {
				n := calls.Add(1)
				var body map[string]any
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
					t.Error(err)
					return
				}
				if body["reasoning_budget_tokens"] != float64(8192) {
					t.Error("xhigh native reasoning allowance missing", body["reasoning_budget_tokens"])
				}
				raw, _ := json.Marshal(body)
				r.Body = io.NopCloser(bytes.NewReader(raw))
				if failBatch && n == 1 {
					w.Header().Set("Content-Type", "text/event-stream")
					w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"{\\\"entries\\\":{}}\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
					return
				}
				mockDocumentResponse(w, r, true)
			})
			s.cfg.Model.DefaultModel = "qwen38fn_exl3"
			s.cfg.Model.ModelType = "qwen38fn_exl3"
			s.cfg.Model.ReasoningEffort = "xhigh"
			s.cfg.Context.WindowTokens = 1048576
			s.llm = llm.New(backend.URL, "qwen38fn_exl3", "", "qwen38fn_exl3")
			source := fixtureSource(t, s, strings.Repeat("All the source text must survive batched translation and saved checkpoints. ", 400)+"LAST_MARKER")
			run, err := s.documentAction(context.Background(), "s", documentStart{Action: "start", Kind: "translate", Sources: []string{source}})
			if err != nil {
				t.Fatal(err)
			}
			run = waitDocument(t, s, run.ID)
			if run.Status != "completed" || run.Document.Completed != run.Document.Total || int(calls.Load()) >= run.Document.Total {
				t.Fatalf("batching did not finish or reduce calls: %+v calls=%d", run.Document, calls.Load())
			}
			units, err := s.db.DocumentUnits("s", run.ID)
			if err != nil {
				t.Fatal(err)
			}
			for _, unit := range units {
				for _, attempt := range unit.Attempts {
					if (attempt.SharedWithOrdinal == nil) != (attempt.Performance != nil) {
						t.Fatal("each model call must own one performance record", attempt)
					}
				}
				for _, in := range unit.Inputs {
					if unit.Output[in.ID] == "" {
						t.Fatal("checkpoint lost source", in.ID)
					}
				}
			}
			if failBatch {
				if len(units[0].Attempts) != 2 || len(units[0].Attempts[0].BatchUnits) < 2 || len(units[0].Attempts[1].BatchUnits) > 1 {
					t.Fatal("failed batch did not fall back to the original checkpoint", units[0].Attempts)
				}
			} else if units[1].Attempts[0].SharedWithOrdinal == nil || *units[1].Attempts[0].SharedWithOrdinal != 0 || units[1].Attempts[0].OutputTokens != 0 {
				t.Fatal("shared call usage duplicated", units[1].Attempts)
			}
			data, _, err := s.db.DocumentOutput("s", run.ID, "output.txt")
			if err != nil || !strings.Contains(string(data), "LAST_MARKER") {
				t.Fatal("batch export lost source tail", err)
			}
		})
	}
}

func TestDocumentRequestReservesReasoningAndAnswerSeparately(t *testing.T) {
	var calls atomic.Int32
	_, backend := documentFixture(t, func(w http.ResponseWriter, r *http.Request) { calls.Add(1); mockDocumentResponse(w, r, true) })
	run := workflows.Run{Document: &workflows.DocumentTask{Kind: "translate", Model: "qwen38fn_exl3", ModelType: "qwen38fn_exl3", Effort: "xhigh", OutputLimit: 131072}}
	unit := workflows.DocumentUnit{Inputs: []workflows.DocumentInput{{ID: "s:b:0", Source: "s", Text: strings.Repeat("source ", 200)}}}
	client := llm.New(backend.URL, "qwen38fn_exl3", "", "qwen38fn_exl3")
	if _, err := requestDocumentUnit(context.Background(), client, run, unit, 8192); !errors.Is(err, llm.ErrOutputLimit) || calls.Load() != 0 {
		t.Fatal("small context consumed answer allowance", err, calls.Load())
	}
	if documentOutputBudget(run, unit) < documentReasoningBudget(run)+documentAnswerBudget(run, unit) {
		t.Fatal("reasoning and answer share an insufficient allowance")
	}
}

func TestAppendedSplitKeepsSourceOrderAndStopsBatchAtTheGap(t *testing.T) {
	run := workflows.Run{Document: &workflows.DocumentTask{Kind: "translate", Model: "qwen38fn_exl3", ModelType: "qwen38fn_exl3", Effort: "xhigh"}}
	units := []workflows.DocumentUnit{
		{Ordinal: 0, Stage: "process", Status: "pending", Inputs: []workflows.DocumentInput{{Source: "s", Block: "b000001", Part: 0, Text: "first half"}}},
		{Ordinal: 1, Stage: "process", Status: "pending", Inputs: []workflows.DocumentInput{{Source: "s", Block: "b000002", Part: 0, Text: "next paragraph"}}},
		{Ordinal: 2, Stage: "process", Status: "pending", Inputs: []workflows.DocumentInput{{Source: "s", Block: "b000001", Part: 10, Text: "second half"}}},
	}
	if !documentUnitBefore(units[2], units[1]) || documentUnitBefore(units[2], units[0]) {
		t.Fatal("split source position ignored")
	}
	_, indices := documentRequestBatch(run, units, 0, 1048576)
	if len(indices) != 1 {
		t.Fatal("batched across an unprocessed split", indices)
	}
}
