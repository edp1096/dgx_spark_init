package server

import (
	"encoding/json"
	"sparktalk/internal/llm"
	"sparktalk/internal/workflows"
	"strings"
	"testing"
)

func TestDocumentSchemaQuotationExampleRoundTrips(t *testing.T) {
	prompt := documentSchemaInstructions([]byte("{}"))
	_, example, ok := strings.Cut(prompt, " is encoded as ")
	if !ok {
		t.Fatal("missing quotation example")
	}
	var decoded string
	if err := json.Unmarshal([]byte(strings.TrimSuffix(example, ".")), &decoded); err != nil || decoded != "He said \"hello\"." {
		t.Fatalf("JSON example changes the literal quotation: %q, %v", decoded, err)
	}
}

func TestDocumentReviewV2KeepsValidSiblingAndRejectsInventedNestedField(t *testing.T) {
	u := workflows.DocumentUnit{Inputs: []workflows.DocumentInput{{ID: "a", Group: "a", Text: "Two dogs."}, {ID: "b", Group: "b", Text: "Blue door."}}, Output: map[string]string{"a": "개 세 마리.", "b": "파란 문."}}
	r := llm.StreamResult{FinishReason: "stop", Content: `{"checks":{"p000001":[{"evidence":{"source_quote":"Two","target_quote":"세"},"finding":{"category":"number_unit","severity":"major","explanation":"둘을 셋으로 바꿨다.","uncertain":false}}],"p000002":[{"evidence":{"source_quote":"Blue","target_quote":"파란"},"finding":{"category":"lexical","severity":"major","explanation":"invented","uncertain":false,"override":"approve"}}]}}`}
	review, err := validateDocumentReviewPartial(u, r)
	if err != nil || review.Protocol != documentReviewProtocolV2 || len(review.Items) != 1 || len(review.Invalid) != 1 || review.Status != "pending" {
		t.Fatal(review, err)
	}
	if review.Items[0].Findings[0].SourceQuote != "Two" {
		t.Fatal("lost original evidence")
	}
}

func TestDocumentReviewUncappedBackendStillReservesThinkingOutput(t *testing.T) {
	for _, model := range []struct {
		id, kind string
		want     int
	}{{"qwen38fn_exl3", "qwen38fn_exl3", 4096}, {"qwen38fn_exl3_q4", "qwen38fn_exl3", 4096}, {"radixark", "qwen3.8", 8192}} {
		run := workflows.Run{Document: &workflows.DocumentTask{Model: model.id, ModelType: model.kind, Effort: "medium"}}
		if got := documentReviewReasoningReserve(run); got != model.want {
			t.Fatal(model, got)
		}
		run.Document.Effort = "low"
		wantLow := 2048
		if model.kind == "qwen3.8" {
			wantLow = 8192
		}
		if got := documentReviewReasoningReserve(run); got != wantLow {
			t.Fatal("soft low effort treated as a hard thinking cap", model, got)
		}
		run.Document.ReasoningBudgetTokens = -1
		if got := documentReviewReasoningReserve(run); got != 8192 {
			t.Fatal("explicit uncapped comparison lost its reserve", model, got)
		}
		run.Document.Effort = "none"
		if documentReviewReasoningReserve(run) != 0 {
			t.Fatal("disabled thinking reserved tokens")
		}
	}
}

func TestDocumentUncappedLowTranslationKeepsAnswerReservation(t *testing.T) {
	u := workflows.DocumentUnit{Inputs: []workflows.DocumentInput{{ID: "s:a:0", Source: "s", Block: "a", Group: "g", Plain: true, Text: "A complete source paragraph."}}}
	run := workflows.Run{Document: &workflows.DocumentTask{Kind: "translate", Model: "radixark", ModelType: "qwen3.8", Effort: "low", Pipeline: workflows.DocumentPipelineVersion}}
	answer := documentAnswerBudget(run, u)
	if got := documentOutputBudget(run, u); got != answer+8192 {
		t.Fatal("soft low reasoning consumed the translation allowance", got, answer)
	}
	run.Document.Pipeline = ""
	if got := documentOutputBudget(run, u); got != answer+16384 {
		t.Fatal("legacy task reservation changed", got, answer)
	}
}
