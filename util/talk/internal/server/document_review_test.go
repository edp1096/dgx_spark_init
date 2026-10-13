package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"sparktalk/internal/llm"
	"sparktalk/internal/workflows"
)

func reviewFixtureUnit() workflows.DocumentUnit {
	return workflows.DocumentUnit{Inputs: []workflows.DocumentInput{{ID: "s:a:0", Source: "s", Text: "She warmed the soup.", Group: "a"}, {ID: "s:b:0", Source: "s", Text: "Then she left.", Group: "b"}}, Output: map[string]string{"s:a:0": "그녀는 수프를 식혔다.", "s:b:0": "그러고는 떠났다."}}
}

func TestDocumentReviewRejectsMissingDuplicateAndUnanchoredEvidence(t *testing.T) {
	u := reviewFixtureUnit()
	f := workflows.DocumentFinding{Category: "negation_direction", Severity: "major", SourceQuote: "warmed", TargetQuote: "식혔다", Explanation: "가열을 냉각으로 바꿈"}
	valid := func() string {
		v, _ := json.Marshal(map[string]any{"reviews": map[string]any{"p000001": []workflows.DocumentFinding{f}, "p000002": []workflows.DocumentFinding{}}})
		return string(v)
	}
	review, err := validateDocumentReview(u, llm.StreamResult{Content: valid(), FinishReason: "stop"})
	if err != nil || review.Status != "needs_review" || len(review.Items) != 2 {
		t.Fatal(review, err)
	}
	for _, raw := range []string{
		`{"reviews":{"p000001":[]}}`,
		`{"reviews":{"p000001":[],"p000001":[],"p000002":[]}}`,
		`{"reviews":{"p000001":null,"p000002":[]}}`,
		`{"reviews":{"p000001":[],"p000003":[]}}`,
		valid() + ` {}`,
		strings.Replace(valid(), "warmed", "invented source", 1),
		strings.Replace(valid(), "식혔다", "존재하지 않는 번역", 1),
		strings.Replace(valid(), "negation_direction", "made_up_category", 1),
		strings.Replace(valid(), `,"uncertain":false`, "", 1),
	} {
		if _, err := validateDocumentReview(u, llm.StreamResult{Content: raw, FinishReason: "stop"}); err == nil {
			t.Error("invalid review accepted", raw)
		}
	}
	if _, err := validateDocumentReview(u, llm.StreamResult{Content: valid(), FinishReason: "length"}); err == nil {
		t.Fatal("truncated review accepted")
	}
	f.SourceQuote = "Then she left."
	f.TargetQuote = ""
	f.Category = "omission"
	if _, err := validateDocumentReview(u, llm.StreamResult{Content: valid(), FinishReason: "stop"}); err == nil {
		t.Fatal("evidence from another paragraph accepted")
	}
}

func TestDocumentReviewUsesDedicatedSchemaAndScopedTemperature(t *testing.T) {
	_, backend := documentFixture(t, func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		if body["temperature"] != 0.2 {
			t.Error("request temperature", body["temperature"])
		}
		format := body["response_format"].(map[string]any)["json_schema"].(map[string]any)
		if format["name"] != "document_review" {
			t.Error("wrong schema", format["name"])
		}
		content := `{"reviews":{"p000001":[],"p000002":[]}}`
		event, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]string{"content": content}, "finish_reason": "stop"}}})
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", event)
	})
	run := workflows.Run{Document: &workflows.DocumentTask{Model: "model", Kind: "translate", Effort: "none"}}
	u := reviewFixtureUnit()
	result, err := requestDocumentReview(llm.WithTemperature(context.Background(), 0.2), llm.New(backend.URL, "model", ""), run, u, 8192, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = validateDocumentReview(u, result); err != nil {
		t.Fatal(err)
	}
}

func TestDocumentReviewQuoteNormalizationKeepsOriginalEvidence(t *testing.T) {
	text := "그녀는 “돌아가자”고 말했다."
	actual, ok := documentAnchoredQuote(text, `그녀는 "돌아가자"고 말했다.`)
	if !ok || actual != text {
		t.Fatal(actual, ok)
	}
	if _, ok := documentAnchoredQuote(text, `그녀는 "남아있자"고 말했다.`); ok {
		t.Fatal("semantic substitution accepted as evidence")
	}
	if _, ok := documentAnchoredQuote(`“same” and “same”`, `"same"`); ok {
		t.Fatal("ambiguous normalized quote accepted")
	}
}

func TestDocumentReviewPartialPreservesAnchoredSiblingsButNeverPassesMissingReview(t *testing.T) {
	u := reviewFixtureUnit()
	raw := `{"reviews":{"p000001":[{"category":"lexical","severity":"major","source_quote":"invented","target_quote":"식혔다","explanation":"wrong evidence","uncertain":false}],"p000002":[]}}`
	review, err := validateDocumentReviewPartial(u, llm.StreamResult{Content: raw, FinishReason: "stop"})
	if err != nil || review.Status != "pending" || len(review.Invalid) != 1 || len(review.Items) != 1 || review.Items[0].ID != "p000002" {
		t.Fatal(review, err)
	}
	if review.Items[0].InputIDs[0] != "s:b:0" {
		t.Fatal("valid review lost its original source reference")
	}
	if _, err = validateDocumentReviewPartial(u, llm.StreamResult{Content: `{"reviews":{"p000001":[],"p000001":[],"p000002":[]}}`}); err == nil {
		t.Fatal("ambiguous duplicate keys partially accepted")
	}
}

func TestDocumentModelsSeeTheSameSchemaThatTheSamplerEnforces(t *testing.T) {
	calls := 0
	_, backend := documentFixture(t, func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			return
		}
		format := body["response_format"].(map[string]any)["json_schema"].(map[string]any)
		schema, _ := json.Marshal(format["schema"])
		messages := body["messages"].([]any)
		system := messages[0].(map[string]any)["content"].(string)
		if !strings.Contains(system, string(schema)) {
			t.Error("sampler schema was hidden from the model")
		}
		if !strings.Contains(system, `escape an ASCII double quote as \" and a backslash as \\`) {
			t.Error("quoted prose escaping instruction missing")
		}
		if format["name"] == "document_review" {
			if !strings.Contains(system, `"lexical"`) || !strings.Contains(system, `"source_quote"`) {
				t.Error("review fields or enums hidden")
			}
			if strings.Contains(string(schema), `"layout"`) || !strings.Contains(system, "script mixture alone is never a major error") {
				t.Error("presentation-only findings can trigger semantic rewrites")
			}
			sendReviewedResult(w, map[string]any{"checks": map[string]any{"p000001": []any{}, "p000002": []any{}}})
		} else {
			sendReviewedResult(w, map[string]any{"entries": map[string]string{"p000001": "그녀는 수프를 데웠다.", "p000002": "그러고는 떠났다."}})
		}
		calls++
	})
	run := workflows.Run{Document: &workflows.DocumentTask{Pipeline: workflows.DocumentPipelineVersion, Kind: "translate", Model: "model", Effort: "none"}}
	client := llm.New(backend.URL, "model", "")
	unit := reviewFixtureUnit()
	if _, err := requestDocumentReview(context.Background(), client, run, unit, 16384, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := requestDocumentUnit(context.Background(), client, run, unit, 16384); err != nil {
		t.Fatal(err)
	}
	if calls != 2 {
		t.Fatal("both stages were not exercised", calls)
	}
}

func TestDocumentReviewProgressKeepsVerifiedSiblingsDuringRepair(t *testing.T) {
	unit := reviewFixtureUnit()
	unit.Status = "completed"
	pairs := documentReviewPairs(unit)
	one := workflows.DocumentReviewItem{ID: pairs[0].ID, InputIDs: pairs[0].InputIDs, SourceHash: documentTextHash(pairs[0].Source), OutputHash: documentTextHash(pairs[0].Target), Findings: []workflows.DocumentFinding{}}
	unit.Quality = &workflows.DocumentUnitQuality{Status: "pending", OutputHash: documentUnitOutputHash(unit), Reviews: []workflows.DocumentReviewItem{one}}
	run := workflows.Run{Document: &workflows.DocumentTask{}}
	documentQualityCounters(&run, []workflows.DocumentUnit{unit})
	if run.Document.Reviewed != 1 || run.Document.ReviewTotal != 2 || run.Document.QualityStatus != "pending" || run.Document.Unresolved != 0 {
		t.Fatal("valid sibling disappeared during pending review", run.Document)
	}
	unit.Quality.Status = "needs_review"
	documentQualityCounters(&run, []workflows.DocumentUnit{unit})
	if run.Document.Reviewed != 1 || run.Document.Unresolved != 1 {
		t.Fatal("exhausted missing review was hidden", run.Document)
	}
	unit.Output["s:a:0"] = "새 번역"
	documentQualityCounters(&run, []workflows.DocumentUnit{unit})
	if run.Document.Reviewed != 0 {
		t.Fatal("changed target retained stale review")
	}
}
