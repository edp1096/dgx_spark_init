package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"testing"

	"sparktalk/internal/workflows"
)

func TestDocumentLanguageScanDoesNotDependOnGlossary(t *testing.T) {
	u := []workflows.DocumentUnit{{Ordinal: 0, Inputs: []workflows.DocumentInput{{ID: "a", Text: "She spoke to Della about the API."}}, Output: map[string]string{"a": "elle는 Della에게 API를 설명했다."}}}
	scan := documentLanguageScan(u)
	if len(scan.Records) != 1 || len(scan.Records[0].Mentions) != 1 || scan.Records[0].Mentions[0].Text != "elle는" {
		t.Fatal(scan)
	}
	// One legitimate source occurrence cannot mask a second misplaced use.
	u[0].Inputs[0].Text = "Elle smiled. She spoke."
	u[0].Output["a"] = "Elle는 웃었다. Elle는 말했다."
	if got := documentLanguageScan(u); len(got.Records) != 1 || len(got.Records[0].Mentions) != 2 {
		t.Fatal(got)
	}
}

func TestDocumentLanguageRepairsOnlyScopedOccurrence(t *testing.T) {
	u := []workflows.DocumentUnit{{Ordinal: 0, Inputs: []workflows.DocumentInput{{ID: "a", Text: "Elle smiled. She spoke."}}, Output: map[string]string{"a": "Elle는 웃었다. Elle는 말했다."}}}
	r := documentLanguageScan(u).Records[0]
	r.Checks = map[string]workflows.DocumentLanguageDecision{"m1": {Status: "retain", Reason: "인명 보존"}, "m2": {Status: "translate", SourceRef: "s1", After: "그녀는", Reason: "프랑스어 대명사 혼입"}}
	out, text, err := applyDocumentLanguage(u, r)
	if err != nil || text != "Elle는 웃었다. 그녀는 말했다." || out["a"] != text {
		t.Fatal(text, err)
	}
	if u[0].Output["a"] != "Elle는 웃었다. Elle는 말했다." {
		t.Fatal("unverified edit applied")
	}
	bad := r.Checks["m2"]
	bad.After = "그녀는 떠났다."
	r.Checks["m2"] = bad
	if _, _, err = applyDocumentLanguage(u, r); err == nil {
		t.Fatal("sentence rewrite accepted")
	}
	bad.After = "그녀는"
	bad.SourceRef = "invented"
	r.Checks["m2"] = bad
	if _, _, err = applyDocumentLanguage(u, r); err == nil {
		t.Fatal("ungrounded edit accepted")
	}
	bad.SourceRef = "s1"
	r.Checks["m2"] = bad
	r.TargetHash = "stale"
	if _, _, err = applyDocumentLanguage(u, r); err == nil {
		t.Fatal("stale edit accepted")
	}
}

func TestDocumentLanguageKeepsConnectedForeignFragmentTogether(t *testing.T) {
	u := []workflows.DocumentUnit{{Ordinal: 0, Inputs: []workflows.DocumentInput{{ID: "a", Text: "But then—I should not be hungry."}}, Output: map[string]string{"a": "하지만 그—if—I가 배가 고프지 않을 거예요."}}}
	r := documentLanguageScan(u).Records[0]
	if len(r.Mentions) != 1 || r.Mentions[0].Text != "if—I가" {
		t.Fatal(r.Mentions)
	}
	r.Checks = map[string]workflows.DocumentLanguageDecision{"m1": {Status: "translate", SourceRef: "s1", After: "내가", Reason: "원문의 1인칭 주어"}}
	_, text, err := applyDocumentLanguage(u, r)
	if err != nil || text != "하지만 그—내가 배가 고프지 않을 거예요." {
		t.Fatal(text, err)
	}
}

func TestDocumentLanguagePreservesGlossesAndURLsWithoutHidingProse(t *testing.T) {
	u := []workflows.DocumentUnit{{Ordinal: 0, Inputs: []workflows.DocumentInput{{ID: "a", Text: "She taught us and shared a link."}}, Output: map[string]string{"a": "elle는 가르쳤다(teach). elle (she) https://example.org 그리고 elle는 말했다."}}}
	r := documentLanguageScan(u).Records[0]
	if len(r.Mentions) != 3 {
		t.Fatal(r.Mentions)
	}
	for _, m := range r.Mentions {
		if m.Text != "elle는" && m.Text != "elle" {
			t.Fatal("gloss/URL included", m)
		}
	}
}

func TestDocumentLanguageCorrectionThroughTalkAction(t *testing.T) {
	for _, reject := range []bool{false, true} {
		t.Run(fmt.Sprint(reject), func(t *testing.T) {
			var calls atomic.Int32
			s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) {
				raw, _ := io.ReadAll(r.Body)
				calls.Add(1)
				response := ""
				switch {
				case strings.Contains(string(raw), "Check the listed expressions"):
					if strings.Contains(string(raw), "예시: 그녀는 ~") {
						t.Error("reviewer example leaked into new proposal")
					}
					response = `{"checks":{"l000001":{"m1":{"status":"translate","source_ref":"s1","after":"그녀는","reason":"같은 뜻의 제3언어 대명사가 출력됨"}},"l000002":{"m1":{"status":"retain","source_ref":"","after":"","reason":"기술 약어 유지"}}}}`
					if calls.Load() > 2 {
						response = `{"checks":{"l000001":{"m1":{"status":"translate","source_ref":"s1","after":"그녀는","reason":"원문 대조"}}}}`
					}
				case strings.Contains(string(raw), "Independently compare the highlighted"):
					response = fmt.Sprintf(`{"valid":%t,"reason":"원문 지칭과 문법 대조"}`, !reject)
					if reject {
						response = `{"valid":false,"reason":"예시: 그녀는 ~라고 말했다."}`
					}
				default:
					t.Errorf("unexpected request")
					response = `{}`
				}
				event, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]string{"content": response}, "finish_reason": "stop"}}})
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", event)
			})
			source := fixtureSource(t, s, "She spoke.\n\nThe graphics processor was fast.")
			book, _ := s.db.Source("s", source, false)
			units := planDocumentTranslation(source, book, 0, 0)
			for i := range units {
				units[i].Status = "completed"
				units[i].Output = map[string]string{}
				for _, in := range units[i].Inputs {
					v := "그 GPU는 빨랐다."
					if strings.HasPrefix(in.Text, "She") {
						v = "elle는 말했다."
					}
					units[i].Output[in.ID] = v
				}
			}
			parent := workflows.Run{ID: "language-parent", SessionID: "s", Status: "completed", Current: 2, Document: &workflows.DocumentTask{Kind: "translate", Sources: []string{source}, Total: len(units), Completed: len(units), Phase: "completed", Revision: 1, RequestSpec: &workflows.DocumentRequestSpec{Origin: "api", UserText: "한국어로 번역"}}, Steps: []workflows.StepState{{Status: "completed"}, {Status: "completed"}}}
			if err := s.db.CreateDocumentRun(&parent, units); err != nil {
				t.Fatal(err)
			}
			run, err := s.documentAction(context.Background(), "s", documentStart{Action: "repair_terms", RunID: parent.ID})
			if err != nil {
				t.Fatal(err)
			}
			run = waitDocument(t, s, run.ID)
			out, _, err := s.db.DocumentOutput("s", run.ID, "output.txt")
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(out), "GPU는") {
				t.Fatal("normal acronym changed", string(out))
			}
			if reject {
				if run.Document.Terms.Changed != 0 || run.Document.Terms.Unresolved != 1 || !strings.Contains(string(out), "elle는") {
					t.Fatal(run.Document.Terms, string(out))
				}
			} else {
				if run.Status != "completed" || run.Document.Terms.Changed != 1 || run.Document.Terms.Unresolved != 0 || strings.Contains(string(out), "elle는") || !strings.Contains(string(out), "그녀는 말했다.") {
					t.Fatal(run.Status, run.Document.Terms, string(out))
				}
			}
			old, _ := s.db.DocumentUnits("s", parent.ID)
			if old[0].Output[old[0].Inputs[0].ID] != "elle는 말했다." {
				t.Fatal("parent changed")
			}
		})
	}
}

func TestDocumentLanguageRejectsUncontractedSubjectPronouns(t *testing.T) {
	u := []workflows.DocumentUnit{{Ordinal: 0, Inputs: []workflows.DocumentInput{{ID: "a", Text: "But then—I should not be hungry."}}, Output: map[string]string{"a": "하지만 if—I가 배가 고프지 않을 거예요."}}}
	r := documentLanguageScan(u).Records[0]
	for _, after := range []string{"나가", "저가", "너가", "내가", "제가", "네가"} {
		r.Checks = map[string]workflows.DocumentLanguageDecision{"m1": {Status: "translate", SourceRef: "s1", After: after, Reason: "원문 대조"}}
		_, _, err := applyDocumentLanguage(u, r)
		invalid := after == "나가" || after == "저가" || after == "너가"
		if (err != nil) != invalid {
			t.Fatal(after, err)
		}
	}
	u[0].Inputs[0].Text = "It is an inexpensive product."
	u[0].Output["a"] = "그것은 low-cost 제품이다."
	r = documentLanguageScan(u).Records[0]
	r.Checks = map[string]workflows.DocumentLanguageDecision{"m1": {Status: "translate", SourceRef: "s1", After: "저가", Reason: "가격을 설명하는 관형어"}}
	if _, _, err := applyDocumentLanguage(u, r); err != nil {
		t.Fatal(err)
	}
}
