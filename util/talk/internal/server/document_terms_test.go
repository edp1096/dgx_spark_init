package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"sparktalk/internal/llm"
	"sparktalk/internal/sourcebook"
	"sparktalk/internal/workflows"
)

func TestTermBoundariesParticlesAndMinimalPatches(t *testing.T) {
	for _, tc := range []struct {
		text, name string
		n          int
	}{
		{"디나단은 디나와 다르다.", "디나", 1},
		{"Ann and Anna visited Annex.", "Ann", 1},
		{"모리에게는 모리의 편지가 있다.", "모리", 2},
		{"Pip's footprints; Pippin stayed.", "Pip", 1},
		{"모리만이 모리에게만은 말한다.", "모리", 2},
		{"모리만들기와 모리단은 이름이 다르다.", "모리", 0},
	} {
		if got := len(termRanges(tc.text, tc.name)); got != tc.n {
			t.Fatalf("%+v: %d", tc, got)
		}
	}
	u := []workflows.DocumentUnit{{Ordinal: 0, Inputs: []workflows.DocumentInput{{ID: "a"}}, Output: map[string]string{"a": "페린단이 왔다. 페린단 이야기를 듣자!"}}}
	r := workflows.DocumentTermReference{Ordinal: 0, InputIDs: []string{"a"}, TargetHash: documentTextHash(u[0].Output["a"]), Forms: []string{"페린단"}}
	patches := []workflows.DocumentTermPatch{{Before: "페린단이", After: "페린이"}, {Before: "페린단 이야기를", After: "페린 이야기를"}}
	fixed, text, err := applyTermPatches(u, r, "페린", patches)
	if err != nil || text != "페린이 왔다. 페린 이야기를 듣자!" || fixed["a"] != text {
		t.Fatalf("%s %v", text, err)
	}
	if u[0].Output["a"] != "페린단이 왔다. 페린단 이야기를 듣자!" {
		t.Fatal("source checkpoint mutated before verification")
	}
	for _, bad := range [][]workflows.DocumentTermPatch{
		{{Before: "페린단이 왔다.", After: "페린이 떠났다."}},
		{{Before: "페린단", After: "페린"}}, // leaves invalid overlapping particle change if other edits supplied
		{{Before: "페린단이", After: "페린이"}, {Before: "페린단이", After: "페린은"}},
	} {
		if _, _, err := applyTermPatches(u, r, "페린", bad); err == nil {
			t.Fatal("unsafe patch accepted", bad)
		}
	}
	r.TargetHash = "stale"
	if _, _, err := applyTermPatches(u, r, "페린", patches); err == nil {
		t.Fatal("stale correction accepted")
	}
}

func TestTermHTMLCorrectionPreservesRubyLinksAndAuthoredBreaks(t *testing.T) {
	s, _ := documentFixture(t, func(http.ResponseWriter, *http.Request) { t.Error("unexpected inference") })
	raw := []byte(`<html><body><main><h2 id="chapter">Chapter I</h2><p><a href="#note">Perrin</a> saw <ruby>山<rt>やま</rt></ruby>.</p><p><em>Perrin</em> returned.<br>Next line.</p><pre>first line
second line</pre><p id="note">Note <a href="#chapter">back</a>.</p></main></body></html>`)
	book, err := sourcebook.Parse(raw, "text/html", "https://example.org/wikisource-shaped-document", nil)
	if err != nil {
		t.Fatal(err)
	}
	source, err := s.storeSource("s", book, raw)
	if err != nil {
		t.Fatal(err)
	}
	units := planDocumentTranslation(source, book, 0, 0)
	mentions := 0
	for i := range units {
		units[i].Status = "completed"
		units[i].Output = map[string]string{}
		for _, in := range units[i].Inputs {
			value := in.Text
			if value == "Perrin" {
				mentions++
				value = "페린"
				if mentions == 2 {
					value = "페린단"
				}
			}
			units[i].Output[in.ID] = value
		}
	}
	run := workflows.Run{ID: "html-terms", SessionID: "s", Status: "completed", Current: 2, Document: &workflows.DocumentTask{Kind: "translate", Sources: []string{source}, Terms: &workflows.DocumentTerms{Candidates: []workflows.DocumentEntity{{Source: "Perrin"}}}}}
	cases := documentTermCases(run, units)
	if mentions != 2 || len(cases) != 1 || len(cases[0].References) != 2 {
		t.Fatal("inline name indexing", cases, mentions)
	}
	ref := cases[0].References[1]
	ref.Forms = []string{"페린단"}
	out, _, err := applyTermPatches(units, ref, "페린", []workflows.DocumentTermPatch{{Before: "페린단", After: "페린"}})
	if err != nil {
		t.Fatal(err)
	}
	units[ref.Ordinal].Output = out
	if err = s.db.CreateDocumentRun(&run, units); err != nil {
		t.Fatal(err)
	}
	if err = s.exportDocument(context.Background(), run, units); err != nil {
		t.Fatal(err)
	}
	html, _, err := s.db.DocumentOutput("s", run.ID, "output.html")
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`<em>페린</em>`, `<ruby>`, `<rt>やま</rt>`, `href="#note"`, `href="#chapter"`, `<br>`, "first line\nsecond line"} {
		if !strings.Contains(string(html), want) {
			t.Fatal("lost markup", want, string(html))
		}
	}
	if strings.Contains(string(html), "페린단") {
		t.Fatal("correction missing")
	}
}

func TestTermExtractionEvidenceAndDecisionCoverage(t *testing.T) {
	refs := []workflows.DocumentTermReference{{ID: "r1", Source: "Perrin was a cat.", Target: "페린은 고양이였다."}, {ID: "r2", Source: "Perrin returned.", Target: "페린단이 돌아왔다."}}
	c := workflows.DocumentTermCase{Source: "Perrin", References: refs}
	result := llm.StreamResult{Content: `{"checks":{"r1":{"status":"named","forms":["페린"],"evidence":"Perrin was a cat.","referent":"고양이","reason":"확인"},"r2":{"status":"named","forms":["페린"],"evidence":"Perrin returned.","referent":"고양이","reason":"확인"}}}`, FinishReason: "stop"}
	invalid := validateTermExtraction(&c, refs, result)
	if invalid["r1"] != nil || invalid["r2"] == nil {
		t.Fatal("a shorter name inside a different spelling was accepted", invalid)
	}
	if c.References[0].Status != "named" {
		t.Fatal("good sibling lost")
	}
	c.References[1].Status = "named"
	c.References[1].Forms = []string{"페린단"}
	for _, raw := range []string{
		`{"groups":[{"references":["r1"],"canonical":"페린","status":"correctable","reason":"표기 혼재"}]}`,
		`{"groups":[{"references":["r1","r1"],"canonical":"페린","status":"correctable","reason":"표기 혼재"}]}`,
		`{"groups":[{"references":["r1","r2"],"canonical":"페린","status":"consistent","reason":"문제없음"}]}`,
	} {
		if validateTermDecision(&c, llm.StreamResult{Content: raw, FinishReason: "stop"}) == nil {
			t.Fatal("unchecked/false agreement accepted")
		}
	}
}

func TestTermCandidatesDoNotBecomeAuthoritativeGlossary(t *testing.T) {
	d := &workflows.DocumentTask{Terms: &workflows.DocumentTerms{}, Glossary: map[string]string{"Perrin": "잘못된이름"}, GlossaryKinds: map[string]string{"Perrin": "person"}, UserGlossary: map[string]string{"Town": "마을"}}
	actual := activeDocumentGlossary(d)
	if len(actual) != 1 || actual["Town"] != "마을" {
		t.Fatal("unverified target propagated", actual)
	}
}

func TestTermCorrectionThroughActionPreservesParentAndRepairsOnlyNames(t *testing.T) {
	for _, mode := range []string{"accepted", "extraction-retry", "rejected", "pause-resume", "cancel-retry"} {
		t.Run(mode, func(t *testing.T) { exerciseTermCorrection(t, mode) })
	}
}
func exerciseTermCorrection(t *testing.T, mode string) {
	var calls, extracts, repairs, verifications atomic.Int32
	entered := make(chan struct{}, 1)
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		text := string(raw)
		calls.Add(1)
		response := ""
		switch {
		case strings.Contains(text, "locate the supplied original expression"):
			n := extracts.Add(1)
			response = `{"checks":{"r000001":{"status":"named","forms":["페린"],"evidence":"Perrin was a cat.","referent":"고양이","reason":"표기 확인"},"r000002":{"status":"named","forms":["페린단"],"evidence":"Perrin returned.","referent":"고양이","reason":"다른 표기"}}}`
			if mode == "extraction-retry" {
				if n == 1 {
					response = strings.Replace(response, `"forms":["페린단"]`, `"forms":["없는이름"]`, 1)
				} else {
					if strings.Contains(text, "r000001") || !strings.Contains(text, "translated form is absent") || !strings.Contains(text, "evidence_options") {
						t.Error("retry did not isolate failed reference with evidence and feedback")
					}
					response = `{"checks":{"r000002":{"status":"named","forms":["페린단"],"evidence":"Perrin returned.","referent":"고양이","reason":"실제 표기 확인"}}}`
				}
			}
		case strings.Contains(text, "Compare the original expression"):
			response = `{"groups":[{"references":["r000001","r000002"],"canonical":"페린","status":"correctable","reason":"동일 고양이의 표기 혼재"}]}`
		case strings.Contains(text, "Propose minimal literal"):
			repairs.Add(1)
			response = `{"patches":{"r000002":[{"before":"페린단이","after":"페린이"}]}}`
		case strings.Contains(text, "Independently check"):
			n := verifications.Add(1)
			if (mode == "pause-resume" || mode == "cancel-retry") && n == 1 {
				entered <- struct{}{}
				<-r.Context().Done()
				return
			}
			response = `{"checks":{"r000002":{"valid":true,"reason":"고양이 이름만 수정됨"}}}`
			if mode == "rejected" {
				response = `{"checks":{"r000002":{"valid":false,"reason":"원문 대조에서 확인되지 않은 교정"}}}`
			}
		default:
			t.Errorf("unexpected model stage")
			response = `{}`
		}
		event, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]string{"content": response}, "finish_reason": "stop"}}, "usage": map[string]int{"prompt_tokens": 100, "completion_tokens": 50}})
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", event)
	})
	source := fixtureSource(t, s, "Perrin was a cat.\n\nPerrin returned.")
	book, _ := s.db.Source("s", source, false)
	cfg, _ := s.snapshot()
	parent := workflows.Run{ID: "parent-terms", SessionID: "s", Status: "completed", Current: 2, Document: &workflows.DocumentTask{Kind: "translate", Sources: []string{source}, Phase: "completed", Total: 1, Completed: 1, Revision: 1, Model: cfg.Model.DefaultModel, ModelType: cfg.Model.ModelType, Endpoint: cfg.Model.Endpoint, Glossary: map[string]string{"Perrin": "페린단"}, RequestSpec: &workflows.DocumentRequestSpec{Origin: "api", UserText: "한국어로 번역"}, Formats: []string{"txt", "md", "html", "zip"}}, Steps: []workflows.StepState{{Status: "completed"}, {Status: "completed"}}}
	units := planDocumentTranslation(source, book, 0, 0)
	for i := range units {
		units[i].Status = "completed"
		units[i].Output = map[string]string{}
		for _, in := range units[i].Inputs {
			if strings.Contains(in.Text, "was") {
				units[i].Output[in.ID] = "페린은 고양이였다."
			} else {
				units[i].Output[in.ID] = "페린단이 돌아왔다."
			}
		}
	}
	if err := s.db.CreateDocumentRun(&parent, units); err != nil {
		t.Fatal(err)
	}
	if err := s.db.SaveDocumentOutput("s", parent.ID, "output.txt", "text/plain", []byte("original result")); err != nil {
		t.Fatal(err)
	}
	run, err := s.documentAction(context.Background(), "s", documentStart{Action: "repair_terms", RunID: parent.ID})
	if err != nil {
		t.Fatal(err)
	}
	if mode == "pause-resume" || mode == "cancel-retry" {
		select {
		case <-entered:
		case <-time.After(3 * time.Second):
			t.Fatal("verification not reached")
		}
		action, status := "pause", "paused"
		if mode == "cancel-retry" {
			action, status = "cancel", "canceled"
		}
		paused, err := s.documentAction(context.Background(), "s", documentStart{Action: action, RunID: run.ID})
		if err != nil || paused.Status != status {
			t.Fatal(paused.Status, err)
		}
		if repairs.Load() != 1 {
			t.Fatal("unexpected pre-pause repairs")
		}
		if mode == "cancel-retry" {
			oldID := run.ID
			run, err = s.documentAction(context.Background(), "s", documentStart{Action: "repair_terms", RunID: parent.ID})
			if err != nil || run.ID == oldID {
				t.Fatal("canceled request was reused", run.ID, err)
			}
		} else if _, err = s.documentAction(context.Background(), "s", documentStart{Action: "resume", RunID: run.ID}); err != nil {
			t.Fatal(err)
		}
	}
	run = waitDocument(t, s, run.ID)
	if mode == "rejected" {
		out, _, err := s.db.DocumentOutput("s", run.ID, "output.txt")
		if err != nil || run.Status != "needs_review" || run.Document.Terms.Changed != 0 || run.Document.Terms.Unresolved != 1 || !strings.Contains(string(out), "페린단이") || repairs.Load() != 2 {
			t.Fatal(run.Status, run.Document.Terms, string(out), err)
		}
		return
	}
	expectedCalls := int32(4)
	if mode == "extraction-retry" {
		expectedCalls = 5
	}
	if mode == "cancel-retry" {
		expectedCalls = 8
	}
	if mode == "pause-resume" {
		expectedCalls = 5
		if repairs.Load() != 1 {
			t.Fatal("completed proposal generated again")
		}
	}
	if run.Status != "completed" || !run.DocumentFinished() || run.Document.Terms.Changed != 1 || run.Document.Terms.Unresolved != 0 || calls.Load() != expectedCalls {
		t.Fatalf("status=%s terms=%+v calls=%d error=%s", run.Status, run.Document.Terms, calls.Load(), run.Document.Error)
	}
	output, _, err := s.db.DocumentOutput("s", run.ID, "output.txt")
	if err != nil || !strings.Contains(string(output), "페린이 돌아왔다.") || strings.Contains(string(output), "페린단") {
		t.Fatal(string(output), err)
	}
	old, _, _ := s.db.DocumentOutput("s", parent.ID, "output.txt")
	if string(old) != "original result" {
		t.Fatal("parent output changed")
	}
	again, err := s.documentAction(context.Background(), "s", documentStart{Action: "repair_terms", RunID: parent.ID})
	if err != nil || again.ID != run.ID {
		t.Fatal("duplicate correction execution", again.ID, err)
	}
	if run.Document.Publication == nil || run.Document.QualityStatus == "reviewed" {
		t.Fatal("missing publication or false full semantic review")
	}
	if mode == "accepted" {
		recheck, err := s.documentAction(context.Background(), "s", documentStart{Action: "repair_terms", RunID: run.ID})
		if err != nil {
			t.Fatal(err)
		}
		recheck = waitDocument(t, s, recheck.ID)
		if recheck.Status != "completed" || recheck.Document.Calls != 0 || recheck.Document.Terms.Reused != 2 || calls.Load() != expectedCalls {
			t.Fatalf("verified revision was reprocessed: %+v calls=%d", recheck.Document.Terms, calls.Load())
		}
		out, _, err := s.db.DocumentOutput("s", recheck.ID, "output.txt")
		if err != nil || string(out) != string(output) {
			t.Fatal("recheck altered verified output", err)
		}
	}
}

func TestTermPatchesRespectInlineBoundariesAndAllOccurrences(t *testing.T) {
	units := []workflows.DocumentUnit{{Ordinal: 0, Output: map[string]string{"a": "루멘단은 ", "b": "루멘단을 기다렸다."}}}
	r := workflows.DocumentTermReference{Ordinal: 0, InputIDs: []string{"a", "b"}, TargetHash: documentTextHash("루멘단은 루멘단을 기다렸다."), Forms: []string{"루멘단"}}
	if _, _, err := applyTermPatches(units, r, "루멘", []workflows.DocumentTermPatch{{Before: "루멘단은", After: "루멘은"}}); err == nil {
		t.Fatal("forgotten occurrence accepted")
	}
	if _, _, err := applyTermPatches(units, r, "루멘", []workflows.DocumentTermPatch{{Before: "루멘단은 루멘단을", After: "루멘은 루멘을"}}); err == nil {
		t.Fatal("whole cross-node rewrite accepted")
	}
	out, _, err := applyTermPatches(units, r, "루멘", []workflows.DocumentTermPatch{{Before: "루멘단은", After: "루멘은"}, {Before: "루멘단을", After: "루멘을"}})
	if err != nil || out["a"] != "루멘은 " || out["b"] != "루멘을 기다렸다." {
		t.Fatal(out, err)
	}
}

func TestTermDecisionKeepsHomonymsSeparate(t *testing.T) {
	c := workflows.DocumentTermCase{Source: "Aurora", References: []workflows.DocumentTermReference{
		{ID: "a", Status: "named", Forms: []string{"오로라"}, Referent: "ship"},
		{ID: "b", Status: "named", Forms: []string{"아우로라"}, Referent: "singer"},
	}}
	r := llm.StreamResult{FinishReason: "stop", Content: `{"groups":[{"references":["a"],"canonical":"오로라","status":"consistent","reason":"ship"},{"references":["b"],"canonical":"아우로라","status":"consistent","reason":"singer"}]}`}
	if err := validateTermDecision(&c, r); err != nil || len(c.Decisions) != 2 {
		t.Fatal(c, err)
	}
	r.Content = `{"groups":[{"references":["a","b"],"canonical":"새이름","status":"correctable","reason":"invented"}]}`
	if validateTermDecision(&c, r) == nil {
		t.Fatal("invented canonical spelling accepted")
	}
}

func TestTermProgressDoesNotDuplicateBookOrMutateSavedEvidence(t *testing.T) {
	d := &workflows.DocumentTask{Terms: &workflows.DocumentTerms{Cases: []workflows.DocumentTermCase{{Source: "Name", References: []workflows.DocumentTermReference{{Source: "source paragraph", Target: "target paragraph", Evidence: "source"}}}}}}
	public := documentPublicTask(d)
	if public.Terms.Cases[0].References[0].Source != "" || public.Terms.Cases[0].References[0].Target != "" || d.Terms.Cases[0].References[0].Source == "" {
		t.Fatal("public view exposed full duplicate text or mutated checkpoint")
	}
}

func TestTermNewTranslationHTTPAutomaticallyChecksCandidates(t *testing.T) {
	var calls atomic.Int32
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			return
		}
		calls.Add(1)
		system, user := body.Messages[0].Content, body.Messages[len(body.Messages)-1].Content
		var response any
		switch {
		case strings.Contains(system, "Translate ALL"):
			parts := strings.SplitN(user, "Reference data (JSON):\n", 2)
			var entries []map[string]string
			if len(parts) != 2 || json.Unmarshal([]byte(parts[1]), &entries) != nil {
				t.Error("missing translation data")
				return
			}
			out := map[string]string{}
			var terms []any
			for _, in := range entries {
				out[in["id"]] = "페린단이 돌아왔다."
				if strings.Contains(in["text"], "was") {
					out[in["id"]] = "페린은 고양이였다."
					terms = append(terms, map[string]any{"source": "Perrin", "target": "페린", "kind": "animal", "evidence": map[string]string{"id": in["id"], "quote": "Perrin was a cat."}})
				}
			}
			response = map[string]any{"entries": out, "terms": terms}
		case strings.Contains(system, "locate the supplied original expression"):
			json.Unmarshal([]byte(`{"checks":{"r000001":{"status":"named","forms":["페린"],"evidence":"Perrin was a cat.","referent":"고양이","reason":"이름"},"r000002":{"status":"named","forms":["페린단"],"evidence":"Perrin returned.","referent":"고양이","reason":"이름"}}}`), &response)
		case strings.Contains(system, "Compare the original expression"):
			json.Unmarshal([]byte(`{"groups":[{"references":["r000001","r000002"],"canonical":"페린","status":"correctable","reason":"동일한 고양이"}]}`), &response)
		case strings.Contains(system, "Propose minimal literal"):
			json.Unmarshal([]byte(`{"patches":{"r000002":[{"before":"페린단이","after":"페린이"}]}}`), &response)
		case strings.Contains(system, "Independently check"):
			json.Unmarshal([]byte(`{"checks":{"r000002":{"valid":true,"reason":"원문과 일치"}}}`), &response)
		default:
			t.Error("unexpected stage")
		}
		raw, _ := json.Marshal(response)
		event, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]string{"content": string(raw)}, "finish_reason": "stop"}}, "usage": map[string]int{"prompt_tokens": 100, "completion_tokens": 50}})
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", event)
	})
	source := fixtureSource(t, s, "Perrin was a cat.\n\nPerrin returned.")
	payload, _ := json.Marshal(map[string]any{"kind": "translate", "sources": []string{source}, "instructions": "한국어로 번역"})
	req := httptest.NewRequest("POST", "/api/document-tasks?session_id=s", strings.NewReader(string(payload)))
	w := httptest.NewRecorder()
	s.documentTaskAPI(w, req)
	if w.Code != 202 {
		t.Fatal(w.Code, w.Body.String())
	}
	var view struct {
		RunID string `json:"run_id"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &view); err != nil || view.RunID == "" {
		t.Fatal(w.Body.String(), err)
	}
	run := waitDocument(t, s, view.RunID)
	if run.Status != "completed" || run.Document.Pipeline != workflows.DocumentTermsVersion || run.Document.Terms.Changed != 1 || run.Document.Calls != 5 || calls.Load() != 5 {
		t.Fatalf("unexpected new translation result: %+v calls=%d", run.Document, calls.Load())
	}
	promptTokens := 0
	for _, stage := range run.Document.StageMetrics {
		promptTokens += stage.PromptTokens
	}
	if promptTokens != 500 || run.Document.OutputTokens != 250 {
		t.Fatal("translation usage lost or duplicated", promptTokens, run.Document.OutputTokens)
	}
	out, _, err := s.db.DocumentOutput("s", run.ID, "output.txt")
	if err != nil || strings.Contains(string(out), "페린단") || !strings.Contains(string(out), "페린이 돌아왔다.") {
		t.Fatal(string(out), err)
	}
}

func TestTermPairedTranslationAndPronunciationAreNotCollapsed(t *testing.T) {
	for _, tc := range []struct {
		text   string
		forms  []string
		status string
	}{
		{"봄의 노래(송 오브 스프링)를 불렀다.", []string{"봄의 노래", "송 오브 스프링"}, "kept"},
		{"메이（May）는 왔다.", []string{"메이", "May"}, "kept"},
		{"아침별 [모닝스타]은 밝다.", []string{"아침별", "모닝스타"}, "kept"},
		{"봄의 노래(송 오브 스프링). 봄의 노래를 들었다.", []string{"봄의 노래", "송 오브 스프링"}, "kept"},
		{"봄의 노래(송 오브 스프링). 봄의 노래를 들었다.", []string{"봄의 노래(송 오브 스프링)", "봄의 노래"}, "kept"},
		{"메이（May）는 왔다. May가 말했다.", []string{"메이（May）", "May"}, "kept"},
		{"봄의 노래(송 오브 스프링). 봄의 노래단을 들었다.", []string{"봄의 노래", "송 오브 스프링", "봄의 노래단"}, "uncertain"},
	} {
		r := workflows.DocumentTermReference{Target: tc.text, Forms: tc.forms, Status: "named", Patches: []workflows.DocumentTermPatch{{Before: tc.forms[0], After: tc.forms[1]}}}
		if !preserveTermPair(&r) || r.Status != tc.status || len(r.Patches) > 0 {
			t.Fatal(tc, r)
		}
	}
	r := workflows.DocumentTermReference{Target: "메이는 왔지만 메이단은 떠났다.", Forms: []string{"메이", "메이단"}, Status: "named"}
	if preserveTermPair(&r) || r.Status != "named" {
		t.Fatal("ordinary inconsistency hidden", r)
	}
}

func TestTermCompoundSourceNamesAcrossLineWraps(t *testing.T) {
	for _, text := range []string{"The White\nRabbit arrived.", "The White\t Rabbit arrived.", "The White\u00a0Rabbit arrived."} {
		if !termSourceMention(text, "White Rabbit") {
			t.Fatal("wrapped name lost", text)
		}
	}
	if termSourceMention("The White Rabbits arrived.", "White Rabbit") || termSourceMention("Joanna arrived.", "Anna") {
		t.Fatal("word boundary lost")
	}
	u := []workflows.DocumentUnit{{Ordinal: 0, Inputs: []workflows.DocumentInput{{ID: "a", Block: "b", Text: "The White\nRabbit arrived."}}, Output: map[string]string{"a": "흰 토끼가 왔다."}}}
	run := workflows.Run{Document: &workflows.DocumentTask{Terms: &workflows.DocumentTerms{Candidates: []workflows.DocumentEntity{{Source: "White Rabbit"}}}}}
	cases := documentTermCases(run, u)
	if len(cases) != 1 || len(cases[0].References) != 1 || cases[0].References[0].Source != "The White\nRabbit arrived." {
		t.Fatal("index lost source offsets", cases)
	}
	result := llm.StreamResult{FinishReason: "stop", Content: `{"checks":{"r000001":{"status":"named","forms":["흰 토끼"],"evidence":"White\nRabbit","referent":"토끼","reason":"원문 이름"}}}`}
	if errors := validateTermExtraction(&cases[0], cases[0].References, result); len(errors) != 0 {
		t.Fatal(errors)
	}
}

func TestTermTargetNamesAcrossSoftWrapsKeepByteOffsets(t *testing.T) {
	for _, tc := range []struct {
		text, name string
		n          int
	}{
		{"‘_윌리엄 아버지_’라고 말했다.", "윌리엄 아버지", 1},
		{"_윌리엄 아버지, 당신은 늙었소_", "윌리엄 아버지", 1},
		{"‘푸른\n토끼요,’라고 말했다.", "푸른 토끼", 1},
		{"Ann and Anna.", "Ann", 1},
	} {
		if len(termRanges(tc.text, tc.name)) != tc.n {
			t.Fatal(tc)
		}
	}
	text := "에드거\n아셀링과 에드거 아셀링은 왔다."
	u := []workflows.DocumentUnit{{Ordinal: 0, Inputs: []workflows.DocumentInput{{ID: "a", Whitespace: "normal"}}, Output: map[string]string{"a": text}}}
	r := workflows.DocumentTermReference{Ordinal: 0, InputIDs: []string{"a"}, Target: text, TargetHash: documentTextHash(text), Forms: []string{"에드거 아셀링"}}
	patches := []workflows.DocumentTermPatch{{Before: "에드거\n아셀링과", After: "에드가 애설링과"}, {Before: "에드거 아셀링은", After: "에드가 애설링은"}}
	out, got, err := applyTermPatches(u, r, "에드가 애설링", patches)
	if err != nil || got != "에드가 애설링과 에드가 애설링은 왔다." || out["a"] != got {
		t.Fatal(got, err)
	}
	if u[0].Output["a"] != text {
		t.Fatal("original changed")
	}
	u[0].Inputs[0].Whitespace = "pre-wrap"
	if _, _, err := applyTermPatches(u, r, "에드가 애설링", patches); err == nil {
		t.Fatal("authored line flattened")
	}
}

func TestTermShortFormNeverEditsInsideLongerCanonicalName(t *testing.T) {
	for _, text := range []string{"푸른 토끼는 웃었다. 토끼는 돌아왔다.", "푸른\n토끼는 웃었다. 토끼는 돌아왔다."} {
		units := []workflows.DocumentUnit{{Ordinal: 0, Output: map[string]string{"a": text}}}
		ref := workflows.DocumentTermReference{Ordinal: 0, InputIDs: []string{"a"}, TargetHash: documentTextHash(text), Forms: []string{"토끼", "푸른 토끼"}}
		_, got, err := applyTermPatches(units, ref, "푸른 토끼", []workflows.DocumentTermPatch{{Before: "토끼는", After: "푸른 토끼는"}})
		expected := strings.TrimSuffix(text, "토끼는 돌아왔다.") + "푸른 토끼는 돌아왔다."
		if err != nil || got != expected {
			t.Fatal(got, err)
		}
	}
}

func TestTermCompoundNameCannotBeShortenedByFrequency(t *testing.T) {
	c := workflows.DocumentTermCase{Source: "Silver Fox", References: []workflows.DocumentTermReference{{ID: "r1", Status: "named", Forms: []string{"은빛 여우", "여우"}}}}
	g := workflows.DocumentTermDecision{References: []string{"r1"}, Canonical: "여우", Status: "correctable", Reason: "frequent spelling"}
	if !termDecisionDropsComponent(c, g) {
		t.Fatal("loss of compound name was accepted")
	}
	raw, _ := json.Marshal(map[string]any{"groups": []workflows.DocumentTermDecision{g}})
	if validateTermDecision(&c, llm.StreamResult{Content: string(raw), FinishReason: "stop"}) == nil {
		t.Fatal("shortening accepted")
	}
	c.Pinned = "여우"
	if termDecisionDropsComponent(c, g) {
		t.Fatal("explicit user preference rejected")
	}
}

func TestTermRevocationRestoresOnlyAnUnchangedSavedCorrection(t *testing.T) {
	for _, mode := range []string{"legacy", "inline", "stale"} {
		t.Run(mode, func(t *testing.T) {
			s, _ := documentFixture(t, func(http.ResponseWriter, *http.Request) { t.Error("unexpected inference") })
			before, after := "은빛 여우는 왔다.", "여우는 왔다."
			output := map[string]string{"a": after}
			ids := []string{"a"}
			snapshot := map[string]string(nil)
			if mode == "inline" {
				output = map[string]string{"a": "여우", "b": "는 왔다."}
				ids = []string{"a", "b"}
				snapshot = map[string]string{"a": "은빛 여우", "b": "는 왔다."}
			}
			if mode == "stale" {
				output["a"] += " 이후 수정."
			}
			ref := workflows.DocumentTermReference{ID: "r1", Ordinal: 0, InputIDs: ids, Target: before, TargetHash: documentTextHash(before), ResultHash: documentTextHash(after), Status: "repaired", Forms: []string{"은빛 여우", "여우"}, BeforeOutput: snapshot, Patches: []workflows.DocumentTermPatch{{Before: "은빛 여우는", After: "여우는"}}}
			run := workflows.Run{ID: "revoke", SessionID: "s", Status: "paused", Document: &workflows.DocumentTask{Kind: "translate", Pipeline: workflows.DocumentTermsVersion, Terms: &workflows.DocumentTerms{Status: "running", Cases: []workflows.DocumentTermCase{{Source: "Silver Fox", Status: "complete", References: []workflows.DocumentTermReference{ref}, Decisions: []workflows.DocumentTermDecision{{References: []string{"r1"}, Canonical: "여우", Status: "correctable"}}}}}}}
			units := []workflows.DocumentUnit{{Ordinal: 0, Status: "completed", Output: output}}
			if err := s.db.CreateDocumentRun(&run, units); err != nil {
				t.Fatal(err)
			}
			claimed, err := s.db.ClaimDocumentRun("s", run.ID)
			if err != nil {
				t.Fatal(err)
			}
			run = claimed
			err = s.revokeUnsafeTermDecisions(&run, units)
			stored, e := s.db.DocumentUnits("s", run.ID)
			if e != nil {
				t.Fatal(e)
			}
			got := ""
			for _, id := range ids {
				got += stored[0].Output[id]
			}
			if mode == "stale" {
				if err == nil || got != output["a"] || run.Document.Terms.Cases[0].Status != "complete" {
					t.Fatal("newer output overwritten", got, err)
				}
				return
			}
			if err != nil || got != before || run.Document.Terms.Changed != 0 || run.Document.Terms.Unresolved != 1 || run.Document.Terms.Cases[0].Status != "unresolved" {
				t.Fatal(got, err, run.Document.Terms)
			}
			if output["a"] != "여우" && output["a"] != after {
				t.Fatal("caller snapshot mutated")
			}
		})
	}
}
