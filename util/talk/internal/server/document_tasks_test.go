package server

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/knowledge"
	"sparktalk/internal/llm"
	"sparktalk/internal/media"
	"sparktalk/internal/sourcebook"
	"sparktalk/internal/workflows"
)

func documentFixture(t *testing.T, handler http.HandlerFunc) (*Server, *httptest.Server) {
	t.Helper()
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, `{"data":[{"id":"model"}]}`)
			return
		}
		handler(w, r)
	}))
	t.Cleanup(backend.Close)
	database, e := db.Open(filepath.Join(t.TempDir(), "test.db"))
	if e != nil {
		t.Fatal(e)
	}
	if _, e = database.CreateSession("s", "test", "model", "off"); e != nil {
		t.Fatal(e)
	}
	s := &Server{db: database, cfg: config.Config{Model: config.ModelConfig{DefaultModel: "model", Endpoint: backend.URL, ModelType: "generic"}}, llm: llm.New(backend.URL, "model", "")}
	t.Cleanup(func() {
		s.tasks.Stop()
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		if e := s.tasks.Wait(ctx); e != nil {
			t.Error(e)
		}
		database.Close()
	})
	return s, backend
}
func mockDocumentResponse(w http.ResponseWriter, r *http.Request, translate bool) {
	var input map[string]any
	json.NewDecoder(r.Body).Decode(&input)
	messages := input["messages"].([]any)
	text := messages[len(messages)-1].(map[string]any)["content"].(string)
	raw := strings.SplitN(text, "Reference data (JSON):\n", 2)
	var entries []map[string]string
	if len(raw) == 2 {
		json.Unmarshal([]byte(raw[1]), &entries)
	}
	output := []map[string]string{{"id": "summary", "text": "검증된 자료 요약"}}
	if translate {
		output = nil
		for _, x := range entries {
			output = append(output, map[string]string{"id": x["id"], "text": "번역 " + x["text"]})
		}
	}
	args, _ := json.Marshal(map[string]any{"entries": output})
	event := map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"tool_calls": []any{map[string]any{"index": 0, "id": "result", "type": "function", "function": map[string]any{"name": "document_result", "arguments": string(args)}}}}, "finish_reason": "tool_calls"}}, "usage": map[string]int{"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700}}
	w.Header().Set("Content-Type", "text/event-stream")
	v, _ := json.Marshal(event)
	fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", v)
}
func fixtureSource(t *testing.T, s *Server, body string) string {
	t.Helper()
	b, e := sourcebook.Parse([]byte(body), "text/plain", "https://example.com/book", nil)
	if e != nil {
		t.Fatal(e)
	}
	id, e := s.storeSource("s", b, []byte(body))
	if e != nil {
		t.Fatal(e)
	}
	return id
}
func waitDocument(t *testing.T, s *Server, id string) workflows.Run {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		run, e := s.db.DocumentRun("s", id)
		if e != nil {
			t.Fatal(e)
		}
		if run.Status != "running" {
			return run
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("document task did not finish")
	return workflows.Run{}
}
func TestDocumentTaskCompletesAllChunksAndExports(t *testing.T) {
	var calls atomic.Int32
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) { calls.Add(1); mockDocumentResponse(w, r, true) })
	body := strings.Repeat("CHAPTER I.\n\nThis is a complete test paragraph with enough text to cross the original preview limit.\n\n", 70) + "THE_FINAL_SENTENCE"
	id := fixtureSource(t, s, body)
	ctx, cancel := context.WithCancel(context.Background())
	run, e := s.documentAction(ctx, "s", documentStart{Action: "start", Kind: "translate", Sources: []string{id}})
	if e != nil {
		t.Fatal(e)
	}
	cancel() // browser/request disconnect does not cancel a task
	run = waitDocument(t, s, run.ID)
	if run.Status != "completed" || run.Document.Completed != run.Document.Total || calls.Load() < 2 {
		t.Fatalf("incomplete: %+v", run.Document)
	}
	data, _, e := s.db.DocumentOutput("s", run.ID, "output.txt")
	if e != nil || !bytes.Contains(data, []byte("THE_FINAL_SENTENCE")) {
		t.Fatal("tail lost", e)
	}
	if _, _, e = s.db.DocumentOutput("other", run.ID, "output.txt"); e == nil {
		t.Fatal("cross-session output exposed")
	}
	if e = s.db.DeleteSource("s", id); e == nil {
		t.Fatal("live run source deleted")
	}
}
func TestDocumentTaskMissingIDsNeverCompletes(t *testing.T) {
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"{\\\"entries\\\":[]}\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	})
	id := fixtureSource(t, s, "A complete paragraph.")
	run, e := s.documentAction(context.Background(), "s", documentStart{Action: "start", Kind: "translate", Sources: []string{id}})
	if e != nil {
		t.Fatal(e)
	}
	run = waitDocument(t, s, run.ID)
	if run.Status != "needs_review" || run.Document.Calls != 3 {
		t.Fatalf("false completion: %+v", run)
	}
	if _, _, e = s.db.DocumentOutput("s", run.ID, "output.txt"); e == nil {
		t.Fatal("unverified result exported")
	}
}
func TestDocumentTaskSummaryAndComparisonReduce(t *testing.T) {
	for _, kind := range []string{"summarize", "compare"} {
		t.Run(kind, func(t *testing.T) {
			var calls atomic.Int32
			s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) { calls.Add(1); mockDocumentResponse(w, r, false) })
			a := fixtureSource(t, s, strings.Repeat("A long source paragraph.\n\n", 350))
			ids := []string{a}
			if kind == "compare" {
				ids = append(ids, fixtureSource(t, s, "Another source with a different finding."))
			}
			run, e := s.documentAction(context.Background(), "s", documentStart{Action: "start", Kind: kind, Sources: ids, Instructions: "차이와 근거를 설명해라"})
			if e != nil {
				t.Fatal(e)
			}
			run = waitDocument(t, s, run.ID)
			if run.Status != "completed" || calls.Load() > 15 {
				t.Fatalf("reduction failed: %s calls=%d %s", run.Status, calls.Load(), run.Document.Error)
			}
			data, _, e := s.db.DocumentOutput("s", run.ID, "output.txt")
			if e != nil || !bytes.Contains(data, []byte("요약")) {
				t.Fatal(e)
			}
		})
	}
}
func TestDocumentTaskPauseResumePreservesCompletedUnits(t *testing.T) {
	var calls atomic.Int32
	entered := make(chan struct{}, 10)
	gate := make(chan struct{})
	var block atomic.Bool
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if block.Load() {
			entered <- struct{}{}
			select {
			case <-gate:
			case <-r.Context().Done():
				return
			}
		}
		mockDocumentResponse(w, r, true)
	})
	id := fixtureSource(t, s, strings.Repeat("A paragraph with enough content.\n\n", 200))
	run, e := s.documentAction(context.Background(), "s", documentStart{Action: "start", Kind: "translate", Sources: []string{id}})
	if e != nil {
		t.Fatal(e)
	}
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		r, _ := s.db.DocumentRun("s", run.ID)
		if r.Document.Completed > 0 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	block.Store(true)
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("second request not entered")
	}
	paused, e := s.documentAction(context.Background(), "s", documentStart{Action: "pause", RunID: run.ID})
	if e != nil || paused.Status != "paused" || paused.Document.Completed < 1 {
		t.Fatal("pause lost progress", e)
	}
	completed := paused.Document.Completed
	block.Store(false)
	close(gate)
	if _, e = s.documentAction(context.Background(), "s", documentStart{Action: "resume", RunID: run.ID, Instructions: "changed"}); e == nil {
		t.Fatal("mixed instructions accepted")
	}
	if _, e = s.documentAction(context.Background(), "other", documentStart{Action: "resume", RunID: run.ID}); e == nil {
		t.Fatal("cross-session resume")
	}
	if _, e = s.documentAction(context.Background(), "s", documentStart{Action: "resume", RunID: run.ID}); e != nil {
		t.Fatal(e)
	}
	result := waitDocument(t, s, run.ID)
	if result.Status != "completed" || result.Document.Completed < completed {
		t.Fatal("resume failed", result.Document.Error)
	}
	units, _ := s.db.DocumentUnits("s", run.ID)
	if len(units[0].Attempts) != 1 {
		t.Fatal("completed first unit repeated")
	}
}
func TestDocumentSplitCoverageAndOutputValidation(t *testing.T) {
	u := workflows.DocumentUnit{Stage: "process", Inputs: []workflows.DocumentInput{{ID: "s:b:0", Source: "s", Block: "b", Text: strings.Repeat("가", 400)}}}
	other, ok := splitDocumentUnit(&u)
	if !ok || u.Inputs[0].Text+other.Inputs[0].Text != strings.Repeat("가", 400) || other.Inputs[0].Part != 200 {
		t.Fatal("split lost source")
	}
	for _, raw := range []string{`{"entries":[]}`, `{"entries":[{"id":"other","text":"wrong"}]}`, `{"entries":[{"id":"s:b:0","text":"x"},{"id":"s:b:0","text":"x"}]}`} {
		if _, e := validateDocumentOutput(u, "translate", llm.StreamResult{Content: raw}); e == nil {
			t.Fatal("invalid coverage accepted", raw)
		}
	}
}

func TestDocumentExportRejectsMissingOriginalBlock(t *testing.T) {
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) { mockDocumentResponse(w, r, true) })
	id := fixtureSource(t, s, "First paragraph.\n\nSecond paragraph.")
	b, _ := s.db.Source("s", id, false)
	run := workflows.Run{ID: "fake", SessionID: "s", Status: "running", Document: &workflows.DocumentTask{Kind: "translate", Sources: []string{id}}, Steps: []workflows.StepState{{}, {}}}
	if e := s.db.SaveWorkflowRun(&run); e != nil {
		t.Fatal(e)
	}
	in := workflows.DocumentInput{ID: "first", Source: id, Block: b.Blocks[0].ID, Text: b.Blocks[0].Text}
	unit := workflows.DocumentUnit{Status: "completed", Inputs: []workflows.DocumentInput{in}, Output: map[string]string{"first": "첫 문단"}}
	if e := s.exportDocument(context.Background(), run, []workflows.DocumentUnit{unit}); e == nil {
		t.Fatal("missing original block was silently treated as a selected range")
	}
}

func TestDocumentConversionReadsStoredResultsWithoutLLM(t *testing.T) {
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) { t.Error("conversion must not call the LLM") })
	var rendererCalls atomic.Int32
	renderer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rendererCalls.Add(1)
		var v struct {
			Text string `json:"text"`
		}
		json.NewDecoder(r.Body).Decode(&v)
		if r.URL.Path != "/v1/documents/text" || v.Text != "저장된 번역 결과" {
			t.Error("stored text not sent to renderer")
		}
		fmt.Fprint(w, `{"files":[{"name":"document.pdf","mime":"application/pdf","data":"JVBERi0xLjQK"}]}`)
	}))
	defer renderer.Close()
	s.cfg.Extra.DocumentsEnabled = true
	s.cfg.Extra.DocumentsEndpoint = renderer.URL
	run := workflows.Run{ID: "convert", SessionID: "s", Status: "completed", Current: 2, Document: &workflows.DocumentTask{Kind: "translate"}}
	if e := s.db.SaveWorkflowRun(&run); e != nil {
		t.Fatal(e)
	}
	if e := s.db.SaveDocumentOutput("s", run.ID, "output.txt", "text/plain", []byte("저장된 번역 결과")); e != nil {
		t.Fatal(e)
	}
	for i := 0; i < 2; i++ {
		if _, e := s.documentAction(context.Background(), "s", documentStart{Action: "export", RunID: run.ID, Format: "pdf"}); e != nil {
			t.Fatal(e)
		}
	}
	if rendererCalls.Load() != 1 {
		t.Fatal("cached conversion was rendered again")
	}
	data, _, e := s.db.DocumentOutput("s", run.ID, "output.pdf")
	if e != nil || !bytes.HasPrefix(data, []byte("%PDF")) {
		t.Fatal("converted result not stored", e)
	}
}

func TestDocumentTaskHandoffDoesNotPollWithModel(t *testing.T) {
	var id string
	var calls atomic.Int32
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		args, _ := json.Marshal(map[string]any{"action": "start", "kind": "translate", "sources": []string{id}})
		event, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"tool_calls": []any{map[string]any{"index": 0, "id": "start", "type": "function", "function": map[string]string{"name": "document_task", "arguments": string(args)}}}}, "finish_reason": "tool_calls"}}})
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", event)
	})
	s.cfg.Model.ReasoningEffort = "xhigh" // The current chat turn selects off.
	id = fixtureSource(t, s, "A paragraph.")
	cfg := config.ToolsConfig{Enabled: true, MaxRounds: 4, Timeout: "1s"}
	ctx, release, e := s.trackGeneration(context.Background())
	if e != nil {
		t.Fatal(e)
	}
	defer release()
	result, e := runCompletionLoopForSession(s, "s", ctx, s.llm, []llm.Message{{Role: "user", Content: "translate whole source"}}, "model", "off", "", cfg, true, func(string, any) error { return nil })
	if e != nil || calls.Load() != 1 || !strings.Contains(result.Content, "번역 작업을 시작") {
		t.Fatalf("model kept polling: %d %s %v", calls.Load(), result.Content, e)
	}
	runs, _ := s.db.WorkflowRuns("s")
	if len(runs) != 1 {
		t.Fatal("task not persisted")
	}
	if runs[0].Document.Effort != "none" {
		t.Fatal("document inherited the global setting instead of this chat turn", runs[0].Document.Effort)
	}
	if _, e = s.documentAction(context.Background(), "s", documentStart{Action: "pause", RunID: runs[0].ID}); e != nil {
		t.Fatal(e)
	}
}

func TestDocumentStartIdempotencyPreservesOneRun(t *testing.T) {
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) { mockDocumentResponse(w, r, true) })
	id := fixtureSource(t, s, "One complete paragraph.")
	ctx, release, e := s.trackGeneration(context.Background())
	if e != nil {
		t.Fatal(e)
	}
	defer release()
	a := documentStart{Action: "start", Kind: "translate", Sources: []string{id}, Key: "same-request"}
	first, e := s.documentAction(ctx, "s", a)
	if e != nil {
		t.Fatal(e)
	}
	second, e := s.documentAction(ctx, "s", a)
	if e != nil || second.ID != first.ID {
		t.Fatal("request replay created a second run", e)
	}
	a.Instructions = "different"
	if _, e = s.documentAction(ctx, "s", a); e == nil {
		t.Fatal("request key accepted changed inputs")
	}
	runs, _ := s.db.WorkflowRuns("s")
	if len(runs) != 1 {
		t.Fatal("duplicate task persisted")
	}
	s.documentAction(context.Background(), "s", documentStart{Action: "pause", RunID: first.ID})
}

func TestFailedIllustrationRetryDoesNotRepeatTranslation(t *testing.T) {
	var calls atomic.Int32
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) { calls.Add(1); mockDocumentResponse(w, r, true) })
	png := base64.StdEncoding.EncodeToString(onePixelPNG)
	b, e := sourcebook.Parse([]byte(`<p>Rabbit story.</p><img src="data:image/png;base64,`+png+`" alt="Rabbit">`), "text/html", "", nil)
	if e != nil {
		t.Fatal(e)
	}
	b.Assets[0].Data = nil
	b.Assets[0].Error = "transient failure"
	b.Assets[0].Pending = false
	id, e := s.storeSource("s", b, []byte("fixture"))
	if e != nil {
		t.Fatal(e)
	}
	run, e := s.documentAction(context.Background(), "s", documentStart{Action: "start", Kind: "translate", Sources: []string{id}})
	if e != nil {
		t.Fatal(e)
	}
	run = waitDocument(t, s, run.ID)
	if run.Status != "needs_review" || run.Document.ImagesReady != 0 {
		t.Fatal("missing image falsely complete")
	}
	count := calls.Load()
	if _, e = s.documentAction(context.Background(), "s", documentStart{Action: "retry_assets", RunID: run.ID}); e != nil {
		t.Fatal(e)
	}
	run = waitDocument(t, s, run.ID)
	if run.Status != "completed" || run.Document.ImagesReady != 1 || calls.Load() != count {
		t.Fatalf("image retry repeated text/failed: %s %d %d", run.Status, calls.Load(), count)
	}
}

func TestDocumentShortOutputRetriesOnlyFailedFragments(t *testing.T) {
	var calls, goodSeen atomic.Int32
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) {
		n := calls.Add(1)
		var body struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			return
		}
		var entries []map[string]string
		json.Unmarshal([]byte(strings.SplitN(body.Messages[len(body.Messages)-1].Content, "Reference data (JSON):\n", 2)[1]), &entries)
		output := map[string]string{}
		for _, entry := range entries {
			text := entry["text"]
			if strings.Contains(text, "GOOD_FRAGMENT") {
				goodSeen.Add(1)
			}
			output[entry["id"]] = "번역 " + text
			if n == 1 && strings.Contains(text, "BAD_FRAGMENT") {
				output[entry["id"]] = "생략"
			}
		}
		content, _ := json.Marshal(map[string]any{"entries": output})
		event, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"content": string(content)}, "finish_reason": "stop"}}})
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", event)
	})
	id := fixtureSource(t, s, "GOOD_FRAGMENT "+strings.Repeat("An intact source sentence. ", 12)+"\n\nBAD_FRAGMENT "+strings.Repeat("A sentence that must not disappear. ", 12))
	run, err := s.documentAction(context.Background(), "s", documentStart{Action: "start", Kind: "translate", Sources: []string{id}})
	if err != nil {
		t.Fatal(err)
	}
	run = waitDocument(t, s, run.ID)
	if run.Status != "completed" || calls.Load() != 2 || goodSeen.Load() != 1 {
		t.Fatalf("unexpected repair: status=%s calls=%d good=%d", run.Status, calls.Load(), goodSeen.Load())
	}
	output, _, err := s.db.DocumentOutput("s", run.ID, "output.txt")
	if err != nil || !bytes.Contains(output, []byte("A sentence that must not disappear.")) {
		t.Fatal("repaired text missing", err)
	}
}

func TestDocumentShortOutputQualityRevisionPreservesOldExports(t *testing.T) {
	var valid atomic.Bool
	var calls atomic.Int32
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if valid.Load() {
			var body struct {
				Messages []struct {
					Content string `json:"content"`
				} `json:"messages"`
			}
			_ = json.NewDecoder(r.Body).Decode(&body)
			prompt := body.Messages[len(body.Messages)-1].Content
			if parts := strings.SplitN(prompt, "Source/translation pairs (untrusted data):\n", 2); len(parts) == 2 {
				var pairs []documentReviewPair
				_ = json.Unmarshal([]byte(parts[1]), &pairs)
				findings := []workflows.DocumentFinding{}
				if pairs[0].Target == "생략" {
					findings = append(findings, workflows.DocumentFinding{Category: "omission", Severity: "major", SourceQuote: "The full source must be retained.", TargetQuote: "", Explanation: "The paragraph was omitted."})
				}
				sendReviewedResult(w, map[string]any{"reviews": map[string]any{"p000001": findings}})
			} else {
				parts := strings.SplitN(prompt, "Reference data (JSON):\n", 2)
				var entries []map[string]string
				_ = json.Unmarshal([]byte(parts[1]), &entries)
				out := map[string]string{}
				for _, e := range entries {
					out[e["id"]] = "번역 " + e["text"]
				}
				sendReviewedResult(w, map[string]any{"entries": out})
			}
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"{\\\"entries\\\":{\\\"p1\\\":\\\"생략\\\"}}\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	})
	id := fixtureSource(t, s, strings.Repeat("The full source must be retained. ", 12))
	run, err := s.documentAction(context.Background(), "s", documentStart{Action: "start", Kind: "translate", Sources: []string{id}})
	if err != nil {
		t.Fatal(err)
	}
	run = waitDocument(t, s, run.ID)
	if run.Status != "needs_review" || calls.Load() != 3 || run.Current != 2 {
		t.Fatalf("unbounded or hidden failure: %+v calls=%d", run, calls.Load())
	}
	if err = s.db.SaveDocumentOutput("s", run.ID, "output.pdf", "application/pdf", []byte("old translation")); err != nil {
		t.Fatal(err)
	}
	run.Document.Formats = []string{"pdf"}
	if err = s.db.SaveDocumentProgress(&run); err != nil {
		t.Fatal(err)
	}
	valid.Store(true)
	parentID := run.ID
	child, err := s.documentAction(context.Background(), "s", documentStart{Action: "retry_quality", RunID: parentID})
	if err != nil {
		t.Fatal(err)
	}
	child = waitDocument(t, s, child.ID)
	if child.ID == parentID || child.Document.Parent != parentID || child.Status != "completed" || calls.Load() != 4 {
		t.Fatalf("bad revision: %+v calls=%d", child, calls.Load())
	}
	if child.Document.Pipeline != "" || child.Document.Profiles != nil {
		t.Fatal("legacy missing-fragment retry silently enabled semantic review")
	}
	old, _, err := s.db.DocumentOutput("s", parentID, "output.pdf")
	if err != nil || string(old) != "old translation" {
		t.Fatal("parent download destroyed", err)
	}
	if _, _, err = s.db.DocumentOutput("s", child.ID, "output.pdf"); err == nil {
		t.Fatal("parent PDF advertised as new text")
	}
	parent, err := s.db.DocumentRun("s", parentID)
	if err != nil || parent.Status != "needs_review" || len(parent.Document.Formats) != 1 || parent.Document.Formats[0] != "pdf" {
		t.Fatal("parent state changed", err)
	}
}

func TestDocumentRequestedConversionCanRecoverWithoutRetranslation(t *testing.T) {
	var available atomic.Bool
	converter := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !available.Load() {
			http.Error(w, "temporarily offline", 503)
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"files": []any{map[string]string{"name": "document.pdf", "mime": "application/pdf", "data": base64.StdEncoding.EncodeToString([]byte("%PDF-1.7\nconverted"))}}})
	}))
	defer converter.Close()
	var calls atomic.Int32
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) { calls.Add(1); mockDocumentResponse(w, r, true) })
	s.cfg.Extra.DocumentsEnabled = true
	s.cfg.Extra.DocumentsEndpoint = converter.URL
	id := fixtureSource(t, s, "A paragraph to translate and then convert.")
	run, err := s.documentAction(context.Background(), "s", documentStart{Action: "start", Kind: "translate", Sources: []string{id}, Format: "pdf"})
	if err != nil {
		t.Fatal(err)
	}
	run = waitDocument(t, s, run.ID)
	if run.Status != "needs_review" || run.Current != 2 {
		t.Fatalf("base files lost on conversion failure: %+v", run)
	}
	available.Store(true)
	run, err = s.documentAction(context.Background(), "s", documentStart{Action: "export", RunID: run.ID, Format: "pdf"})
	if err != nil {
		t.Fatal(err)
	}
	if run.Status != "completed" || len(run.Document.Warnings) != 0 || calls.Load() != 1 {
		t.Fatalf("conversion did not recover: %+v calls=%d", run, calls.Load())
	}
	view := documentView(run)
	outputs := view["outputs"].(map[string]string)
	for _, format := range []string{"txt", "md", "html", "zip", "pdf"} {
		if outputs[format] == "" {
			t.Fatalf("output link disappeared: %s", format)
		}
	}
}

func TestDocumentBudgetRenewalPreservesProgress(t *testing.T) {
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) { mockDocumentResponse(w, r, true) })
	id := fixtureSource(t, s, "A completed paragraph remains intact.")
	run, err := s.documentAction(context.Background(), "s", documentStart{Action: "start", Kind: "translate", Sources: []string{id}})
	if err != nil {
		t.Fatal(err)
	}
	run = waitDocument(t, s, run.ID)
	run.Status = "paused"
	run.Current = 0
	run.Document.Error = "task execution budget reached"
	run.Document.MaxCalls = run.Document.Calls
	if err = s.db.SaveDocumentProgress(&run); err != nil {
		t.Fatal(err)
	}
	if _, err = s.documentAction(context.Background(), "s", documentStart{Action: "resume", RunID: run.ID}); err == nil {
		t.Fatal("budget silently renewed")
	}
	calls := run.Document.Calls
	if _, err = s.documentAction(context.Background(), "s", documentStart{Action: "resume", RunID: run.ID, ExtendBudget: true}); err != nil {
		t.Fatal(err)
	}
	run = waitDocument(t, s, run.ID)
	if run.Status != "completed" || run.Document.Calls != calls || run.Document.MaxCalls != calls+2048 {
		t.Fatalf("budget renewal lost progress: %+v", run)
	}
}

func TestDocumentPartialSingleBlockExcludesOutsideIllustration(t *testing.T) {
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) { mockDocumentResponse(w, r, true) })
	raw := []byte(`<html><body><img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg=="><p>A single paragraph containing the selected middle range.</p></body></html>`)
	book, err := sourcebook.Parse(raw, "text/html", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	id, err := s.storeSource("s", book, raw)
	if err != nil {
		t.Fatal(err)
	}
	run, err := s.documentAction(context.Background(), "s", documentStart{Action: "start", Kind: "translate", Sources: []string{id}, Start: 9, End: 30})
	if err != nil {
		t.Fatal(err)
	}
	run = waitDocument(t, s, run.ID)
	if run.Status != "completed" || run.Document.ImageTotal != 0 {
		t.Fatalf("outside illustration retained: %+v", run)
	}
	data, _, err := s.db.DocumentOutput("s", run.ID, "output.html")
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(data, []byte("<img")) {
		t.Fatal("outside image in selected export")
	}
}

func TestLegacyArchiveListingCannotBecomeWholeDocument(t *testing.T) {
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) { t.Error("incomplete source reached model") })
	store, err := media.New(filepath.Join(t.TempDir(), "media"))
	if err != nil {
		t.Fatal(err)
	}
	s.media = store
	s.knowledgeIndex = &knowledge.Extractor{}
	var raw bytes.Buffer
	z := zip.NewWriter(&raw)
	file, err := z.Create("README.txt")
	if err != nil {
		t.Fatal(err)
	}
	file.Write([]byte("This is a project archive listing, not an HTML book."))
	if err = z.Close(); err != nil {
		t.Fatal(err)
	}
	item, err := store.SaveReader(bytes.NewReader(raw.Bytes()), "project.zip", "application/zip", 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	id, book, err := s.attachmentSource(context.Background(), "s", item)
	if err != nil {
		t.Fatal(err)
	}
	if !book.Incomplete {
		t.Fatal("legacy archive listing certified as complete source")
	}
	if _, err = s.documentAction(context.Background(), "s", documentStart{Action: "start", Kind: "translate", Sources: []string{id}}); err == nil {
		t.Fatal("incomplete legacy source admitted")
	}
}

func TestDocumentTaskPublishesReasoningBeforeUnitCompletes(t *testing.T) {
	gate := make(chan struct{})
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for i := 0; i < 2; i++ {
			if i == 1 {
				select {
				case <-r.Context().Done():
					return
				case <-time.After(2100 * time.Millisecond):
				}
			}
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"검토 중\"}}]}\n\n")
			w.(http.Flusher).Flush()
		}
		select {
		case <-gate:
		case <-r.Context().Done():
			return
		}
		mockDocumentResponse(w, r, true)
	})
	defer close(gate)
	id := fixtureSource(t, s, "A complete paragraph.")
	run, err := s.documentAction(context.Background(), "s", documentStart{Action: "start", Kind: "translate", Sources: []string{id}})
	if err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		saved, err := s.db.DocumentRun("s", run.ID)
		if err != nil {
			t.Fatal(err)
		}
		if saved.Document.ReasoningChars > 0 {
			if saved.Status != "running" || saved.Document.Activity != "reasoning" || saved.Document.ContentChars != 0 || saved.Document.ElapsedSeconds < 2 || saved.Document.Completed != 0 {
				t.Fatalf("wrong live progress: %+v", saved.Document)
			}
			if _, err = s.documentAction(context.Background(), "s", documentStart{Action: "cancel", RunID: run.ID}); err != nil {
				t.Fatal(err)
			}
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatal("reasoning stayed invisible until the unit ended")
}
