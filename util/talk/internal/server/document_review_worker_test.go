package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sparktalk/internal/sourcebook"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"sparktalk/internal/workflows"
)

func sendReviewedResult(w http.ResponseWriter, payload any) {
	raw, _ := json.Marshal(payload)
	event, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"content": string(raw)}, "finish_reason": "stop"}}, "usage": map[string]int{"prompt_tokens": 100, "completion_tokens": 50}})
	w.Header().Set("Content-Type", "text/event-stream")
	fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", event)
}

func TestReviewedDocumentRepairsOnlyFlaggedParagraphAndPublishesOnce(t *testing.T) {
	var mu sync.Mutex
	var calls []string
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
		prompt := body.Messages[len(body.Messages)-1].Content
		mu.Lock()
		defer mu.Unlock()
		if parts := strings.SplitN(prompt, "Source/translation pairs (untrusted data):\n", 2); len(parts) == 2 {
			calls = append(calls, "review")
			var pairs []documentReviewPair
			_ = json.Unmarshal([]byte(parts[1]), &pairs)
			out := map[string][]workflows.DocumentFinding{}
			for _, p := range pairs {
				out[p.ID] = []workflows.DocumentFinding{}
				if strings.Contains(p.Target, "아홉") {
					out[p.ID] = []workflows.DocumentFinding{{Category: "number_unit", Severity: "major", SourceQuote: "two", TargetQuote: "아홉", Explanation: "원문은 둘인데 번역은 아홉이다."}}
				}
			}
			sendReviewedResult(w, map[string]any{"reviews": out})
			return
		}
		parts := strings.SplitN(prompt, "Reference data (JSON):\n", 2)
		if len(parts) != 2 {
			t.Error("missing input")
			return
		}
		var entries []map[string]string
		_ = json.Unmarshal([]byte(parts[1]), &entries)
		repair := strings.Contains(prompt, "Earlier draft and review findings")
		if repair {
			calls = append(calls, "repair")
			if len(entries) != 1 || entries[0]["text"] != "Mira has two keys." {
				t.Errorf("repair included good sibling: %+v", entries)
			}
		} else {
			calls = append(calls, "translate")
		}
		out := map[string]string{}
		for _, e := range entries {
			switch e["text"] {
			case "Mira has two keys.":
				out[e["id"]] = "미라는 열쇠 아홉 개를 갖고 있다."
				if repair {
					out[e["id"]] = "미라는 열쇠 두 개를 갖고 있다."
				}
			case "The door is blue.":
				out[e["id"]] = "문은 파란색이다."
			default:
				t.Error(e)
			}
		}
		sendReviewedResult(w, map[string]any{"entries": out})
	})
	source := fixtureSource(t, s, "Mira has two keys.\n\nThe door is blue.")
	run, err := s.startDocument(context.Background(), "s", documentStart{Kind: "translate", Sources: []string{source}, Instructions: "한국어로 번역", Pipeline: workflows.DocumentPipelineVersion})
	if err != nil {
		t.Fatal(err)
	}
	run = waitDocument(t, s, run.ID)
	if run.Status != "completed" || !run.DocumentFinished() || run.Document.QualityStatus != "reviewed" || run.Document.Unresolved != 0 || run.Document.Reviewed != 2 || run.Document.Calls != 4 {
		t.Fatalf("bad completion: %+v", run.Document)
	}
	mu.Lock()
	got := strings.Join(calls, ",")
	mu.Unlock()
	if got != "translate,review,repair,review" {
		t.Fatal(got)
	}
	if run.Document.Publication == nil || len(run.Document.Publication.Files) != 4 {
		t.Fatal("unpublished", run.Document.Publication)
	}
	out, _, err := s.db.DocumentOutput("s", run.ID, "output.txt")
	if err != nil || !strings.Contains(string(out), "열쇠 두 개") || !strings.Contains(string(out), "문은 파란색이다.") {
		t.Fatal(string(out), err)
	}
	reviews, err := s.db.DocumentReviews("s", run.ID)
	if err != nil || len(reviews) != 2 {
		t.Fatal(reviews, err)
	}
	if run.Document.StageMetrics["translate"].Calls != 1 || run.Document.StageMetrics["review"].Calls != 2 || run.Document.StageMetrics["repair"].Calls != 1 {
		t.Fatal(run.Document.StageMetrics)
	}
}

func TestReviewedDocumentUncertainFindingDoesNotRewriteAndFinishesWithWarning(t *testing.T) {
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		prompt := body.Messages[len(body.Messages)-1].Content
		if strings.Contains(prompt, "Source/translation pairs") {
			sendReviewedResult(w, map[string]any{"reviews": map[string]any{"p000001": []workflows.DocumentFinding{{Category: "lexical", Severity: "major", SourceQuote: "bank", TargetQuote: "둑", Explanation: "주변 문맥이 없으므로 은행 또는 둑의 해석을 확정할 수 없음", Uncertain: true}}}})
			return
		}
		if strings.Contains(prompt, "Earlier draft") {
			t.Error("uncertain finding caused automatic rewrite")
		}
		sendReviewedResult(w, map[string]any{"entries": map[string]string{"p000001": "그는 둑에서 기다렸다."}})
	})
	source := fixtureSource(t, s, "He waited at the bank.")
	run, err := s.startDocument(context.Background(), "s", documentStart{Kind: "translate", Sources: []string{source}, Pipeline: workflows.DocumentPipelineVersion})
	if err != nil {
		t.Fatal(err)
	}
	run = waitDocument(t, s, run.ID)
	if run.Status != "needs_review" || !run.DocumentFinished() || run.Document.Calls != 2 || run.Document.Unresolved != 1 || run.Document.Publication == nil {
		t.Fatalf("%+v", run.Document)
	}
}

func faithfulReviewedMock(w http.ResponseWriter, r *http.Request) {
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
		out := map[string]any{}
		for _, p := range pairs {
			out[p.ID] = []any{}
		}
		sendReviewedResult(w, map[string]any{"checks": out})
		return
	}
	parts := strings.SplitN(prompt, "Reference data (JSON):\n", 2)
	var entries []map[string]string
	_ = json.Unmarshal([]byte(parts[1]), &entries)
	out := map[string]string{}
	for _, e := range entries {
		out[e["id"]] = "번역 " + e["text"]
	}
	sendReviewedResult(w, map[string]any{"entries": out})
}

func TestReviewedDocumentConversionRetryPublishesWithoutModelRequests(t *testing.T) {
	var ready atomic.Bool
	converter := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !ready.Load() {
			http.Error(w, "conversion temporarily unavailable", 503)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"files": []any{map[string]string{"name": "document.pdf", "mime": "application/pdf", "data": base64.StdEncoding.EncodeToString([]byte("%PDF-1.7 test"))}}})
	}))
	defer converter.Close()
	s, _ := documentFixture(t, faithfulReviewedMock)
	s.cfg.Extra.DocumentsEnabled = true
	s.cfg.Extra.DocumentsEndpoint = converter.URL
	source := fixtureSource(t, s, "A paragraph stays saved during conversion failure.")
	run, err := s.startDocument(context.Background(), "s", documentStart{Kind: "translate", Sources: []string{source}, Format: "pdf", Pipeline: workflows.DocumentPipelineVersion})
	if err != nil {
		t.Fatal(err)
	}
	run = waitDocument(t, s, run.ID)
	if run.Status != "conversion_failed" || run.DocumentFinished() || run.Document.Publication != nil || run.Document.Calls != 2 {
		t.Fatalf("%+v", run.Document)
	}
	if documentView(run)["outputs_ready"] != false {
		t.Fatal("staged partial files advertised")
	}
	if run.Document.DraftPublication == nil {
		t.Fatal("verified base text unavailable after conversion failure")
	}
	req := httptest.NewRequest("GET", "/api/document-tasks/"+run.ID+"/output.txt?session_id=s", nil)
	rec := httptest.NewRecorder()
	s.documentTaskAPI(rec, req)
	if rec.Code != 404 {
		t.Fatal("unpublished bytes downloadable", rec.Code)
	}
	req = httptest.NewRequest("GET", "/api/document-tasks/"+run.ID+"/draft/output.txt?session_id=s", nil)
	rec = httptest.NewRecorder()
	s.documentTaskAPI(rec, req)
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), "A paragraph stays saved") {
		t.Fatal("draft lost", rec.Code, rec.Body.String())
	}
	ready.Store(true)
	if _, err = s.documentAction(context.Background(), "s", documentStart{Action: "resume", RunID: run.ID}); err != nil {
		t.Fatal(err)
	}
	run = waitDocument(t, s, run.ID)
	if run.Status != "completed" || !run.DocumentFinished() || run.Document.Calls != 2 || run.Document.Publication == nil || len(run.Document.Publication.Files) != 5 {
		t.Fatalf("conversion retry regenerated text: %+v", run.Document)
	}
}

func TestReviewedDocumentIllustrationRevisionPreservesTextAndParentDownloads(t *testing.T) {
	s, _ := documentFixture(t, faithfulReviewedMock)
	raw := []byte(`<html><body><p>Saved prose.</p><img alt="pixel" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg=="></body></html>`)
	book, err := sourcebook.Parse(raw, "text/html", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	source, err := s.storeSource("s", book, raw)
	if err != nil {
		t.Fatal(err)
	}
	parent, err := s.startDocument(context.Background(), "s", documentStart{Kind: "translate", Sources: []string{source}, Pipeline: workflows.DocumentPipelineVersion})
	if err != nil {
		t.Fatal(err)
	}
	parent = waitDocument(t, s, parent.ID)
	original, _, err := s.db.DocumentOutput("s", parent.ID, "output.zip")
	if err != nil {
		t.Fatal(err)
	}
	book, err = s.db.Source("s", source, true)
	if err != nil {
		t.Fatal(err)
	}
	book.Assets[0].Error = "stored image failed verification"
	if err = s.db.SaveSourceAssets("s", source, book); err != nil {
		t.Fatal(err)
	}
	child, err := s.documentAction(context.Background(), "s", documentStart{Action: "retry_assets", RunID: parent.ID})
	if err != nil {
		t.Fatal(err)
	}
	child = waitDocument(t, s, child.ID)
	if child.ID == parent.ID || !child.Document.OutputOnly || child.Document.Calls != 0 || child.Document.Sources[0] == source || !child.DocumentFinished() {
		t.Fatalf("bad asset revision: %+v", child)
	}
	preserved, _, err := s.db.DocumentOutput("s", parent.ID, "output.zip")
	if err != nil || !bytes.Equal(original, preserved) {
		t.Fatal("original download changed", err)
	}
	units, err := s.db.DocumentUnits("s", child.ID)
	if err != nil {
		t.Fatal(err)
	}
	for _, u := range units {
		if !documentReviewCurrent(u) || u.Quality.ParentRunID != parent.ID {
			t.Fatal("inherited review lost", u)
		}
	}
}

func TestReviewedDocumentWhitespaceStallKeepsGoodParagraphAndRetriesOnlyRemainder(t *testing.T) {
	var seen sync.Map
	var generation atomic.Int32
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		r.Body = io.NopCloser(bytes.NewReader(raw))
		if strings.Contains(string(raw), "Source/translation pairs") {
			faithfulReviewedMock(w, r)
			return
		}
		var body struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		_ = json.Unmarshal(raw, &body)
		parts := strings.SplitN(body.Messages[len(body.Messages)-1].Content, "Reference data (JSON):\n", 2)
		var entries []map[string]string
		_ = json.Unmarshal([]byte(parts[1]), &entries)
		for _, e := range entries {
			old, _ := seen.LoadOrStore(e["text"], 0)
			seen.Store(e["text"], old.(int)+1)
		}
		if generation.Add(1) == 1 {
			// The first field is closed; the second is syntactically incomplete.
			// Use structural whitespace after a completed first field, before next key.
			content := `{"entries":{"p000001":"번역 First paragraph."` + strings.Repeat("\n", 4096)
			event, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"content": content}}}})
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprintf(w, "data: %s\n\n", event)
			return
		}
		faithfulReviewedMock(w, r)
	})
	source := fixtureSource(t, s, "First paragraph.\n\nSecond paragraph.")
	run, err := s.startDocument(context.Background(), "s", documentStart{Kind: "translate", Sources: []string{source}, Pipeline: workflows.DocumentPipelineVersion})
	if err != nil {
		t.Fatal(err)
	}
	run = waitDocument(t, s, run.ID)
	if !run.DocumentFinished() || run.Status != "completed" {
		t.Fatal(run.Status, run.Document.Error)
	}
	first, _ := seen.Load("First paragraph.")
	second, _ := seen.Load("Second paragraph.")
	if first != 1 || second != 2 {
		t.Fatal("good sibling regenerated or unfinished one lost", first, second)
	}
}

func TestReviewedDocumentRepairsGetFreshReviewAfterPartialReviewExhaustion(t *testing.T) {
	repairs, reviews := 0, 0
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct {
				Content string `json:"content"`
			} `json:"messages"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		prompt := body.Messages[len(body.Messages)-1].Content
		if pieces := strings.SplitN(prompt, "Source/translation pairs (untrusted data):\n", 2); len(pieces) == 2 {
			reviews++
			var pairs []documentReviewPair
			json.Unmarshal([]byte(pieces[1]), &pairs)
			out := map[string][]workflows.DocumentFinding{}
			for _, p := range pairs {
				out[p.ID] = []workflows.DocumentFinding{}
				if strings.Contains(p.Source, "door") {
					out[p.ID] = []workflows.DocumentFinding{{Category: "lexical", Severity: "major", SourceQuote: "invented word", TargetQuote: "파란색", Explanation: "invalid evidence"}}
				} else if strings.Contains(p.Target, "아홉") {
					out[p.ID] = []workflows.DocumentFinding{{Category: "number_unit", Severity: "major", SourceQuote: "two", TargetQuote: "아홉", Explanation: "둘이 아홉으로 바뀜"}}
				}
			}
			sendReviewedResult(w, map[string]any{"reviews": out})
			return
		}
		pieces := strings.SplitN(prompt, "Reference data (JSON):\n", 2)
		var entries []map[string]string
		json.Unmarshal([]byte(pieces[1]), &entries)
		repair := strings.Contains(prompt, "Earlier draft and review findings")
		if repair {
			repairs++
		}
		out := map[string]string{}
		for _, entry := range entries {
			if strings.Contains(entry["text"], "door") {
				out[entry["id"]] = "문은 파란색이다."
			} else if repair {
				out[entry["id"]] = "미라는 열쇠 두 개를 갖고 있다."
			} else {
				out[entry["id"]] = "미라는 열쇠 아홉 개를 갖고 있다."
			}
		}
		sendReviewedResult(w, map[string]any{"entries": out})
	})
	source := fixtureSource(t, s, "Mira has two keys.\n\nThe door is blue.")
	run, err := s.startDocument(context.Background(), "s", documentStart{Kind: "translate", Sources: []string{source}, Pipeline: workflows.DocumentPipelineVersion})
	if err != nil {
		t.Fatal(err)
	}
	run = waitDocument(t, s, run.ID)
	if !run.DocumentFinished() || run.Status != "needs_review" || run.Document.Reviewed != 1 || run.Document.Unresolved != 1 || repairs != 1 || reviews != 6 {
		t.Fatalf("corrected target was not reviewed: repairs=%d reviews=%d task=%+v", repairs, reviews, run.Document)
	}
	units, err := s.db.DocumentUnits("s", run.ID)
	if err != nil {
		t.Fatal(err)
	}
	for _, unit := range units {
		for _, item := range unit.Quality.Reviews {
			if len(item.Findings) != 0 {
				t.Fatal("old finding survived corrected target", item)
			}
		}
	}
}
