package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"sparktalk/internal/llm"
	"sparktalk/internal/sourcebook"
	"sparktalk/internal/workflows"
)

func TestDocumentPlainWireCoalescesRubyButProtectsMarkup(t *testing.T) {
	raw := `<html><body><p>A <ruby><rb>漢字</rb><rp>(</rp><rt>かんじ</rt><rp>)</rp></ruby> word.</p><p>Keep <em>emphasis</em> here.</p><p>A <a href="#note">note</a> reference.</p><p><span>White</span> <span>space</span>.</p><p id="note">Footnote.</p><table><tr><td>First cell</td><td>Second cell</td></tr></table><p>Verse one<br>Verse two</p></body></html>`
	b, err := sourcebook.Parse([]byte(raw), "text/html", "https://example.org/book", nil)
	if err != nil {
		t.Fatal(err)
	}
	u := workflows.DocumentUnit{}
	for _, block := range b.Blocks {
		u.Inputs = append(u.Inputs, workflows.DocumentInput{ID: "source:" + block.ID + ":0", Source: "source", Block: block.ID, Text: block.Text})
	}
	units := []workflows.DocumentUnit{u}
	decorateDocumentInputs(units, "source", b)
	u = units[0]
	wire := documentWireEntries(u)
	if wire[0].Text != "A 漢字 word." || !strings.Contains(wire[0].Readings, "かんじ") || len(wire[0].Indexes) != 3 {
		t.Fatalf("plain ruby paragraph not consolidated: %+v", wire[0])
	}
	result := map[string]string{}
	for _, entry := range wire {
		result[entry.ID] = "번역 " + entry.Text
	}
	data, _ := json.Marshal(map[string]any{"entries": result})
	out, err := validateDocumentOutput(u, "translate", llm.StreamResult{Content: string(data)})
	if err != nil || len(out) != len(u.Inputs) {
		t.Fatal("original IDs not restored", err)
	}
	translated := map[string]string{}
	for _, in := range u.Inputs {
		translated[in.Block] = out[in.ID]
	}
	page, err := b.HTML(translated, true)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{`<em>번역 emphasis</em>`, `<a href="#note">번역 note</a>`, `White space`, `<td>번역 First cell</td>`, `<td>번역 Second cell</td>`, `<br>번역 Verse two`} {
		if !strings.Contains(string(page), want) {
			t.Fatalf("lost protected structure %q: %s", want, page)
		}
	}
	if strings.Contains(string(page), "かんじ") {
		t.Fatal("reading was translated as prose")
	}
}

func TestDocumentBatchCountsWireEntriesNotRubyAnnotations(t *testing.T) {
	for _, model := range []struct{ id, kind string }{{"qwen38fn_exl3", "qwen38fn_exl3"}, {"qwen38fn_exl3_q4", "qwen38fn_exl3"}, {"radixark", "qwen3.8"}} {
		run := workflows.Run{Document: &workflows.DocumentTask{Kind: "translate", Model: model.id, ModelType: model.kind, Effort: "low"}}
		units := make([]workflows.DocumentUnit, 3)
		for i := range units {
			u := &units[i]
			u.Ordinal = i
			u.Stage = "process"
			u.Status = "pending"
			for n := 0; n < 100; n++ {
				u.Inputs = append(u.Inputs, workflows.DocumentInput{ID: fmt.Sprintf("s:%d:%d", i, n), Source: "s", Block: fmt.Sprintf("b%06d", i*100+n), Text: "字", Group: fmt.Sprintf("g%d", i), Plain: true, Annotation: n%2 == 1})
			}
		}
		batch, indices := documentRequestBatch(run, units, 0, 1048576)
		if len(indices) != 3 || len(documentWireEntries(batch)) != 3 {
			t.Fatal(model, "annotation slots blocked batch", indices)
		}
		_, small := documentRequestBatch(run, units, 0, 4096)
		if len(small) != 1 {
			t.Fatal("small context incorrectly admitted batch")
		}
	}
}

func TestDocumentPartialBatchCommitsGoodCheckpointsOnlyOnce(t *testing.T) {
	var requests [][]string
	s, backend := documentFixture(t, func(w http.ResponseWriter, r *http.Request) {
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
		parts := strings.SplitN(body.Messages[len(body.Messages)-1].Content, "Reference data (JSON):\n", 2)
		if len(parts) != 2 {
			t.Error("no document payload")
			return
		}
		if err := json.Unmarshal([]byte(parts[1]), &entries); err != nil {
			t.Error(err)
			return
		}
		texts := []string{}
		out := map[string]string{}
		for i, entry := range entries {
			texts = append(texts, entry["text"])
			if len(requests) == 0 && i == 1 {
				continue
			}
			out[entry["id"]] = "번역 " + entry["text"]
		}
		requests = append(requests, texts)
		content, _ := json.Marshal(map[string]any{"entries": out})
		event, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]string{"content": string(content)}, "finish_reason": "stop"}}, "usage": map[string]int{"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700}})
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", event)
	})
	s.cfg.Model.DefaultModel = "qwen38fn_exl3"
	s.cfg.Model.ModelType = "qwen38fn_exl3"
	s.cfg.Model.ReasoningEffort = "low"
	s.cfg.Context.WindowTokens = 1048576
	s.llm = llm.New(backend.URL, "qwen38fn_exl3", "", "qwen38fn_exl3")
	var paragraphs []string
	for i := 0; i < 4; i++ {
		paragraphs = append(paragraphs, fmt.Sprintf("PART_%d ", i)+strings.Repeat("A complete sentence. ", 95))
	}
	id := fixtureSource(t, s, strings.Join(paragraphs, "\n\n"))
	run, err := s.documentAction(context.Background(), "s", documentStart{Action: "start", Kind: "translate", Sources: []string{id}})
	if err != nil {
		t.Fatal(err)
	}
	run = waitDocument(t, s, run.ID)
	if run.Status != "completed" || len(requests) != 2 || len(requests[0]) != 4 || len(requests[1]) != 1 || !strings.Contains(requests[1][0], "PART_1") {
		t.Fatalf("usable checkpoints were repeated: status=%s calls=%d requests=%v", run.Status, len(requests), requests)
	}
	units, _ := s.db.DocumentUnits("s", run.ID)
	owners := 0
	for _, u := range units {
		for _, a := range u.Attempts {
			if a.SharedWithOrdinal == nil {
				owners++
			}
		}
	}
	if owners != 2 {
		t.Fatal("usage duplicated across batch", owners)
	}
	output, _, err := s.db.DocumentOutput("s", run.ID, "output.txt")
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 4; i++ {
		if strings.Count(string(output), fmt.Sprintf("PART_%d", i)) != 1 {
			t.Fatal("source repeated or omitted", i)
		}
	}
}

func TestDocumentPartialOutputRejectsAmbiguousOrMalformedIDs(t *testing.T) {
	u := workflows.DocumentUnit{Inputs: []workflows.DocumentInput{{ID: "a", Text: "First paragraph"}, {ID: "b", Text: "Second paragraph"}}}
	out, err := validateDocumentOutput(u, "translate", llm.StreamResult{Content: `{"entries":{"p000001":"첫 문단"}}`})
	if err == nil || out["a"] != "첫 문단" {
		t.Fatal("unambiguous partial result unavailable", out, err)
	}
	for _, raw := range []string{
		`{"entries":{"p000001":"첫 문단","p1":"중복"}}`,
		`{"entries":{"p000001":"첫 문단","unexpected":"알 수 없음"}}`,
		`{"entries":{"p000001":"첫 문단","p000002":null}}`,
		`{"entries":{"p000001":"첫 문단"},"entries":{}}`,
		`{"entries":{"p000001":"첫 문단"}`,
	} {
		out, err := validateDocumentOutput(u, "translate", llm.StreamResult{Content: raw})
		if err == nil || out != nil {
			t.Fatal("ambiguous result can be committed", raw, out, err)
		}
	}
}

func TestDocumentRepairContextDoesNotUseLaterCompletedProse(t *testing.T) {
	units := []workflows.DocumentUnit{
		{Status: "completed", Inputs: []workflows.DocumentInput{{ID: "first", Source: "s", Block: "b000001"}, {ID: "last", Source: "s", Block: "b000003"}}, Output: map[string]string{"first": "이전 문단", "last": "나중 문단"}},
		{Status: "pending", Inputs: []workflows.DocumentInput{{ID: "missing", Source: "s", Block: "b000002"}}},
	}
	if got := documentContextBefore(units, units[1]); strings.TrimSpace(got) != "이전 문단" {
		t.Fatal("repair saw future context", got)
	}
}

func TestDocumentInlineImageAndEmptyAnchorBlockCoalescing(t *testing.T) {
	raw := `<html><body><p>Before<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg==" alt="Picture">After</p><p>Left<a id="page2"></a>Right</p></body></html>`
	b, err := sourcebook.Parse([]byte(raw), "text/html", "https://example.org/book", nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, slot := range b.TranslationSlots() {
		if slot.Plain {
			t.Fatal("image or anchor position can be moved by coalescing", slot)
		}
	}
}
