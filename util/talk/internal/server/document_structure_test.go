package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sparktalk/internal/llm"
	"sparktalk/internal/sourcebook"
	"sparktalk/internal/workflows"
	"strings"
	"sync/atomic"
	"testing"
	"unicode/utf8"
)

func TestDocumentValidationRetryFeedbackAndParagraphSplit(t *testing.T) {
	var calls atomic.Int32
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) {
		n := calls.Add(1)
		if n <= 2 {
			var body struct {
				Messages []llm.Message `json:"messages"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Error(err)
			}
			if n == 2 && !strings.Contains(fmt.Sprint(body.Messages[len(body.Messages)-1].Content), "Previous response was rejected") {
				t.Error("unchanged blind retry")
			}
			w.Header().Set("Content-Type", "text/event-stream")
			w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"{\\\"entries\\\":{}}\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
			return
		}
		mockDocumentResponse(w, r, true)
	})
	id := fixtureSource(t, s, "First paragraph.\n\nSecond paragraph.\n\nFinal paragraph.")
	run, err := s.documentAction(context.Background(), "s", documentStart{Action: "start", Kind: "translate", Sources: []string{id}})
	if err != nil {
		t.Fatal(err)
	}
	run = waitDocument(t, s, run.ID)
	if run.Status != "completed" || run.Document.Total != 2 || calls.Load() != 4 {
		t.Fatalf("retry did not adapt: %+v, calls %d", run.Document, calls.Load())
	}
	data, _, err := s.db.DocumentOutput("s", run.ID, "output.txt")
	if err != nil || !strings.Contains(string(data), "Final paragraph.") {
		t.Fatal("split source tail lost", err)
	}
}

func TestDocumentParagraphValidationRequiresCoverageNotEveryInlineSlot(t *testing.T) {
	u := workflows.DocumentUnit{Inputs: []workflows.DocumentInput{
		{ID: "a", Group: "paragraph1", Text: "A "}, {ID: "b", Group: "paragraph1", Text: "word"},
		{ID: "c", Group: "paragraph1", Text: "reading", Annotation: true}, {ID: "d", Group: "paragraph2", Text: "Tail"},
	}}
	valid := `{"entries":{"p1":"단어","p2":"","p4":"끝"}}`
	out, err := validateDocumentOutput(u, "translate", llm.StreamResult{Content: valid})
	if err != nil || len(out) != 4 || out["c"] != "" {
		t.Fatal(out, err)
	}
	for _, invalid := range []string{
		`{"entries":{"p1":"단어","p4":"끝"}}`,
		`{"entries":{"p1":"단어","p2":"","p2":"","p4":"끝"}}`,
		`{"entries":{"p1":"단어","p2":null,"p4":"끝"}}`,
		`{"entries":{"p1":"단어","p2":"","p4":""}}`,
		`{"entries":{"p1":"","p2":"","p4":"단어 끝"}}`,
		`{"entries":{"p1":"단어","p2":"","p4":"끝","p3":"읽기"}}`,
	} {
		if _, err := validateDocumentOutput(u, "translate", llm.StreamResult{Content: invalid}); err == nil {
			t.Fatal("invalid accepted", invalid)
		}
	}
}

func TestDocumentQualityAndRepairUseWholeParagraph(t *testing.T) {
	u := workflows.DocumentUnit{Inputs: []workflows.DocumentInput{
		{ID: "a", Group: "g", Text: strings.Repeat("source ", 40)}, {ID: "b", Group: "g", Text: "end"},
	}, Output: map[string]string{"a": "", "b": strings.Repeat("번역 ", 40)}}
	if bad := shortDocumentOutputs(u); len(bad) > 0 {
		t.Fatal("valid word relocation marked short", bad)
	}
	u.Output["b"] = "끝"
	bad := shortDocumentOutputs(u)
	if len(bad) != 2 {
		t.Fatal("repair must include whole prose group", bad)
	}
	u.Inputs = append(u.Inputs, workflows.DocumentInput{ID: "c", Group: "other", Text: "tail"})
	u.Output["c"] = "꼬리"
	repair := extractDocumentRepair(&u, bad, 1)
	if repair == nil || len(repair.Inputs) != 2 || len(u.Inputs) != 1 || u.Output["c"] != "꼬리" {
		t.Fatal("repair lost paragraph or good result")
	}
}

func TestDocumentLegacyRubyTaskExportsAndSkipsAnnotationRequests(t *testing.T) {
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) { mockDocumentResponse(w, r, true) })
	raw := []byte(`<html><body><h1>Title</h1><p>A <ruby><rb>漢字</rb><rp>(</rp><rt>かんじ</rt><rp>)</rp></ruby> word.</p><p>Tail.</p></body></html>`)
	b, err := sourcebook.Parse(raw, "text/html", "https://example.com/book", nil)
	if err != nil {
		t.Fatal(err)
	}
	b.Version = 2
	var downgrade func(*sourcebook.Node)
	downgrade = func(n *sourcebook.Node) {
		if n.Tag == "ruby" || n.Tag == "rb" || n.Tag == "rp" || n.Tag == "rt" {
			n.Tag = "div"
		}
		for _, c := range n.Children {
			downgrade(c)
		}
	}
	downgrade(b.Root)
	id, err := s.storeSource("s", b, raw)
	if err != nil {
		t.Fatal(err)
	}
	run, err := s.documentAction(context.Background(), "s", documentStart{Action: "start", Kind: "translate", Sources: []string{id}})
	if err != nil {
		t.Fatal(err)
	}
	run = waitDocument(t, s, run.ID)
	if run.Status != "completed" {
		t.Fatalf("legacy ruby failed: %+v", run.Document)
	}
	units, _ := s.db.DocumentUnits("s", run.ID)
	n := 0
	for _, u := range units {
		for _, in := range u.Inputs {
			if in.Annotation {
				n++
				v, ok := u.Output[in.ID]
				if !ok || v != "" {
					t.Fatal("annotation output not tracked", in.ID)
				}
			}
		}
	}
	if n != 3 {
		t.Fatal("legacy annotations not recovered", n)
	}
	for _, format := range []string{"txt", "md", "html", "zip"} {
		data, _, err := s.db.DocumentOutput("s", run.ID, "output."+format)
		if err != nil || len(data) == 0 {
			t.Fatal(format, err)
		}
		if format != "zip" && strings.Contains(string(data), "かんじ") {
			t.Fatal("annotation translated as prose")
		}
	}
}

// Read-only reproduction against a saved source, without DB startup or LLM calls.
func TestDocumentSavedStructureRecovery(t *testing.T) {
	dir := os.Getenv("SPARKTALK_DOCUMENT_STRUCTURE_FIXTURE")
	if dir == "" {
		t.Skip("opt-in saved snapshot")
	}
	data, err := os.ReadFile(filepath.Join(dir, "source.json"))
	if err != nil {
		t.Fatal(err)
	}
	var b sourcebook.Book
	if err = json.Unmarshal(data, &b); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "source.raw"))
	if err != nil {
		t.Fatal(err)
	}
	if !b.RestoreInlineStructure(raw) {
		t.Fatal("saved snapshot cannot safely recover inline structure")
	}
	var f struct {
		Run    workflows.Run
		Unit   workflows.DocumentUnit
		Window int
	}
	data, err = os.ReadFile(filepath.Join(dir, "fixture.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(data, &f); err != nil {
		t.Fatal(err)
	}
	units := []workflows.DocumentUnit{f.Unit}
	decorateDocumentInputs(units, f.Run.Document.Sources[0], &b)
	f.Unit = units[0]
	annotations := 0
	groups := map[string]bool{}
	for _, in := range f.Unit.Inputs {
		if in.Annotation {
			annotations++
		} else {
			groups[in.Group] = true
		}
	}
	if annotations == 0 || len(groups) < 2 {
		t.Fatal("missing paragraph/annotation recovery")
	}
	data, _ = json.Marshal(f)
	if err = os.WriteFile(filepath.Join(dir, "fixture-decorated.json"), data, 0600); err != nil {
		t.Fatal(err)
	}
	t.Logf("all %d source nodes retained; first checkpoint %d slots, %d annotations, %d prose groups", len(b.Blocks), len(f.Unit.Inputs), annotations, len(groups))
}

// Export the saved, contiguous completed prefix in an isolated DB. This checks
// real model outputs against the production renderer without changing task scope
// or writing partial deliverables into the user's conversation.
func TestDocumentSavedCompletedPrefixExports(t *testing.T) {
	dir := os.Getenv("SPARKTALK_DOCUMENT_STRUCTURE_FIXTURE")
	if dir == "" {
		t.Skip("opt-in saved completed checkpoints")
	}
	data, err := os.ReadFile(filepath.Join(dir, "verified-units.json"))
	if os.IsNotExist(err) {
		t.Skip("checkpoints have not completed yet")
	}
	if err != nil {
		t.Fatal(err)
	}
	var all []workflows.DocumentUnit
	if err = json.Unmarshal(data, &all); err != nil {
		t.Fatal(err)
	}
	units := []workflows.DocumentUnit{}
	for _, u := range all {
		if u.Status != "completed" {
			break
		}
		units = append(units, u)
	}
	if len(units) == 0 {
		t.Fatal("no completed contiguous prefix")
	}
	data, err = os.ReadFile(filepath.Join(dir, "verified-run.json"))
	if err != nil {
		t.Fatal(err)
	}
	var run workflows.Run
	if err = json.Unmarshal(data, &run); err != nil {
		t.Fatal(err)
	}
	data, err = os.ReadFile(filepath.Join(dir, "source.json"))
	if err != nil {
		t.Fatal(err)
	}
	var b sourcebook.Book
	if err = json.Unmarshal(data, &b); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "source.raw"))
	if err != nil {
		t.Fatal(err)
	}
	if !b.RestoreInlineStructure(raw) {
		t.Fatal("could not recover original ruby structure")
	}
	starts := map[string]int{}
	for _, block := range b.Blocks {
		starts[block.ID] = block.Start
	}
	end := 0
	for _, u := range units {
		if err := sourcebook.ValidateTranslation(documentTranslationSlots(u), u.Output); err != nil {
			t.Fatal(err)
		}
		if len(shortDocumentOutputs(u)) > 0 {
			t.Fatal("short real translation")
		}
		for _, in := range u.Inputs {
			end = max(end, starts[in.Block]+in.Part+utf8.RuneCountInString(in.Text))
		}
	}
	run.SessionID = "s"
	run.Document.End = end
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) { t.Error("export must not prompt model") })
	if err = s.db.SaveSource(run.Document.Sources[0], "s", &b, raw); err != nil {
		t.Fatal(err)
	}
	if err = s.db.SaveWorkflowRun(&run); err != nil {
		t.Fatal(err)
	}
	if err = s.exportDocument(context.Background(), run, units); err != nil {
		t.Fatal(err)
	}
	for _, format := range []string{"txt", "md", "html", "zip"} {
		data, _, err := s.db.DocumentOutput("s", run.ID, "output."+format)
		if err != nil || len(data) == 0 {
			t.Fatal(format, err)
		}
		if err = os.WriteFile(filepath.Join(dir, "verified-prefix."+format), data, 0600); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("%d real completed checkpoints exported to TXT/MD/HTML/ZIP with exact original source coverage", len(units))
}
