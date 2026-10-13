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

func documentLayoutFixture(t *testing.T, raw string) (*sourcebook.Book, workflows.DocumentUnit) {
	t.Helper()
	b, err := sourcebook.Parse([]byte(raw), "text/html", "https://example.org/book", nil)
	if err != nil {
		t.Fatal(err)
	}
	u := workflows.DocumentUnit{}
	for _, block := range b.Blocks {
		u.Inputs = append(u.Inputs, workflows.DocumentInput{ID: "s:" + block.ID + ":0", Source: "s", Block: block.ID, Text: block.Text})
	}
	units := []workflows.DocumentUnit{u}
	decorateDocumentInputs(units, "s", b)
	return b, units[0]
}

func TestDocumentRepairsOnlyParagraphWhoseLinkWasEmptied(t *testing.T) {
	var requests [][]map[string]string
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
		var entries []map[string]string
		parts := strings.SplitN(body.Messages[len(body.Messages)-1].Content, "Reference data (JSON):\n", 2)
		if len(parts) != 2 {
			t.Error("missing payload")
			return
		}
		if err := json.Unmarshal([]byte(parts[1]), &entries); err != nil {
			t.Error(err)
			return
		}
		output := map[string]string{}
		for _, entry := range entries {
			output[entry["id"]] = "번역 " + entry["text"]
			if len(requests) == 0 && entry["text"] == "reference" {
				output[entry["id"]] = ""
			}
		}
		requests = append(requests, entries)
		content, _ := json.Marshal(map[string]any{"entries": output})
		event, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]string{"content": string(content)}, "finish_reason": "stop"}}})
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", event)
	})
	raw := []byte(`<html><body><p>Read <a href="#note">reference</a> carefully.</p><p id="note">Complete footnote.</p></body></html>`)
	b, err := sourcebook.Parse(raw, "text/html", "https://example.org/book", nil)
	if err != nil {
		t.Fatal(err)
	}
	id, err := s.storeSource("s", b, raw)
	if err != nil {
		t.Fatal(err)
	}
	run, err := s.documentAction(context.Background(), "s", documentStart{Action: "start", Kind: "translate", Sources: []string{id}})
	if err != nil {
		t.Fatal(err)
	}
	run = waitDocument(t, s, run.ID)
	if run.Status != "completed" || len(requests) != 2 {
		t.Fatal("layout repair did not finish", run.Status, len(requests))
	}
	for _, entry := range requests[1] {
		if strings.Contains(entry["text"], "footnote") {
			t.Fatal("valid footnote was regenerated")
		}
	}
	page, _, err := s.db.DocumentOutput("s", run.ID, "output.html")
	if err != nil || !strings.Contains(string(page), `<a href="#note">번역 reference</a>`) {
		t.Fatal("link lost in final export", err, string(page))
	}
}

func TestDocumentRejectsEmptyLinkAndEmphasisDespiteNonemptyParagraph(t *testing.T) {
	for _, markup := range []string{`<a href="#note">reference</a>`, `<em>reference</em>`, `<sup>reference</sup>`, `<code>reference</code>`} {
		_, u := documentLayoutFixture(t, `<html><body><p>Read `+markup+` carefully.</p><p id="note">Complete footnote.</p></body></html>`)
		entries := map[string]string{}
		for _, entry := range documentWireEntries(u) {
			entries[entry.ID] = "번역 " + entry.Text
			if entry.Text == "reference" {
				entries[entry.ID] = ""
			}
			if strings.Contains(entry.Text, "Read") {
				entries[entry.ID] = "참조 내용을 주의 깊게 읽어라"
			}
		}
		payload, _ := json.Marshal(map[string]any{"entries": entries})
		out, err := validateDocumentOutput(u, "translate", llm.StreamResult{Content: string(payload)})
		if err == nil {
			t.Fatalf("empty protected span was accepted: %s", markup)
		}
		u.Output = out
		bad := incompleteDocumentGroups(u)
		for _, in := range u.Inputs {
			_, rejected := bad[in.ID]
			if strings.Contains(in.Text, "footnote") == rejected {
				t.Fatalf("repair must retain complete footnote and retry its affected paragraph: %+v", bad)
			}
		}
	}
}

func TestDocumentPreservesTextOnBothSidesOfEmptyAnchor(t *testing.T) {
	_, u := documentLayoutFixture(t, `<html><body><p>Before <a id="page2"><span> </span></a> after.</p></body></html>`)
	entries := map[string]string{}
	for _, entry := range documentWireEntries(u) {
		entries[entry.ID] = ""
		if strings.Contains(entry.Text, "Before") {
			entries[entry.ID] = "이전 이후"
		}
	}
	payload, _ := json.Marshal(map[string]any{"entries": entries})
	if _, err := validateDocumentOutput(u, "translate", llm.StreamResult{Content: string(payload)}); err == nil {
		t.Fatal("text crossed an empty anchor")
	}
}

func TestDocumentAllowsRelocationInsideOneLink(t *testing.T) {
	b, u := documentLayoutFixture(t, `<html><body><p>Read <a href="#note"><span>the </span><span>reference</span></a>.</p><p id="note">Note.</p></body></html>`)
	entries := map[string]string{}
	for _, entry := range documentWireEntries(u) {
		entries[entry.ID] = "번역 " + entry.Text
		if entry.Text == "the " {
			entries[entry.ID] = "참조"
		}
		if entry.Text == "reference" {
			entries[entry.ID] = ""
		}
	}
	payload, _ := json.Marshal(map[string]any{"entries": entries})
	out, err := validateDocumentOutput(u, "translate", llm.StreamResult{Content: string(payload)})
	if err != nil {
		t.Fatal("valid word reordering within link rejected", err)
	}
	translated := map[string]string{}
	for _, in := range u.Inputs {
		translated[in.Block] = out[in.ID]
	}
	page, err := b.HTML(translated, true)
	if err != nil || !strings.Contains(string(page), `<a href="#note"><span>참조`) {
		t.Fatal("linked text lost", err, string(page))
	}
}

func TestDocumentAllowsTargetLanguageWordOrderAroundLink(t *testing.T) {
	_, u := documentLayoutFixture(t, `<html><body><p>Read <a href="#note">the reference</a> carefully.</p><p id="note">Note.</p></body></html>`)
	entries := map[string]string{}
	for _, entry := range documentWireEntries(u) {
		switch entry.Text {
		case "Read ":
			entries[entry.ID] = ""
		case "the reference":
			entries[entry.ID] = "참조"
		case " carefully.":
			entries[entry.ID] = "를 주의 깊게 읽으세요."
		default:
			entries[entry.ID] = "주석."
		}
	}
	payload, _ := json.Marshal(map[string]any{"entries": entries})
	if _, err := validateDocumentOutput(u, "translate", llm.StreamResult{Content: string(payload)}); err != nil {
		t.Fatal("natural word order was prohibited", err)
	}
}

func TestWholeParagraphScopeAddsNoRedundantPromptMetadata(t *testing.T) {
	_, u := documentLayoutFixture(t, `<html><body><h1><a id="chapter1">Chapter 1</a></h1><p>Ordinary prose.</p></body></html>`)
	entries, _ := documentWirePayload(u)
	for _, entry := range entries {
		if entry["scope"] != "" {
			t.Fatal("whole-paragraph scope repeats existing protection", entry)
		}
	}
	entries[0]["text"] = ""
	output := map[string]string{}
	for _, entry := range entries {
		output[entry["id"]] = entry["text"]
	}
	payload, _ := json.Marshal(map[string]any{"entries": output})
	if _, err := validateDocumentOutput(u, "translate", llm.StreamResult{Content: string(payload)}); err == nil {
		t.Fatal("empty whole-paragraph span was accepted")
	}
}

func TestDocumentRejectsInvisibleControlsWithoutLanguagePurityFilter(t *testing.T) {
	u := workflows.DocumentUnit{Inputs: []workflows.DocumentInput{{ID: "a", Group: "first", Text: "First"}, {ID: "b", Group: "second", Text: "Second"}}}
	for _, badText := range []string{"\x00", "보존\x00할 문장", "\x01", "\uffff"} {
		payload, _ := json.Marshal(map[string]any{"entries": map[string]string{"p000001": badText, "p000002": "정상 문장"}})
		out, err := validateDocumentOutput(u, "translate", llm.StreamResult{Content: string(payload)})
		if err == nil || out["b"] != "정상 문장" {
			t.Fatal("control was accepted or good paragraph lost", err, out)
		}
		u.Output = out
		bad := incompleteDocumentGroups(u)
		if len(bad) != 1 || bad["a"] == "" {
			t.Fatal("control error must target only its paragraph", bad)
		}
	}
	payload, _ := json.Marshal(map[string]any{"entries": map[string]string{"p000001": "漢字・カナ・GPU\n\t정상 문장", "p000002": "둘째 문장"}})
	if _, err := validateDocumentOutput(u, "translate", llm.StreamResult{Content: string(payload)}); err != nil {
		t.Fatal("permitted mixed script or whitespace rejected", err)
	}
}
