package server

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"sparktalk/internal/db"
	"sparktalk/internal/sourcebook"
	"sparktalk/internal/workflows"
)

// Opt-in, isolated fixture creation. The input pairs are frozen source/output
// evidence; this function never opens a production database or translates text.
func TestDocumentLanguageFreeze(t *testing.T) {
	dir := os.Getenv("SPARKTALK_LANGUAGE_FREEZE")
	if dir == "" {
		t.Skip("explicit frozen language evidence")
	}
	raw, e := os.ReadFile(filepath.Join(dir, "pairs.json"))
	if e != nil {
		t.Fatal(e)
	}
	var pairs [][2]string
	if e = json.Unmarshal(raw, &pairs); e != nil {
		t.Fatal(e)
	}
	texts := []string{}
	for _, p := range pairs {
		texts = append(texts, p[0])
	}
	source := []byte(strings.Join(texts, "\n\n"))
	book, e := sourcebook.Parse(source, "text/plain", "https://example.org/frozen-language-check", nil)
	if e != nil {
		t.Fatal(e)
	}
	if len(book.Blocks) != len(pairs) {
		t.Fatal("fixture block mismatch", len(book.Blocks), len(pairs))
	}
	database, e := db.Open(filepath.Join(dir, "before.db"))
	if e != nil {
		t.Fatal(e)
	}
	defer database.Close()
	const sid, rid, sourceID = "language-fixture", "language-parent", "language-source"
	if _, e = database.CreateSession(sid, "번역 언어 검사", "qwen38fn_exl3", "medium"); e != nil {
		t.Fatal(e)
	}
	if e = database.SaveSource(sourceID, sid, book, source); e != nil {
		t.Fatal(e)
	}
	values := map[string]string{}
	for i, b := range book.Blocks {
		values[b.ID] = pairs[i][1]
	}
	units := planDocumentTranslation(sourceID, book, 0, 0)
	for i := range units {
		units[i].Status = "completed"
		units[i].Output = map[string]string{}
		for _, in := range units[i].Inputs {
			units[i].Output[in.ID] = values[in.Block]
		}
	}
	run := workflows.Run{ID: rid, SessionID: sid, Status: "completed", Current: 2, Document: &workflows.DocumentTask{Kind: "translate", Sources: []string{sourceID}, Phase: "completed", Total: len(units), Completed: len(units), Revision: 1, RequestSpec: &workflows.DocumentRequestSpec{Origin: "api", UserText: "한국어로 번역"}}, Steps: []workflows.StepState{{Status: "completed"}, {Status: "completed"}}}
	if e = database.CreateDocumentRun(&run, units); e != nil {
		t.Fatal(e)
	}
	text, e := book.Plain(values)
	if e != nil {
		t.Fatal(e)
	}
	if e = database.SaveDocumentOutput(sid, rid, "output.txt", "text/plain", text); e != nil {
		t.Fatal(e)
	}
	fixture := map[string]any{"session": sid, "run": rid, "forbidden": []string{"ella는", "if—I가", "that means", "~"}, "preserve": pairs[4:]}
	raw, _ = json.MarshalIndent(fixture, "", "  ")
	if e = os.WriteFile(filepath.Join(dir, "fixture.json"), raw, 0600); e != nil {
		t.Fatal(e)
	}
}
