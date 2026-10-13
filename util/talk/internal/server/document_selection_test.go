package server

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"unicode/utf8"

	"sparktalk/internal/sourcebook"
	"sparktalk/internal/workflows"
)

// Evaluator-only document positions. No production code knows these books.
func TestDocumentSelectionFreeze(t *testing.T) {
	dir := os.Getenv("SPARKTALK_DOCUMENT_SELECTION_FREEZE")
	if dir == "" {
		t.Skip("opt-in selection sample preparation")
	}
	out := filepath.Join(dir, "selection-fixture.json")
	if _, err := os.Stat(out); err == nil {
		t.Skip("selection samples already frozen")
	}
	cases := []map[string]any{}
	selection := []map[string]any{}
	for _, item := range []struct{ name, first, title string }{{"rashomon", "b000003", "羅生門 — 芥川龍之介"}, {"individualism", "b000004", "私の個人主義 — 夏目漱石"}, {"gift_magi", "b000096", "The Gift of the Magi — O. Henry"}, {"art_of_war", "b000672", "The Art of War — Sun Tzu, translated by Lionel Giles"}} {
		raw, err := os.ReadFile(filepath.Join(dir, item.name+".book.json"))
		if err != nil {
			t.Fatal(err)
		}
		var book sourcebook.Book
		if err = json.Unmarshal(raw, &book); err != nil {
			t.Fatal(err)
		}
		all := planDocumentTranslation(item.name, &book, 0, 0)
		chosen := workflows.DocumentUnit{Stage: "process", Status: "pending"}
		chars := 0
		found := false
		for _, unit := range all {
			for _, group := range documentTextGroups(unit) {
				if len(group.Inputs) == 0 {
					continue
				}
				if group.Inputs[0].Block >= item.first {
					found = true
				}
				if !found || chars >= 1400 {
					continue
				}
				chosen.Inputs = append(chosen.Inputs, group.Inputs...)
				for _, in := range group.Inputs {
					if !in.Annotation {
						chars += utf8.RuneCountInString(in.Text)
					}
				}
			}
		}
		if chars < 1000 || chars > 5000 {
			t.Fatal("sample span size", item.name, chars)
		}
		sourceContext := documentSourceContext(all, chosen, 1280)
		for _, effort := range []string{"low", "medium"} {
			for _, temp := range []float64{0.7, 0.2} {
				label := "t07"
				if temp == 0.2 {
					label = "t02"
				}
				run := workflows.Run{Document: &workflows.DocumentTask{Kind: "translate", Pipeline: workflows.DocumentPipelineVersion, Instructions: "주어진 원문을 내용 누락 없이 자연스러운 한국어로 번역한다.", SourceContext: sourceContext, SourceTitles: map[string]string{item.name: item.title}, SourceURLs: map[string]string{item.name: book.URL}}}
				cases = append(cases, map[string]any{"name": item.name + "-" + effort + "-" + label, "run": run, "units": []workflows.DocumentUnit{chosen}, "effort": effort, "temperature": temp})
			}
		}
		selection = append(selection, map[string]any{"work": item.name, "source_hash": book.Hash, "prose_chars": chars, "inputs": chosen.Inputs, "source_context": sourceContext})
	}
	fixture := map[string]any{"database": "/home/edp1096/workspace/dgx_spark_init/util/talk/dist/sparktalk.db", "config_url": "http://127.0.0.1:8585/api/config", "window": 1048576, "cases": cases}
	raw, _ := json.MarshalIndent(fixture, "", "  ")
	if err := os.WriteFile(out, append(raw, '\n'), 0600); err != nil {
		t.Fatal(err)
	}
	raw, _ = json.MarshalIndent(selection, "", "  ")
	if err := os.WriteFile(filepath.Join(dir, "selection-spans.json"), append(raw, '\n'), 0600); err != nil {
		t.Fatal(err)
	}
}

// A second, disjoint window per selection work, frozen before its generation.
// These evaluator positions never enter production decisions.
func TestDocumentSelectionSupplementFreeze(t *testing.T) {
	dir := os.Getenv("SPARKTALK_DOCUMENT_SELECTION_FREEZE")
	if dir == "" {
		t.Skip("opt-in selection sample preparation")
	}
	out := filepath.Join(dir, "selection-supplement-fixture.json")
	if _, err := os.Stat(out); err == nil {
		t.Skip("already frozen")
	}
	var cases []map[string]any
	var spans []map[string]any
	for _, name := range []string{"rashomon", "individualism", "gift_magi", "art_of_war"} {
		raw, err := os.ReadFile(filepath.Join(dir, name+".book.json"))
		if err != nil {
			t.Fatal(err)
		}
		var book sourcebook.Book
		if err = json.Unmarshal(raw, &book); err != nil {
			t.Fatal(err)
		}
		all := planDocumentTranslation(name, &book, 0, 0)
		var groups [][]workflows.DocumentInput
		total := 0
		for _, u := range all {
			for _, g := range documentTextGroups(u) {
				if len(g.Inputs) == 0 || (name == "gift_magi" && g.Inputs[0].Chapter != "THE GIFT OF THE MAGI") {
					continue
				}
				groups = append(groups, g.Inputs)
				for _, in := range g.Inputs {
					if !in.Annotation {
						total += utf8.RuneCountInString(in.Text)
					}
				}
			}
		}
		chosen := workflows.DocumentUnit{Stage: "process", Status: "pending"}
		seen, chars := 0, 0
		for _, group := range groups {
			n := 0
			for _, in := range group {
				if !in.Annotation {
					n += utf8.RuneCountInString(in.Text)
				}
			}
			if seen >= total/2 && chars < 1600 {
				chosen.Inputs = append(chosen.Inputs, group...)
				chars += n
			}
			seen += n
		}
		if chars < 1200 || chars > 6000 {
			t.Fatal(name, "invalid second window", chars)
		}
		context := documentSourceContext(all, chosen, 1280)
		for _, effort := range []string{"low", "medium"} {
			run := workflows.Run{Document: &workflows.DocumentTask{Kind: "translate", Pipeline: workflows.DocumentPipelineVersion, Instructions: "주어진 원문을 내용 누락 없이 자연스러운 한국어로 번역한다.", SourceContext: context, SourceTitles: map[string]string{name: book.Title}, SourceURLs: map[string]string{name: book.URL}}}
			cases = append(cases, map[string]any{"name": name + "-window2-" + effort + "-t02", "effort": effort, "temperature": 0.2, "run": run, "units": []workflows.DocumentUnit{chosen}})
		}
		spans = append(spans, map[string]any{"work": name, "source_hash": book.Hash, "prose_chars": chars, "inputs": chosen.Inputs, "source_context": context})
	}
	fixture := map[string]any{"database": "/home/edp1096/workspace/dgx_spark_init/util/talk/dist/sparktalk.db", "config_url": "http://127.0.0.1:8585/api/config", "window": 1048576, "cases": cases}
	for path, value := range map[string]any{out: fixture, filepath.Join(dir, "selection-supplement-spans.json"): spans} {
		raw, err := json.MarshalIndent(value, "", "  ")
		if err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(path, append(raw, '\n'), 0600); err != nil {
			t.Fatal(err)
		}
	}
}

// Evaluate review granularity on a frozen, real pipeline output, with reading
// hints and surrounding source preserved. This is selection data, not heldout.
func TestDocumentReviewPilotFreeze(t *testing.T) {
	root := os.Getenv("SPARKTALK_DOCUMENT_REVIEW_PILOT_FREEZE")
	if root == "" {
		t.Skip("opt-in fixed pipeline review fixture")
	}
	raw, err := os.ReadFile(filepath.Join(root, "pipeline-pilot-q4", "rashomon.json"))
	if err != nil {
		t.Fatal(err)
	}
	var saved struct {
		Run   workflows.Run            `json:"run"`
		Units []workflows.DocumentUnit `json:"units"`
	}
	if err = json.Unmarshal(raw, &saved); err != nil {
		t.Fatal(err)
	}
	type item struct {
		Pair documentReviewPair
		Unit workflows.DocumentUnit
		ID   string
	}
	var all []item
	ordinal := 0
	for _, unit := range saved.Units {
		for _, group := range documentTextGroups(unit) {
			one := workflows.DocumentUnit{Inputs: group.Inputs, Output: unit.Output}
			pairs := documentReviewPairs(one)
			if len(pairs) == 0 {
				continue
			}
			all = append(all, item{pairs[0], one, fmt.Sprintf("rashomon-%03d", ordinal)})
			ordinal++
		}
	}
	var selected []item
	for i, it := range all {
		if i < 12 || (i >= 22 && i < 32) || (i >= 39 && i < 43) {
			selected = append(selected, it)
		}
	}
	for _, width := range []int{4, 8} {
		var batches [][]map[string]string
		var contexts []string
		for start := 0; start < len(selected); start += width {
			var batch []map[string]string
			combined := workflows.DocumentUnit{}
			for _, it := range selected[start:min(len(selected), start+width)] {
				batch = append(batch, map[string]string{"id": it.ID, "source": it.Pair.Source, "target": it.Pair.Target, "context": it.Pair.Readings})
				combined.Inputs = append(combined.Inputs, it.Unit.Inputs...)
			}
			batches = append(batches, batch)
			contexts = append(contexts, documentSourceContext(saved.Units, combined, 1280))
		}
		dir := filepath.Join(root, fmt.Sprintf("reviewer-pilot-focus%d-q4", width))
		if err = os.MkdirAll(dir, 0700); err != nil {
			t.Fatal(err)
		}
		out := filepath.Join(dir, "fixture.json")
		if _, err = os.Stat(out); err == nil {
			continue
		}
		fixture := map[string]any{"database": "/home/edp1096/workspace/dgx_spark_init/util/talk/dist/sparktalk.db", "config_url": "http://127.0.0.1:8585/api/config", "efforts": []string{"low"}, "pipeline": workflows.DocumentPipelineVersion, "source_titles": saved.Run.Document.SourceTitles, "batches": batches, "shared_contexts": contexts}
		data, _ := json.MarshalIndent(fixture, "", "  ")
		if err = os.WriteFile(out, append(data, '\n'), 0600); err != nil {
			t.Fatal(err)
		}
	}
}
