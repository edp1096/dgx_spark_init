package sourcebook

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestPlainLayoutSavedBookReplay(t *testing.T) {
	dir := os.Getenv("SPARKTALK_PLAIN_LAYOUT_REPLAY")
	if dir == "" {
		t.Skip("isolated saved book replay")
	}
	raw, err := os.ReadFile(filepath.Join(dir, "alice-source.txt"))
	if err != nil {
		t.Fatal(err)
	}
	b, err := Parse(raw, "text/plain", "", nil)
	if err != nil {
		t.Fatal(err)
	}
	priorRaw, _ := os.ReadFile(filepath.Join(dir, "alice-source.json"))
	var old Book
	if err = json.Unmarshal(priorRaw, &old); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(b.Blocks, old.Blocks) || b.Hash != old.Hash {
		t.Fatal("source identities/offsets changed")
	}
	var units []struct {
		Inputs []struct{ ID, Block string }
		Output map[string]string
	}
	unitRaw, _ := os.ReadFile(filepath.Join(dir, "alice-units.json"))
	if err = json.Unmarshal(unitRaw, &units); err != nil {
		t.Fatal(err)
	}
	values := map[string]string{}
	for _, u := range units {
		for _, in := range u.Inputs {
			values[in.Block] += u.Output[in.ID]
		}
	}
	text, err := b.Plain(values)
	if err != nil {
		t.Fatal(err)
	}
	html, err := b.HTML(values, true)
	if err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(dir, "layout-replay.txt"), text, 0600)
	os.WriteFile(filepath.Join(dir, "layout-replay.html"), html, 0600)
	modes := b.textLayouts()
	folded, retained := 0, 0
	for _, block := range b.Blocks {
		if strings.Contains(block.Text, "\n") {
			if modes[block.ID] == "normal" {
				folded++
			} else {
				retained++
			}
		}
	}
	report, _ := json.MarshalIndent(map[string]any{"source_blocks": len(b.Blocks), "source_unchanged": true, "folded_prose_blocks": folded, "preserved_multiline_blocks": retained}, "", "  ")
	os.WriteFile(filepath.Join(dir, "layout-replay.json"), report, 0600)
	if folded < 300 {
		t.Fatal("wrapped prose not recognized", folded)
	}
	t.Logf("%d prose paragraphs folded; %d authored/ambiguous blocks retained", folded, retained)
}
