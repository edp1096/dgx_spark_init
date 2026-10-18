package server

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"sparktalk/internal/workflows"
)

// Replay a completed correction's accepted edits against its parent, without
// inference or database writes. Every published byte must follow the scoped
// patch ledger; neither a counter nor a model's valid=true is sufficient.
func TestDocumentTermsAcceptedPatchReplay(t *testing.T) {
	dir := os.Getenv("SPARKTALK_TERMS_AUDIT_REPLAY")
	if dir == "" {
		t.Skip("explicit completed correction audit")
	}
	var run workflows.Run
	var before, final []workflows.DocumentUnit
	for _, item := range []struct {
		name  string
		value any
	}{{"run.json", &run}, {"before-units.json", &before}, {"units.json", &final}} {
		raw, err := os.ReadFile(filepath.Join(dir, item.name))
		if err != nil {
			t.Fatal(err)
		}
		if err = json.Unmarshal(raw, item.value); err != nil {
			t.Fatal(err)
		}
	}
	old := map[string]string{}
	for _, u := range before {
		for _, in := range u.Inputs {
			old[fmt.Sprintf("%s:%d", in.Block, in.Part)] = u.Output[in.ID]
		}
	}
	units := append([]workflows.DocumentUnit(nil), final...)
	for i := range units {
		units[i].Output = map[string]string{}
		for _, in := range units[i].Inputs {
			text, ok := old[fmt.Sprintf("%s:%d", in.Block, in.Part)]
			if !ok {
				t.Fatal("missing baseline", in.ID)
			}
			units[i].Output[in.ID] = text
		}
	}
	accepted := 0
	for _, c := range run.Document.Terms.Cases {
		for _, g := range c.Decisions {
			for _, id := range g.References {
				for _, r := range c.References {
					if r.ID != id || r.Status != "repaired" {
						continue
					}
					for key, text := range r.BeforeOutput {
						if units[r.Ordinal].Output[key] != text {
							t.Fatal("before checkpoint mismatch", c.Source, id, key)
						}
					}
					retained := []string{}
					if c.Pinned == "" {
						retained = append(retained, c.Source)
					}
					out, text, err := applyTermPatches(units, r, g.Canonical, r.Patches, retained...)
					if err != nil || documentTextHash(text) != r.ResultHash {
						t.Fatalf("%s/%s: result mismatch %v", c.Source, id, err)
					}
					for _, key := range r.InputIDs {
						units[r.Ordinal].Output[key] = out[key]
					}
					accepted++
				}
			}
		}
	}
	if run.Document.Terms.Language != nil {
		for _, r := range run.Document.Terms.Language.Records {
			if r.Status != "repaired" {
				continue
			}
			for key, text := range r.BeforeOutput {
				if units[r.Ordinal].Output[key] != text {
					t.Fatal("language baseline mismatch", r.ID, key)
				}
			}
			out, text, err := applyDocumentLanguage(units, r)
			if err != nil || documentTextHash(text) != r.ResultHash {
				t.Fatalf("language result mismatch %s: %v", r.ID, err)
			}
			for _, id := range r.InputIDs {
				units[r.Ordinal].Output[id] = out[id]
			}
			accepted++
		}
	}
	for i, u := range final {
		for id, text := range u.Output {
			if units[i].Output[id] != text {
				t.Fatal("published text differs from accepted edits", id)
			}
		}
	}
	t.Log("replayed", accepted, "accepted paragraph corrections; published output matches exactly")
}
