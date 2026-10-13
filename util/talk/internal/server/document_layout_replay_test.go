package server

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"sparktalk/internal/llm"
	"sparktalk/internal/sourcebook"
	"sparktalk/internal/workflows"
)

// Replay recorded model replies through the current verifier without generating
// or modifying production data. Fixtures contain the original source snapshots.
func TestDocumentRecordedRepliesRemainValid(t *testing.T) {
	dir := os.Getenv("SPARKTALK_DOCUMENT_REPLAY")
	if dir == "" {
		t.Skip("opt-in recorded translation audit")
	}
	var fixture struct {
		Cases []struct {
			Name  string                      `json:"name"`
			Units []workflows.DocumentUnit    `json:"units"`
			Books map[string]*sourcebook.Book `json:"books"`
			Raw   map[string][]byte           `json:"raw_sources"`
		} `json:"cases"`
	}
	raw, err := os.ReadFile(filepath.Join(dir, "comparison-fixture.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	for _, tc := range fixture.Cases {
		for source, book := range tc.Books {
			book.RestoreInlineStructure(tc.Raw[source])
			decorateDocumentInputs(tc.Units, source, book)
		}
		for _, variant := range []string{"improved-exl3", "improved-nvfp4", "off-exl3", "off-nvfp4"} {
			if tc.Name != "wordplay" && (variant == "off-exl3" || variant == "off-nvfp4") {
				continue
			}
			t.Run(variant+"/"+tc.Name, func(t *testing.T) {
				var result struct {
					Valid  bool              `json:"valid"`
					Output map[string]string `json:"output"`
					Calls  []struct {
						Indices []int  `json:"indices"`
						Raw     string `json:"raw_content"`
					} `json:"calls"`
				}
				raw, err := os.ReadFile(filepath.Join(dir, variant, tc.Name+".json"))
				if err != nil {
					t.Fatal(err)
				}
				if err = json.Unmarshal(raw, &result); err != nil {
					t.Fatal(err)
				}
				if !result.Valid {
					t.Fatal("recorded case was incomplete")
				}
				for _, call := range result.Calls {
					u := workflows.DocumentUnit{}
					for _, index := range call.Indices {
						u.Inputs = append(u.Inputs, documentRequestInputs(tc.Units[index])...)
					}
					out, err := validateDocumentOutput(u, "translate", llm.StreamResult{Content: call.Raw})
					if err != nil {
						t.Fatal(err)
					}
					if len(out) != len(u.Inputs) {
						t.Fatal("coverage changed")
					}
					for id, value := range out {
						if result.Output[id] != value {
							t.Fatal("recorded output mapping changed", id)
						}
					}
				}
			})
		}
	}
}
