package server

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"sparktalk/internal/llm"
	"sparktalk/internal/workflows"
)

func TestDocumentReviewReplay(t *testing.T) {
	dir := os.Getenv("SPARKTALK_DOCUMENT_REVIEW_REPLAY")
	if dir == "" {
		t.Skip("opt-in recorded review replay")
	}
	var fixture struct {
		Batches [][]struct {
			ID     string `json:"id"`
			Source string `json:"source"`
			Target string `json:"target"`
		} `json:"batches"`
	}
	data, err := os.ReadFile(filepath.Join(dir, "fixture.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err = json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	files, err := filepath.Glob(filepath.Join(dir, "*-*.json"))
	if err != nil {
		t.Fatal(err)
	}
	var records []map[string]any
	for _, file := range files {
		var record struct {
			IDs          []string `json:"ids"`
			Raw          string   `json:"raw_content"`
			FinishReason string   `json:"finish_reason"`
		}
		raw, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		if json.Unmarshal(raw, &record) != nil || len(record.IDs) == 0 {
			continue
		}
		u := workflows.DocumentUnit{Output: map[string]string{}}
		for i, id := range record.IDs {
			found := false
			for _, batch := range fixture.Batches {
				for _, pair := range batch {
					if pair.ID == id {
						local := documentLocalID(i)
						u.Inputs = append(u.Inputs, workflows.DocumentInput{ID: local, Source: "fixture", Group: local, Text: pair.Source})
						u.Output[local] = pair.Target
						found = true
					}
				}
			}
			if !found {
				t.Fatal("unmatched case", id)
			}
		}
		review, err := validateDocumentReview(u, llm.StreamResult{Content: record.Raw, FinishReason: record.FinishReason})
		out := map[string]any{"file": filepath.Base(file), "ids": record.IDs, "review": review, "valid": err == nil}
		if err != nil {
			out["error"] = err.Error()
		}
		records = append(records, out)
	}
	result, _ := json.MarshalIndent(records, "", "  ")
	if err = os.WriteFile(filepath.Join(dir, "revalidated.json"), result, 0600); err != nil {
		t.Fatal(err)
	}
	t.Log(len(records), "saved responses revalidated without model calls")
}
