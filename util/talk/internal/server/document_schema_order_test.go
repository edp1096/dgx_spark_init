package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"testing"

	"sparktalk/internal/llm"
	"sparktalk/internal/workflows"
)

func TestDocumentSchemaPropertyOrderFollowsSourceAcrossDecimalBoundaries(t *testing.T) {
	u := workflows.DocumentUnit{}
	wanted := []string{}
	for i := 0; i < 163; i++ {
		annotation := i%7 == 3
		u.Inputs = append(u.Inputs, workflows.DocumentInput{ID: fmt.Sprintf("source:b%06d:0", i+1), Source: "source", Text: "word", Annotation: annotation})
		if !annotation {
			wanted = append(wanted, documentLocalID(i))
		}
	}
	_, backend := documentFixture(t, func(w http.ResponseWriter, r *http.Request) {
		data, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
			return
		}
		var request struct {
			ResponseFormat struct {
				JSONSchema struct {
					Schema struct {
						Properties struct {
							Entries struct {
								Properties json.RawMessage `json:"properties"`
								Required   []string        `json:"required"`
							} `json:"entries"`
						} `json:"properties"`
					} `json:"schema"`
				} `json:"json_schema"`
			} `json:"response_format"`
		}
		if err = json.Unmarshal(data, &request); err != nil {
			t.Error(err)
			return
		}
		entries := request.ResponseFormat.JSONSchema.Schema.Properties.Entries
		decoder := json.NewDecoder(bytes.NewReader(entries.Properties))
		if _, err = decoder.Token(); err != nil {
			t.Error(err)
			return
		}
		actual := []string{}
		for decoder.More() {
			key, err := decoder.Token()
			if err != nil {
				t.Error(err)
				return
			}
			actual = append(actual, key.(string))
			var value json.RawMessage
			if err = decoder.Decode(&value); err != nil {
				t.Error(err)
				return
			}
		}
		if fmt.Sprint(actual) != fmt.Sprint(wanted) || fmt.Sprint(entries.Required) != fmt.Sprint(wanted) {
			t.Error("wire schema forces generation out of source order", actual)
		}
		r.Body = io.NopCloser(bytes.NewReader(data))
		mockDocumentResponse(w, r, true)
	})
	run := workflows.Run{Document: &workflows.DocumentTask{Kind: "translate", Model: "model", ModelType: "generic", Effort: "off"}}
	result, err := requestDocumentUnit(context.Background(), llm.New(backend.URL, "model", ""), run, u, 1048576)
	if err != nil {
		t.Fatal(err)
	}
	out, err := validateDocumentOutput(u, "translate", result)
	if err != nil || len(out) != len(u.Inputs) {
		t.Fatal("padded IDs did not map back to original saved IDs", err, len(out))
	}
	for _, in := range u.Inputs {
		if _, ok := out[in.ID]; !ok {
			t.Fatal("lost immutable source ID", in.ID)
		}
	}
}
