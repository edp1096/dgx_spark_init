package server

import (
	"encoding/json"
	"strings"
	"testing"

	"sparktalk/internal/sourcebook"
	"sparktalk/internal/workflows"
)

func TestDocumentWireNormalizesOnlyHTMLPresentationWhitespace(t *testing.T) {
	for _, mime := range []string{"text/html", "text/plain"} {
		raw := "One long\n  line.\n\nAnother line."
		if mime == "text/html" {
			raw = "<p>One long\n  <span>line.</span></p><pre>Code\n  indent</pre><p class='verse'>Verse\nnext</p>"
		}
		b, err := sourcebook.Parse([]byte(raw), mime, "", nil)
		if err != nil {
			t.Fatal(err)
		}
		u := workflows.DocumentUnit{}
		for _, block := range b.Blocks {
			u.Inputs = append(u.Inputs, workflows.DocumentInput{ID: "s:" + block.ID, Source: "s", Block: block.ID, Text: block.Text})
		}
		units := []workflows.DocumentUnit{u}
		decorateDocumentInputs(units, "s", b)
		before, _ := json.Marshal(units)
		payload, _ := documentWirePayload(units[0])
		after, _ := json.Marshal(units)
		if string(before) != string(after) {
			t.Fatal("wire formatting mutated stored inputs")
		}
		if mime == "text/html" {
			if payload[0]["text"] != "One long line." || payload[0]["layout"] != "" {
				t.Fatal("HTML source wraps reached model", payload)
			}
			if payload[1]["text"] != "Code\n  indent" || payload[1]["layout"] != "preserve_lines" || payload[2]["layout"] != "preserve_lines" {
				t.Fatal("protected lines lost", payload)
			}
		} else if !strings.Contains(payload[0]["text"], "\n  ") || payload[0]["layout"] != "preserve_lines" {
			t.Fatal("text lines changed", payload)
		}
	}
}
