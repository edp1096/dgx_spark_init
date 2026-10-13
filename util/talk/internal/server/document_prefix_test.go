package server

import (
	"strings"
	"testing"

	"sparktalk/internal/workflows"
)

func TestDocumentCompletePrefixPreservesOnlyFinishedUnambiguousFields(t *testing.T) {
	u := workflows.DocumentUnit{Inputs: []workflows.DocumentInput{{ID: "s:a:0", Source: "s", Block: "a", Group: "a", Plain: true, Text: "First paragraph."}, {ID: "s:b:0", Source: "s", Block: "b", Group: "b", Plain: true, Text: "Second paragraph."}}}
	good := []string{`{"entries":{"p000001":"첫 문단.","p000002":"아직 끝나지`, `{"entries":{"p000001":"첫 문단."` + strings.Repeat("\n", 3000), `{"entries":{"p000001":"첫 문단."},"terms":[`}
	for _, raw := range good {
		out, err := documentCompletePrefix(u, raw)
		if err != nil || out["s:a:0"] != "첫 문단." {
			t.Fatal("lost completed paragraph", err, out)
		}
		if _, ok := out["s:b:0"]; ok {
			t.Fatal("invented incomplete output", out)
		}
	}
	for _, raw := range []string{`{"entries":{"p000001":"첫 문단.","p1":"duplicate`, `{"entries":{"p000001":"첫 문단.","unexpected":"unfinished`, `{"entries":{"p000001":"첫 문단.","p000002":null`, `{"entries":{"p000001":"첫 문단.","p000002":"bad\q`, `{"entries":{"p000001":"첫 문단."},"entries":`, `{"entries":{"p000001":"첫 문단."}} trailing`} {
		if out, err := documentCompletePrefix(u, raw); err == nil || out != nil {
			t.Fatal("ambiguous/invalid prefix salvaged", raw, out, err)
		}
	}
}

func TestDocumentWhitespaceWatchdogPreservesAuthorWhitespaceInsideStrings(t *testing.T) {
	var state documentJSONWhitespace
	if state.push(`{"a":"`+strings.Repeat(" ", 8000)+`\`) || state.push(`"`+strings.Repeat(" ", 8000)+` done"`) {
		t.Fatal("authored string whitespace rejected")
	}
	if state.push(strings.Repeat("\n", 2048)) {
		t.Fatal("budget boundary")
	}
	if !state.push("\n") {
		t.Fatal("JSON whitespace loop not stopped")
	}
}
