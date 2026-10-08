package server

import (
	"encoding/json"
	"os"
	"sparktalk/internal/db"
	"sparktalk/internal/media"
	"strings"
	"testing"
)

func TestDocumentInitialTextBudget(t *testing.T) {
	text := strings.Repeat("x", 40000) + "LAST_FILE_END"
	block := documentAttachmentBlock(db.Attachment{ID: "zip"}, media.DocumentCache{Text: text})
	if !strings.Contains(block, text) || !strings.Contains(block, `truncated="false"`) {
		t.Fatal("small archive truncated")
	}
	block = documentAttachmentBlock(db.Attachment{ID: "zip"}, media.DocumentCache{Text: strings.Repeat("x", maxDocumentPromptRunes+1)})
	if !strings.Contains(block, `offset=256000`) || !strings.Contains(block, "attachment_read_required") {
		t.Fatal("missing explicit continuation")
	}
}
func TestUploadedZIPTextComplete(t *testing.T) {
	paths := os.Getenv("SPARKTALK_VERIFY_ZIP_CACHES")
	if paths == "" {
		t.Skip("opt-in read-only uploaded ZIP check")
	}
	for _, p := range strings.Split(paths, ":") {
		b, e := os.ReadFile(p)
		if e != nil {
			t.Fatal(e)
		}
		var c media.DocumentCache
		if e = json.Unmarshal(b, &c); e != nil {
			t.Fatal(e)
		}
		block := documentAttachmentBlock(db.Attachment{ID: "verify"}, c)
		if !strings.Contains(block, strings.TrimSpace(c.Text)) || !strings.Contains(block, `truncated="false"`) {
			t.Fatalf("uploaded ZIP incomplete: %s", p)
		}
		t.Logf("full extracted contents included: %d characters", len([]rune(c.Text)))
	}
}
