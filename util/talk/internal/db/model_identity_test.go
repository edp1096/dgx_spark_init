package db

import (
	"path/filepath"
	"sparktalk/internal/modelidentity"
	"testing"
)

func TestQwen38FNEXL3MigratesStoredIDsWithoutRewritingConversation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "chat.db")
	d, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	_, err = d.CreateSession("old", "My chat", "temporary", "medium")
	if err != nil {
		t.Fatal(err)
	}
	_, err = d.CreateSession("external", "External", "other/custom-model", "low")
	if err != nil {
		t.Fatal(err)
	}
	content := "A historical quotation: " + modelidentity.LegacyQwenModel
	m, err := d.AddMessage("old", "assistant", content, "", nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	_, err = d.AddContextSegment("old", m.ID, m.ID, "summary", "", "temporary", 10)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = d.conn.Exec("UPDATE sessions SET model=? WHERE id='old'", modelidentity.LegacyQwenModel); err != nil {
		t.Fatal(err)
	}
	if _, err = d.conn.Exec("UPDATE context_segments SET model=?", modelidentity.LegacyQwenModel); err != nil {
		t.Fatal(err)
	}
	d.Close()
	d, err = Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	s, err := d.Session("old")
	if err != nil || s.Model != modelidentity.Qwen38FNEXL3 || s.Reasoning != "medium" {
		t.Fatal(s, err)
	}
	segments, err := d.ContextSegments("old")
	if err != nil || segments[0].Model != modelidentity.Qwen38FNEXL3 {
		t.Fatal(segments, err)
	}
	messages, err := d.Messages("old")
	if err != nil || messages[0].Content != content {
		t.Fatal("historical message was rewritten", err)
	}
	s, err = d.Session("external")
	if err != nil || s.Model != "other/custom-model" {
		t.Fatal("external model changed")
	}
}
