package db

import (
	"path/filepath"
	"testing"
)

func TestTurnInputsPersistAndDoNotCrossEdits(t *testing.T) {
	d, err := Open(filepath.Join(t.TempDir(), "test.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	session, err := d.CreateSession("session", "test", "model", "none")
	if err != nil {
		t.Fatal(err)
	}
	m, err := d.AddPendingMessage(session.ID, "original", nil)
	if err != nil {
		t.Fatal(err)
	}
	input := TurnInput{ID: "test-input-123", Content: "use Korean"}
	added, err := d.SaveTurnInput(session.ID, m.ID, "original", input)
	if err != nil || !added {
		t.Fatal(added, err)
	}
	added, err = d.SaveTurnInput(session.ID, m.ID, "original", input)
	if err != nil || added {
		t.Fatal(added, err)
	}
	if _, err = d.SaveTurnInput(session.ID, m.ID, "original", TurnInput{ID: input.ID, Content: "changed"}); err == nil {
		t.Fatal("accepted conflicting replay")
	}
	if _, err = d.SaveTurnInput("another-session", m.ID, "original", TurnInput{ID: "other-input", Content: "bad"}); err == nil {
		t.Fatal("cross-session write accepted")
	}
	if _, err = d.SaveTurnInput("another-session", m.ID, "original", input); err == nil {
		t.Fatal("cross-session replay accepted")
	}
	inputs, err := d.TurnInputs(m.ID, "edited text")
	if err != nil || len(inputs) != 0 {
		t.Fatal(inputs, err)
	}
	messages, err := d.Messages(session.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(messages[0].TurnInputs) != 1 || len(messages[0].Variants[0].TurnInputs) != 1 || messages[0].Content != "original" {
		t.Fatalf("bad persisted turn: %+v", messages[0])
	}
}
