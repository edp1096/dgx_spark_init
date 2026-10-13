package server

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"unicode"

	"sparktalk/internal/db"
)

// Offline replay of an explicitly supplied backup. The live DB and inference
// endpoints are never opened; a fresh disposable DB receives every write.
func TestDocumentWhitespaceReplay(t *testing.T) {
	dir := os.Getenv("SPARKTALK_DOCUMENT_WHITESPACE_REPLAY")
	if dir == "" {
		t.Skip("opt-in offline completed-book export audit")
	}
	raw, err := os.ReadFile(filepath.Join(dir, "before.db"))
	if err != nil {
		t.Fatal(err)
	}
	private := filepath.Join(t.TempDir(), "replay.db")
	if err = os.WriteFile(private, raw, 0600); err != nil {
		t.Fatal(err)
	}
	database, err := db.Open(private)
	if err != nil {
		t.Fatal(err)
	}
	defer database.Close()
	s := &Server{db: database}
	reports := []map[string]any{}
	for _, ids := range [][2]string{
		{"chat-1791804532471788219", "chat-1791804593987680807"},
		{"chat-1791784517738091247", "chat-1791784571896538049"},
		{"chat-1791716007391613355", "chat-1791719302825162593"},
	} {
		session, id := ids[0], ids[1]
		run, err := database.DocumentRun(session, id)
		if err != nil || !run.DocumentFinished() {
			t.Fatal("unfinished fixture", id, err)
		}
		units, err := database.DocumentUnits(session, id)
		if err != nil {
			t.Fatal(err)
		}
		beforeUnits, _ := json.Marshal(units)
		beforeText, _, err := database.DocumentOutput(session, id, "output.txt")
		if err != nil {
			t.Fatal(err)
		}
		book, err := database.Source(session, run.Document.Sources[0], true)
		if err != nil {
			t.Fatal(err)
		}
		if book.Version != 4 {
			t.Fatal("source layout could not safely upgrade", id)
		}
		for _, a := range book.Assets {
			if a.Pending {
				t.Fatal("network asset download forbidden in offline replay")
			}
		}
		if err = s.exportDocument(context.Background(), run, units); err != nil {
			t.Fatal(id, err)
		}
		afterUnits, _ := database.DocumentUnits(session, id)
		afterEncoded, _ := json.Marshal(afterUnits)
		if string(beforeUnits) != string(afterEncoded) {
			t.Fatal("completed checkpoint changed", id)
		}
		outDir := filepath.Join(dir, id)
		if err = os.MkdirAll(outDir, 0700); err != nil {
			t.Fatal(err)
		}
		files := map[string]any{}
		for _, ext := range []string{"txt", "md", "html", "zip"} {
			name := "output." + ext
			data, mime, err := database.DocumentOutput(session, id, name)
			if err != nil {
				t.Fatal(err)
			}
			if err = os.WriteFile(filepath.Join(outDir, name), data, 0600); err != nil {
				t.Fatal(err)
			}
			hash := sha256.Sum256(data)
			files[name] = map[string]any{"bytes": len(data), "sha256": hex.EncodeToString(hash[:]), "mime": mime}
			if ext == "txt" && whitespaceFree(string(data)) != whitespaceFree(string(beforeText)) {
				t.Fatal("non-whitespace content changed", id)
			}
		}
		afterText, _, _ := database.DocumentOutput(session, id, "output.txt")
		reports = append(reports, map[string]any{"run": id, "session": session, "files": files, "before": whitespaceStats(string(beforeText)), "after": whitespaceStats(string(afterText)), "non_whitespace_content_unchanged": true, "checkpoints_unchanged": true, "model_calls": 0, "source_version": book.Version})
	}
	data, _ := json.MarshalIndent(reports, "", "  ")
	if err := os.WriteFile(filepath.Join(dir, "replay.json"), append(data, '\n'), 0600); err != nil {
		t.Fatal(err)
	}
}

func whitespaceFree(s string) string {
	return strings.Map(func(r rune) rune {
		if unicode.IsSpace(r) {
			return -1
		}
		return r
	}, s)
}

func whitespaceStats(s string) map[string]int {
	maxBreaks, breaks, runs := 0, 0, 0
	for _, r := range s {
		if r == '\n' {
			breaks++
			if breaks == 3 {
				runs++
			}
			maxBreaks = max(maxBreaks, breaks)
		} else if r != ' ' && r != '\t' && r != '\u3000' {
			breaks = 0
		}
	}
	return map[string]int{"lines": strings.Count(s, "\n"), "max_consecutive_newlines": maxBreaks, "three_plus_newline_runs": runs}
}
