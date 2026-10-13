package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"sparktalk/internal/llm"
	"sparktalk/internal/workflows"
)

// Recover only completed string fields from a syntactically valid JSON prefix.
// Unknown IDs, duplicate aliases, wrong types or observed syntax errors discard
// the entire response. No missing quote/value is guessed or repaired.
func documentCompletePrefix(u workflows.DocumentUnit, raw string) (map[string]string, error) {
	d := json.NewDecoder(strings.NewReader(raw))
	eof := func(err error) bool { return errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) }
	t, err := d.Token()
	if err != nil || t != json.Delim('{') {
		return nil, fmt.Errorf("not a JSON object prefix")
	}
	allowed := map[string]string{}
	for _, e := range documentWireEntries(u) {
		original := u.Inputs[e.Indexes[0]].ID
		allowed[e.ID] = original
		allowed[original] = original
		allowed[fmt.Sprintf("p%d", e.Indexes[0]+1)] = original
	}
	seenRoot, seenIDs := map[string]bool{}, map[string]bool{}
	fields := map[string]json.RawMessage{}
	finish := func() (map[string]string, error) {
		if len(fields) == 0 {
			return nil, fmt.Errorf("no complete translation field")
		}
		data, _ := json.Marshal(map[string]any{"entries": fields})
		out, err := validateDocumentOutput(u, "translate", llm.StreamResult{Content: string(data), FinishReason: "stop"})
		if out == nil {
			return nil, err
		}
		return out, nil
	}
	valuePrefix := func() string {
		tail := strings.TrimSpace(raw[d.InputOffset():])
		if strings.HasPrefix(tail, ":") {
			tail = strings.TrimSpace(tail[1:])
		}
		return tail
	}
	for d.More() {
		k, e := d.Token()
		if eof(e) {
			return finish()
		}
		if e != nil {
			return nil, e
		}
		key, ok := k.(string)
		if !ok || seenRoot[key] || (key != "entries" && key != "terms") {
			return nil, fmt.Errorf("invalid or duplicate prefix envelope key")
		}
		seenRoot[key] = true
		if key == "terms" {
			if tail := valuePrefix(); tail != "" && tail[0] != '[' {
				return nil, fmt.Errorf("invalid terms in prefix")
			}
			var terms json.RawMessage
			if e = d.Decode(&terms); eof(e) {
				return finish()
			}
			if e != nil {
				return nil, e
			}
			continue
		}
		open, e := d.Token()
		if eof(e) {
			return finish()
		}
		if e != nil || open != json.Delim('{') {
			return nil, fmt.Errorf("prefix entries must be an object")
		}
		for d.More() {
			k, e := d.Token()
			if eof(e) {
				return finish()
			}
			if e != nil {
				return nil, e
			}
			id, ok := k.(string)
			canonical := allowed[id]
			if !ok || canonical == "" || seenIDs[canonical] {
				return nil, fmt.Errorf("unexpected or duplicate prefix output id")
			}
			seenIDs[canonical] = true
			if tail := valuePrefix(); tail != "" && tail[0] != '"' {
				return nil, fmt.Errorf("prefix value must be a string")
			}
			var text json.RawMessage
			if e = d.Decode(&text); eof(e) {
				return finish()
			}
			if e != nil {
				return nil, e
			}
			if len(text) == 0 || text[0] != '"' {
				return nil, fmt.Errorf("prefix value must be a string")
			}
			fields[id] = text
		}
		close, e := d.Token()
		if eof(e) {
			return finish()
		}
		if e != nil || close != json.Delim('}') {
			return nil, fmt.Errorf("invalid entries prefix")
		}
	}
	close, e := d.Token()
	if eof(e) {
		return finish()
	}
	if e != nil || close != json.Delim('}') {
		return nil, fmt.Errorf("invalid envelope prefix")
	}
	if _, e = d.Token(); e != io.EOF {
		return nil, fmt.Errorf("unexpected trailing prefix data")
	}
	return finish()
}
