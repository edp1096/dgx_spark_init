package server

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// Reject ambiguous envelope keys before permitting any partial-result recovery.
func documentEntriesJSON(raw string) (json.RawMessage, error) {
	decoder := json.NewDecoder(strings.NewReader(raw))
	first, err := decoder.Token()
	if err != nil || first != json.Delim('{') {
		return nil, fmt.Errorf("document result must be one JSON object")
	}
	seen := map[string]bool{}
	var entries json.RawMessage
	for decoder.More() {
		key, err := decoder.Token()
		if err != nil {
			return nil, err
		}
		name, ok := key.(string)
		if !ok || seen[name] {
			return nil, fmt.Errorf("duplicate or invalid result field %q", name)
		}
		seen[name] = true
		var value json.RawMessage
		if err = decoder.Decode(&value); err != nil {
			return nil, err
		}
		if name == "entries" {
			entries = value
		}
	}
	if _, err = decoder.Token(); err != nil {
		return nil, err
	}
	if _, err = decoder.Token(); err != io.EOF {
		return nil, fmt.Errorf("trailing data after document result")
	}
	if len(entries) == 0 {
		return nil, fmt.Errorf("document result has no entries")
	}
	return entries, nil
}
