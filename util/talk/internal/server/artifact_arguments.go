package server

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"sparktalk/internal/db"
)

type codeProjectArguments struct {
	Action  string
	ID      string
	Title   string
	Base    int
	Version int
	Summary string
	Files   []db.ArtifactFile
	Edits   []db.ArtifactEdit
}

// Presence is significant: an omitted payload must never become an empty file.
// source is accepted for create/write edits as well as for initial files.
func decodeCodeProjectArguments(raw string) (codeProjectArguments, error) {
	var out codeProjectArguments
	var wire struct {
		Action  string `json:"action"`
		ID      string `json:"project_id"`
		Title   string `json:"title"`
		Base    int    `json:"base_version"`
		Version int    `json:"version"`
		Summary string `json:"summary"`
		Files   []struct {
			Name   string  `json:"name"`
			Source *string `json:"source"`
		} `json:"files"`
		Edits []struct {
			Name      string  `json:"name"`
			Operation string  `json:"operation"`
			Old       *string `json:"old"`
			New       *string `json:"new"`
			Source    *string `json:"source"`
		} `json:"edits"`
	}
	dec := json.NewDecoder(strings.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&wire); err != nil {
		return out, fmt.Errorf("invalid code_project arguments (nothing saved): %w", err)
	}
	if err := dec.Decode(new(any)); err != io.EOF {
		return out, fmt.Errorf("code_project expects exactly one JSON object; nothing saved")
	}
	out = codeProjectArguments{Action: wire.Action, ID: wire.ID, Title: wire.Title, Base: wire.Base, Version: wire.Version, Summary: wire.Summary}
	switch wire.Action {
	case "create":
		if len(wire.Edits) > 0 {
			return out, fmt.Errorf("create uses files, not edits; nothing saved")
		}
		if len(wire.Files) == 0 {
			return out, fmt.Errorf("create requires files with name and source; nothing saved")
		}
	case "edit":
		if len(wire.Files) > 0 {
			return out, fmt.Errorf("edit uses edits, not files; use operation write with source for full file replacement; nothing saved")
		}
		if len(wire.Edits) == 0 {
			return out, fmt.Errorf("edit requires edits; nothing saved")
		}
	case "restore":
		if len(wire.Files) > 0 || len(wire.Edits) > 0 {
			return out, fmt.Errorf("restore takes version and base_version only; nothing saved")
		}
	case "list", "read", "history":
		if len(wire.Files) > 0 || len(wire.Edits) > 0 {
			return out, fmt.Errorf("read-only action cannot include files or edits; nothing saved")
		}
	default:
		return out, fmt.Errorf("unknown code_project action %q", wire.Action)
	}
	if wire.Action != "create" && wire.Action != "list" && wire.ID == "" {
		return out, fmt.Errorf("project_id is required")
	}
	if (wire.Action == "edit" || wire.Action == "restore") && wire.Base < 1 {
		return out, fmt.Errorf("base_version from the latest read is required; nothing saved")
	}
	for _, file := range wire.Files {
		if file.Source == nil {
			return out, fmt.Errorf("file %q requires source (missing or null); nothing saved", file.Name)
		}
		out.Files = append(out.Files, db.ArtifactFile{Name: file.Name, Source: *file.Source})
	}
	for _, edit := range wire.Edits {
		e := db.ArtifactEdit{Name: edit.Name, Operation: edit.Operation}
		if edit.New != nil && edit.Source != nil {
			return out, fmt.Errorf("%s: provide only one of new or source; nothing saved", edit.Name)
		}
		switch edit.Operation {
		case "delete":
			if edit.New != nil || edit.Source != nil || edit.Old != nil {
				return out, fmt.Errorf("delete takes only name and operation; nothing saved")
			}
		case "create", "write":
			value := edit.Source
			if value == nil {
				value = edit.New
			}
			if value == nil {
				return out, fmt.Errorf("%s requires source or new, including an explicit string; nothing saved", edit.Operation)
			}
			if edit.Old != nil {
				return out, fmt.Errorf("%s does not take old; use replace for a partial edit", edit.Operation)
			}
			e.New = *value
		case "replace":
			if edit.Old == nil || edit.New == nil || edit.Source != nil {
				return out, fmt.Errorf("replace requires old and new; for a full file rewrite use operation write with source; nothing saved")
			}
			e.Old = *edit.Old
			e.New = *edit.New
		default:
			return out, fmt.Errorf("operation is required: create, write, replace or delete; nothing saved")
		}
		out.Edits = append(out.Edits, e)
	}
	return out, nil
}
