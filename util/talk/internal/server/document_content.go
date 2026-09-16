package server

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"sparktalk/internal/db"
	"strings"
)

// Gather legacy image arrays and ordered block images through one resolver.
// The resolver still rejects paths, URLs, and user-supplied encoded bytes.
func documentImageScopes(root map[string]json.RawMessage) ([]map[string]json.RawMessage, func() error, error) {
	data, err := json.Marshal(root)
	if err != nil {
		return nil, nil, err
	}
	var tree map[string]any
	if err = json.Unmarshal(data, &tree); err != nil {
		return nil, nil, err
	}
	var scopes []map[string]json.RawMessage
	var setters []func(any)
	var walk func(any, int) error
	walk = func(value any, depth int) error {
		if depth > 12 {
			return fmt.Errorf("document nesting exceeds limit")
		}
		switch v := value.(type) {
		case map[string]any:
			for k, x := range v {
				if k == "images" {
					if list, ok := x.([]any); ok && len(list) > 0 {
						if first, ok := list[0].(map[string]any); ok {
							if _, wrapped := first["image"]; wrapped {
								if err := walk(x, depth+1); err != nil {
									return err
								}
								continue
							}
						}
					}
				}
				if k == "images" || k == "image" {
					raw := x
					if k == "image" {
						raw = []any{x}
					}
					b, e := json.Marshal(raw)
					if e != nil {
						return e
					}
					scopes = append(scopes, map[string]json.RawMessage{"images": b})
					parent, key := v, k
					setters = append(setters, func(items any) {
						if key == "image" {
							parent[key] = items.([]any)[0]
						} else {
							parent[key] = items
						}
					})
					continue
				}
				if err := walk(x, depth+1); err != nil {
					return err
				}
			}
		case []any:
			for _, x := range v {
				if err := walk(x, depth+1); err != nil {
					return err
				}
			}
		}
		return nil
	}
	if err = walk(tree, 0); err != nil {
		return nil, nil, err
	}
	commit := func() error {
		for i, s := range scopes {
			var items []any
			if e := json.Unmarshal(s["images"], &items); e != nil {
				return e
			}
			setters[i](items)
		}
		for k, v := range tree {
			b, e := json.Marshal(v)
			if e != nil {
				return e
			}
			root[k] = b
		}
		return nil
	}
	return scopes, commit, nil
}

func (s *Server) documentMediaPayload(sessionID string, root map[string]json.RawMessage) error {
	raw, ok := root["slides"]
	if !ok {
		return nil
	}
	var slides []map[string]any
	if err := json.Unmarshal(raw, &slides); err != nil {
		return err
	}
	var attachments map[string]db.Attachment
	count, total := 0, 0
	var walk func(any) error
	walk = func(value any) error {
		switch v := value.(type) {
		case map[string]any:
			if media, ok := v["media"]; ok {
				m, ok := media.(map[string]any)
				if !ok || len(m) != 1 {
					return fmt.Errorf("media에는 현재 대화의 media_id만 지정하세요")
				}
				id, ok := m["media_id"].(string)
				if !ok || id == "" || sessionID == "" {
					return fmt.Errorf("문서 미디어에는 현재 대화의 첨부 ID가 필요합니다")
				}
				count++
				if count > 3 {
					return fmt.Errorf("발표 미디어는 최대 3개입니다")
				}
				if attachments == nil {
					messages, err := s.db.Messages(sessionID)
					if err != nil {
						return err
					}
					attachments = map[string]db.Attachment{}
					for _, message := range messages {
						for _, a := range message.Attachments {
							attachments[a.ID] = a
						}
					}
				}
				a, ok := attachments[id]
				if !ok {
					return fmt.Errorf("현재 대화에서 사용할 수 없는 미디어입니다")
				}
				mime := documentMediaMIME(a.MIME)
				if mime != "audio/mpeg" && mime != "audio/wav" && mime != "video/mp4" {
					return fmt.Errorf("PPTX 미디어는 MP3/WAV/MP4만 지원합니다")
				}
				file, err := s.media.Open(a)
				if err != nil {
					return err
				}
				data, err := io.ReadAll(io.LimitReader(file, (8<<20)+1))
				file.Close()
				if err != nil {
					return err
				}
				total += len(data)
				if total > 8<<20 {
					return fmt.Errorf("발표 미디어 합계는 8 MiB 이하여야 합니다")
				}
				v["media"] = map[string]any{"data": base64.StdEncoding.EncodeToString(data), "mime": mime, "name": a.Name}
				return nil
			}
			for _, x := range v {
				if err := walk(x); err != nil {
					return err
				}
			}
		case []any:
			for _, x := range v {
				if err := walk(x); err != nil {
					return err
				}
			}
		}
		return nil
	}
	for _, slide := range slides {
		if err := walk(slide); err != nil {
			return err
		}
	}
	root["slides"], _ = json.Marshal(slides)
	return nil
}

func documentMediaCatalog(s *Server, sessionID string) string {
	messages, err := s.db.Messages(sessionID)
	if err != nil {
		return ""
	}
	seen := map[string]bool{}
	var lines []string
	for _, message := range messages {
		for _, item := range message.Attachments {
			if seen[item.ID] {
				continue
			}
			seen[item.ID] = true
			mime := documentMediaMIME(item.MIME)
			if mime == "audio/mpeg" || mime == "audio/wav" || mime == "video/mp4" {
				lines = append(lines, fmt.Sprintf("- media_id=%s, name=%q, mime=%s, bytes=%d", item.ID, item.Name, item.MIME, item.Size))
			}
		}
	}
	if len(lines) == 0 {
		return ""
	}
	sort.Strings(lines)
	return "\nAvailable PPTX media attachments (max 8 MiB total):\n" + strings.Join(lines, "\n")
}

func documentMediaMIME(value string) string {
	mime := strings.ToLower(strings.TrimSpace(strings.Split(value, ";")[0]))
	switch mime {
	case "audio/x-wav", "audio/wave":
		return "audio/wav"
	case "audio/mp3":
		return "audio/mpeg"
	}
	return mime
}
