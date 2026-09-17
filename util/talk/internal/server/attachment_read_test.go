package server

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"sparktalk/internal/db"
	"sparktalk/internal/knowledge"
	"sparktalk/internal/llm"
	"sparktalk/internal/media"
	"strings"
	"testing"
)

func TestAttachmentReadZIPPagesAndIsolation(t *testing.T) {
	s, _ := testImageServer(t)
	s.knowledgeIndex = &knowledge.Extractor{}
	var b bytes.Buffer
	w := zip.NewWriter(&b)
	f, _ := w.Create("src/main.go")
	f.Write([]byte(strings.Repeat("hello code\n", 2000) + "END_OF_FILE"))
	w.Close()
	a, e := s.media.SaveReader(bytes.NewReader(b.Bytes()), "source.zip", "application/zip", media.MaxAttachmentBytes)
	if e != nil {
		t.Fatal(e)
	}
	if _, e = s.db.AddMessage("session", "user", "read this", "", nil, []db.Attachment{a}); e != nil {
		t.Fatal(e)
	}
	reg := completionToolRegistry{handlers: make(map[string]registeredToolHandler)}
	s.registerAttachmentReader(&reg, "session")
	read := func(id string, offset int) map[string]any {
		args, _ := json.Marshal(map[string]any{"attachment_id": id, "offset": offset})
		v, e := reg.handlers["attachment_read"](context.Background(), llm.ToolCall{Function: llm.FunctionCall{Arguments: string(args)}}, nil, nil)
		if e != nil {
			t.Fatal(e)
		}
		var out map[string]any
		json.Unmarshal([]byte(v.Result), &out)
		return out
	}
	if len(read("", 0)["attachments"].([]any)) != 1 {
		t.Fatal("file list missing")
	}
	first := read(a.ID, 0)
	if first["has_more"] != true || !strings.Contains(first["content"].(string), "src/main.go") {
		t.Fatal(first)
	}
	next := read(a.ID, int(first["next_offset"].(float64)))
	if next["has_more"] != false || !strings.Contains(next["content"].(string), "END_OF_FILE") {
		t.Fatal("pagination lost tail")
	}
	other := completionToolRegistry{handlers: make(map[string]registeredToolHandler)}
	s.registerAttachmentReader(&other, "other")
	args, _ := json.Marshal(map[string]any{"attachment_id": a.ID})
	if _, e = other.handlers["attachment_read"](context.Background(), llm.ToolCall{Function: llm.FunctionCall{Arguments: string(args)}}, nil, nil); e == nil {
		t.Fatal("cross-session attachment exposed")
	}
}
