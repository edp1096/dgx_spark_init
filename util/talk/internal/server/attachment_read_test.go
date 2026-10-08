package server

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"image"
	"image/jpeg"
	"net/http"
	"net/http/httptest"
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
	if len(read("", 0)["attachments"].([]any)) != 2 {
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

func TestAttachmentReadListsAndReloadsStoredVideo(t *testing.T) {
	s, _ := testImageServer(t)
	a, err := s.media.SaveReader(bytes.NewReader(append([]byte{0, 0, 0, 12}, []byte("ftypisomvideo")...)), "clip.mp4", "video/mp4", media.MaxAttachmentBytes)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = s.db.AddMessage("session", "user", "analyze", "", nil, []db.Attachment{a}); err != nil {
		t.Fatal(err)
	}
	frames := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		jpeg.Encode(w, image.NewRGBA(image.Rect(0, 0, 16, 8)), nil)
	}))
	defer frames.Close()
	s.cfg.Model.VideoInputs = map[string]string{s.cfg.Model.Endpoint + "\n" + s.cfg.Model.DefaultModel: "frames"}
	s.cfg.ASR.Enabled = false
	s.cfg.ASR.FFmpegEndpoint = frames.URL
	reg := completionToolRegistry{handlers: make(map[string]registeredToolHandler)}
	s.registerAttachmentReader(&reg, "session")
	call := func(args string) (registeredToolResult, error) {
		return reg.handlers["attachment_read"](context.Background(), llm.ToolCall{Function: llm.FunctionCall{Arguments: args}}, nil, nil)
	}
	listed, err := call(`{}`)
	if err != nil || !strings.Contains(listed.Result, a.ID) {
		t.Fatalf("video omitted: %s %v", listed.Result, err)
	}
	args, _ := json.Marshal(map[string]string{"attachment_id": a.ID})
	loaded, err := call(string(args))
	if err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(loaded.Followups)
	if strings.Contains(string(b), "video_url") || !strings.Contains(string(b), "data:image/jpeg;base64,") || !strings.Contains(string(b), a.ID) {
		t.Fatalf("media reload payload incorrect: %s", b)
	}
}
