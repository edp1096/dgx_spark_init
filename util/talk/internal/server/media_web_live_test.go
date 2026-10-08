package server

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/webtools"
)

// Opt-in: actual public search, page fetch, image import, and live DocMS render.
func TestWebImagePPTXLive(t *testing.T) {
	out := os.Getenv("SPARKTALK_WEB_IMAGE_TEST_OUT")
	if out == "" {
		t.Skip("live external services")
	}
	if err := os.MkdirAll(out, 0755); err != nil {
		t.Fatal(err)
	}
	s, _ := testImageServer(t)
	s.cfg.Extra.DocumentsEnabled = true
	s.cfg.Extra.DocumentsEndpoint = "http://127.0.0.1:8696"
	ctx := context.Background()
	runner := webtools.New(8, 30*time.Second)
	conversation := []llm.Message{{Role: "user", Content: "웹에서 숭례문 사진을 찾아 PPTX에 넣어라."}}
	web := func(name, args, id string) string {
		raw, err := runner.Execute(ctx, name, args)
		if err != nil {
			t.Fatal(err)
		}
		conversation = append(conversation, llm.Message{Role: "assistant", ToolCalls: []llm.ToolCall{{ID: id, Function: llm.FunctionCall{Name: name, Arguments: args}}}}, llm.Message{Role: "tool", ToolCallID: id, Content: raw})
		os.WriteFile(filepath.Join(out, id+".json"), []byte(raw), 0644)
		return raw
	}
	raw := web("web_search", `{"query":"Sungnyemun Wikimedia Commons photo"}`, "search")
	var search struct {
		Results []struct {
			URL string `json:"url"`
		} `json:"results"`
	}
	json.Unmarshal([]byte(raw), &search)
	if len(search.Results) == 0 {
		t.Fatal("no search results")
	}
	var imported mediaToolExecution
	for i, r := range search.Results {
		args, _ := json.Marshal(map[string]string{"url": r.URL})
		page, err := runner.Execute(ctx, "web_fetch", string(args))
		if err != nil {
			t.Log(err)
			continue
		}
		id := string(rune('a' + i))
		conversation = append(conversation, llm.Message{Role: "assistant", ToolCalls: []llm.ToolCall{{ID: id, Function: llm.FunctionCall{Name: "web_fetch"}}}}, llm.Message{Role: "tool", ToolCallID: id, Content: page})
		var fetched struct {
			Images []webtools.ImageSource `json:"images"`
		}
		json.Unmarshal([]byte(page), &fetched)
		for _, im := range fetched.Images {
			if !strings.Contains(strings.ToLower(im.URL), "sungnyemun") && !strings.Contains(strings.ToLower(im.URL), "namdaemun") {
				continue
			}
			arg, _ := json.Marshal(map[string]string{"url": im.URL})
			imported, err = s.executeMediaImportTool(ctx, llm.ToolCall{Function: llm.FunctionCall{Name: "media_import", Arguments: string(arg)}}, conversation)
			if err != nil {
				t.Log(err)
				continue
			}
			break
		}
		if imported.Attachment.ID != "" {
			os.WriteFile(filepath.Join(out, "page.json"), []byte(page), 0644)
			break
		}
	}
	if imported.Attachment.ID == "" {
		t.Fatal("no usable photo imported")
	}
	os.WriteFile(filepath.Join(out, "import.json"), []byte(imported.Result), 0644)
	if _, err := s.db.AddMessage("session", "user", "검색한 사진", "", nil, []db.Attachment{imported.Attachment}); err != nil {
		t.Fatal(err)
	}
	args, _ := json.Marshal(map[string]any{"format": "pptx", "filename": "web_photo_validation", "title": "숭례문", "slides": []any{map[string]any{"title": "웹에서 찾은 숭례문 사진", "images": []any{map[string]any{"image_id": imported.Attachment.ID, "width_cm": 12, "caption": imported.Attachment.SourceURL}}}}})
	result, err := s.executeDocumentGenerateForSession(ctx, "session", llm.ToolCall{Function: llm.FunctionCall{Name: "document_generate", Arguments: string(args)}})
	if err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(out, "document.json"), []byte(result.Result), 0644)
	found := false
	for _, a := range result.Attachments {
		f, err := s.media.Open(a)
		if err != nil {
			t.Fatal(err)
		}
		data, err := io.ReadAll(f)
		f.Close()
		if err != nil {
			t.Fatal(err)
		}
		os.WriteFile(filepath.Join(out, a.Name), data, 0644)
		if strings.HasSuffix(a.Name, ".pptx") {
			z, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
			if err != nil {
				t.Fatal(err)
			}
			for _, entry := range z.File {
				if strings.HasPrefix(entry.Name, "ppt/media/") && entry.UncompressedSize64 > 100 {
					found = true
				}
			}
		}
	}
	if !found {
		t.Fatal("PPTX contains no embedded image")
	}
	t.Log("actual search -> page -> image import -> embedded PPTX image passed", imported.Attachment.SourceURL)
}
