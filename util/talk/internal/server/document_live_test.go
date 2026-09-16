package server

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sparktalk/internal/llm"
	"strings"
	"testing"
)

// Optional end-to-end check: Talk's argument normalization, attachment resolver,
// real document service, and returned downloadable files. No LLM is needed.
func TestLiveDocumentTableAndImage(t *testing.T) {
	endpoint := os.Getenv("SPARKTALK_DOCMS_TEST_URL")
	if endpoint == "" {
		t.Skip("set SPARKTALK_DOCMS_TEST_URL for live renderer verification")
	}
	s, image := testImageServer(t)
	s.cfg.Extra.DocumentsEndpoint = endpoint
	args := fmt.Sprintf(`{"format":"pptx","title":"실제 표 검증","filename":"table_check","slides":[{"title":"표 생성","blocks":[{"type":"table","rows":[[{"text":"항목","width":1},{"text":"값","width":4}],["검증","성공"]]},{"type":"image","image":{"image_id":%q,"width_cm":2,"caption":"첨부 이미지"}}]}]}`, image.ID)
	result, err := s.executeDocumentGenerateForSession(context.Background(), "session", llm.ToolCall{ID: "live-doc", Function: llm.FunctionCall{Name: "document_generate", Arguments: args}})
	if err != nil {
		t.Fatal(err)
	}
	var metadata struct {
		Presentation struct {
			SlideCount int `json:"slide_count"`
			Tables     []struct {
				Split bool `json:"split"`
			} `json:"tables"`
			VisuallyVerified bool `json:"visually_verified"`
		} `json:"presentation"`
	}
	if err := json.Unmarshal([]byte(result.Result), &metadata); err != nil {
		t.Fatal(err)
	}
	if metadata.Presentation.SlideCount != 1 || len(metadata.Presentation.Tables) != 1 || metadata.Presentation.Tables[0].Split || metadata.Presentation.VisuallyVerified {
		t.Fatalf("Invalid structure report: %s", result.Result)
	}
	if len(result.Attachments) != 2 {
		t.Fatalf("Expected PPTX and PDF: %+v", result.Attachments)
	}
	file, err := s.media.Open(result.Attachments[0])
	if err != nil {
		t.Fatal(err)
	}
	data, err := io.ReadAll(file)
	file.Close()
	if err != nil {
		t.Fatal(err)
	}
	z, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	table, picture := false, false
	for _, f := range z.File {
		if strings.HasPrefix(f.Name, "ppt/media/") {
			picture = true
		}
		if strings.HasPrefix(f.Name, "ppt/slides/slide") && strings.HasSuffix(f.Name, ".xml") {
			r, e := f.Open()
			if e != nil {
				t.Fatal(e)
			}
			xml, _ := io.ReadAll(r)
			r.Close()
			if bytes.Contains(xml, []byte("<a:tbl>")) {
				table = true
			}
		}
	}
	if !table || !picture {
		t.Fatalf("Missing real table or image: table=%v image=%v", table, picture)
	}
}

func TestLiveDocumentEncodedSections(t *testing.T) {
	endpoint := os.Getenv("SPARKTALK_DOCMS_TEST_URL")
	if endpoint == "" {
		t.Skip("set SPARKTALK_DOCMS_TEST_URL")
	}
	s, _ := testImageServer(t)
	s.cfg.Extra.DocumentsEndpoint = endpoint
	sections := `[{"heading":"입력 검증","paragraphs":["한글 본문 7~8시를 보존합니다."]}]`
	args, _ := json.Marshal(map[string]any{"format": "docx", "title": "형식 검증", "filename": "sections_check", "sections": sections})
	result, err := s.executeDocumentGenerateForSession(context.Background(), "session", llm.ToolCall{Function: llm.FunctionCall{Arguments: string(args)}})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Attachments) != 2 {
		t.Fatalf("expected docx and preview, got %d", len(result.Attachments))
	}
	f, err := s.media.Open(result.Attachments[0])
	if err != nil {
		t.Fatal(err)
	}
	data, err := io.ReadAll(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	archive, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range archive.File {
		if entry.Name != "word/document.xml" {
			continue
		}
		r, err := entry.Open()
		if err != nil {
			t.Fatal(err)
		}
		body, err := io.ReadAll(r)
		r.Close()
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Contains(body, []byte("한글 본문 7~8시를 보존합니다.")) {
			t.Fatal("document content missing")
		}
		return
	}
	t.Fatal("missing document XML")
}
