package server

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sparktalk/internal/llm"
	"testing"
)

func TestDocumentToolPersistsOnlyCompleteResult(t *testing.T) {
	for _, broken := range []bool{false, true} {
		t.Run(map[bool]string{false: "success", true: "missing_output"}[broken], func(t *testing.T) {
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/v1/documents" {
					t.Error("wrong endpoint")
				}
				files := []map[string]string{{"name": "document.pdf", "mime": "application/pdf", "data": base64.StdEncoding.EncodeToString([]byte("%PDF-1.4\nvalidated renderer output"))}}
				if broken {
					files = nil
				}
				json.NewEncoder(w).Encode(map[string]any{"files": files})
			}))
			defer upstream.Close()
			s, _ := testImageServer(t)
			s.cfg.Extra.DocumentsEndpoint = upstream.URL
			s.cfg.Extra.DocumentsEnabled = true
			result, err := s.executeDocumentGenerate(context.Background(), llm.ToolCall{Function: llm.FunctionCall{Name: "document_generate", Arguments: `{"format":"pdf","filename":"../../한글","title":"test","sections":[{"paragraphs":["text"]}]}`}})
			if broken {
				if err == nil || len(result.Attachments) != 0 {
					t.Fatal("incomplete result exposed")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if len(result.Attachments) != 1 || result.Attachments[0].Name != "document.pdf" {
				t.Fatalf("unsafe filename or missing output: %+v", result)
			}
			reg := newCompletionToolRegistry(s, "session", s.cfg.Tools, false, nil)
			if reg.handlers["document_generate"] == nil {
				t.Fatal("tool missing")
			}
			s.cfg.Extra.DocumentsEnabled = false
			reg = newCompletionToolRegistry(s, "session", s.cfg.Tools, false, nil)
			if reg.handlers["document_generate"] != nil {
				t.Fatal("disabled tool exposed")
			}
		})
	}
}

func TestSpreadsheetPreviewOptionalOnlyWithWarning(t *testing.T) {
	for _, tc := range []struct {
		name                        string
		preview, warning, wantError bool
	}{
		{"with_preview", true, false, false}, {"preview_failed", false, true, false}, {"missing_without_warning", false, false, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var archive bytes.Buffer
			zw := zip.NewWriter(&archive)
			for name, content := range map[string]string{"[Content_Types].xml": "<Types/>", "xl/workbook.xml": "<workbook/>"} {
				f, err := zw.Create(name)
				if err != nil {
					t.Fatal(err)
				}
				f.Write([]byte(content))
			}
			if err := zw.Close(); err != nil {
				t.Fatal(err)
			}
			upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var input map[string]any
				json.NewDecoder(r.Body).Decode(&input)
				if input["format"] != "xlsx" {
					t.Error("XLSX format was lost")
				}
				files := []map[string]string{{"name": "document.xlsx", "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "data": base64.StdEncoding.EncodeToString(archive.Bytes())}}
				if tc.preview {
					files = append(files, map[string]string{"name": "document.pdf", "mime": "application/pdf", "data": base64.StdEncoding.EncodeToString([]byte("%PDF-1.4\npreview"))})
				}
				result := map[string]any{"files": files}
				if tc.warning {
					result["warning"] = "PDF conversion failed; original available"
				}
				json.NewEncoder(w).Encode(result)
			}))
			defer upstream.Close()
			s, _ := testImageServer(t)
			s.cfg.Extra.DocumentsEndpoint = upstream.URL
			result, err := s.executeDocumentGenerate(context.Background(), llm.ToolCall{Function: llm.FunctionCall{Name: "document_generate", Arguments: `{"format":"xlsx","title":"매출","filename":"sales","sheets":[]}`}})
			if tc.wantError {
				if err == nil || len(result.Attachments) != 0 {
					t.Fatal("silent missing preview accepted")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			want := 1
			if tc.preview {
				want = 2
			}
			if len(result.Attachments) != want || result.Attachments[0].Name != "sales.xlsx" {
				t.Fatalf("wrong attachments: %+v", result.Attachments)
			}
			if tc.preview && result.Attachments[1].Name != "sales_preview.pdf" {
				t.Fatal("wrong preview name")
			}
			var payload map[string]any
			json.Unmarshal([]byte(result.Result), &payload)
			if payload["preview_available"] != tc.preview {
				t.Fatal("incorrect preview state")
			}
		})
	}
}

func TestGeneratedHWPTextIsAvailableForLaterConversation(t *testing.T) {
	body := append([]byte{0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1}, []byte("HWP Document File\x00")...)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"files": []map[string]string{{"name": "document.hwp", "mime": "application/x-hwp", "data": base64.StdEncoding.EncodeToString(body)}}, "warning": "PDF unavailable", "text": "한글 본문과 표 내용", "page_count": 1})
	}))
	defer upstream.Close()
	s, _ := testImageServer(t)
	s.cfg.Extra.DocumentsEndpoint = upstream.URL
	result, err := s.executeDocumentGenerate(context.Background(), llm.ToolCall{Function: llm.FunctionCall{Name: "document_generate", Arguments: `{"format":"hwp","title":"한글","filename":"korean","sections":[{"paragraphs":["본문"]}]}`}})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Attachments) != 1 || result.Attachments[0].MIME != "application/x-hwp" {
		t.Fatal("HWP attachment missing")
	}
	cached, err := s.extractDocumentAttachment(context.Background(), result.Attachments[0])
	if err != nil {
		t.Fatal(err)
	}
	if cached.Text != "한글 본문과 표 내용" || cached.PageCount != 1 {
		t.Fatal("Generated HWP text was lost")
	}
}
