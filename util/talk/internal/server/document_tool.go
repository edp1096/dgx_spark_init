package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"time"

	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/media"
)

func documentToolDefinition() llm.Tool {
	return llm.Tool{Type: "function", Function: llm.ToolFunction{Name: "document_generate", Description: "Create a downloadable DOCX report, PPTX presentation, XLSX spreadsheet, HWP/HWPX Korean document, or PDF. Office files include a separately rendered PDF when available; its layout may differ from the Office original. Warnings indicate PDF generation failure. Use only when the user wants an actual file. Supply finished content, never code, paths or URLs. Titles/body may be Korean; output filenames use ASCII. For docx/pdf/hwp/hwpx provide sections, for pptx provide slides, for xlsx provide sheets with columns and rows. XLSX formulas are recalculated; use explicit formula objects and date objects. Split long slides; Inline PNG/JPEG/WebP conversation attachments can be included in section images; arbitrary HTML is not supported.", Parameters: json.RawMessage(`{"type":"object","properties":{"format":{"type":"string","enum":["docx","pptx","xlsx","pdf","hwp","hwpx"]},"title":{"type":"string","maxLength":240},"filename":{"type":"string","description":"ASCII file stem without extension"},"sections":{"type":"array","maxItems":80,"items":{"type":"object","properties":{"heading":{"type":"string"},"paragraphs":{"type":"array","items":{"type":"string"}},"table":{"type":"array","description":"Rectangular rows; first row is the header. Up to 100 rows and 8 columns.","items":{"type":"array","items":{"type":"string"}}},"images":{"type":"array","maxItems":6,"description":"Inline images after this section table/body. DOCX/PDF/HWP/HWPX only. Use actual conversation attachment IDs.","items":{"type":"object","properties":{"image_id":{"type":"string"},"width_cm":{"type":"number","minimum":1,"maximum":16,"description":"Default 12 cm; aspect ratio preserved"},"caption":{"type":"string","maxLength":240}},"required":["image_id"],"additionalProperties":false}}},"required":["paragraphs"]}},"slides":{"type":"array","maxItems":40,"items":{"type":"object","properties":{"title":{"type":"string","maxLength":120},"bullets":{"type":"array","maxItems":8,"items":{"type":"string","maxLength":200}}},"required":["title","bullets"]}},"sheets":{"type":"array","minItems":1,"maxItems":8,"description":"XLSX only. Row 1 contains column titles; data starts at row 2. Up to 20000 cells total. Formulas can reference only supplied cells; use SUM, AVERAGE, COUNT, COUNTA, MIN, MAX, IF, AND, OR, NOT, ROUND, ROUNDUP, ROUNDDOWN, ABS, COUNTIF, SUMIF, IFERROR. AND/OR/NOT require individual cells or comparisons, not ranges. Strings are literal, not formulas.","items":{"type":"object","properties":{"name":{"type":"string","maxLength":31},"columns":{"type":"array","minItems":1,"maxItems":32,"items":{"type":"object","properties":{"title":{"type":"string"},"width":{"type":"number","minimum":6,"maximum":60},"format":{"type":"string","enum":["general","number","integer","currency","percent","date"]}},"required":["title"],"additionalProperties":false}},"rows":{"type":"array","maxItems":1000,"items":{"type":"array","items":{"anyOf":[{"type":"string"},{"type":"number"},{"type":"boolean"},{"type":"null"},{"type":"object","properties":{"formula":{"type":"string","maxLength":512}},"required":["formula"],"additionalProperties":false},{"type":"object","properties":{"date":{"type":"string","description":"YYYY-MM-DD"}},"required":["date"],"additionalProperties":false}]}}},"freeze_header":{"type":"boolean","description":"Default true"},"filter":{"type":"boolean","description":"Default true"},"orientation":{"type":"string","enum":["portrait","landscape"]}},"required":["name","columns","rows"],"additionalProperties":false}}},"required":["format","title"]}`)}}
}

var documentStem = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$`)

func (s *Server) executeDocumentGenerate(ctx context.Context, call llm.ToolCall) (registeredToolResult, error) {
	return s.executeDocumentGenerateForSession(ctx, "", call)
}

func (s *Server) executeDocumentGenerateForSession(ctx context.Context, sessionID string, call llm.ToolCall) (registeredToolResult, error) {
	if len(call.Function.Arguments) > 1<<20 {
		return registeredToolResult{}, fmt.Errorf("문서 요청이 너무 큽니다")
	}
	var input struct {
		Format   string `json:"format"`
		Filename string `json:"filename"`
	}
	if err := json.Unmarshal([]byte(call.Function.Arguments), &input); err != nil {
		return registeredToolResult{}, err
	}
	if input.Format != "docx" && input.Format != "pptx" && input.Format != "xlsx" && input.Format != "pdf" && input.Format != "hwp" && input.Format != "hwpx" {
		return registeredToolResult{}, fmt.Errorf("지원하지 않는 문서 형식입니다")
	}
	stem := input.Filename
	if !documentStem.MatchString(stem) {
		stem = "document"
	}
	payload, err := s.documentImagePayload(sessionID, call.Function.Arguments)
	if err != nil {
		return registeredToolResult{}, err
	}
	cfg, _ := s.snapshot()
	ctx, cancel := context.WithTimeout(ctx, 100*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(cfg.Extra.DocumentsEndpoint, "/")+"/v1/documents", bytes.NewReader(payload))
	if err != nil {
		return registeredToolResult{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	response, err := http.DefaultClient.Do(req)
	if err != nil {
		return registeredToolResult{}, fmt.Errorf("문서 생성 서비스: %w", err)
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, 70<<20))
	if err != nil {
		return registeredToolResult{}, err
	}
	if response.StatusCode != http.StatusOK {
		return registeredToolResult{}, fmt.Errorf("문서 생성 HTTP %d: %.1000s", response.StatusCode, body)
	}
	var output struct {
		Warning   string `json:"warning"`
		Text      string `json:"text"`
		PageCount int    `json:"page_count"`
		Files     []struct {
			Name string `json:"name"`
			MIME string `json:"mime"`
			Data string `json:"data"`
		} `json:"files"`
	}
	if err = json.Unmarshal(body, &output); err != nil {
		return registeredToolResult{}, err
	}
	expected := 1
	if input.Format != "pdf" {
		expected = 2
	}
	previewMissing := input.Format != "pdf" && len(output.Files) == 1 && strings.TrimSpace(output.Warning) != ""
	if previewMissing {
		expected = 1
	}
	if len(output.Files) != expected {
		return registeredToolResult{}, fmt.Errorf("생성 결과 파일이 누락되었습니다")
	}
	// Validate every file before persisting any of them.
	data := make([][]byte, expected)
	for i, f := range output.Files {
		ext := input.Format
		if input.Format == "pdf" || (expected == 2 && i == 1) {
			ext = "pdf"
		}
		if f.Name != "document."+ext {
			return registeredToolResult{}, fmt.Errorf("잘못된 생성 파일명")
		}
		data[i], err = base64.StdEncoding.DecodeString(f.Data)
		if err != nil || len(data[i]) == 0 || len(data[i]) > 24<<20 {
			return registeredToolResult{}, fmt.Errorf("잘못된 생성 파일 데이터")
		}
	}
	if (input.Format == "hwp" || input.Format == "hwpx") && (strings.TrimSpace(output.Text) == "" || len(output.Text) > 4*maxDocumentCacheRunes || output.PageCount < 1) {
		return registeredToolResult{}, fmt.Errorf("한글 문서의 텍스트 검증 결과가 누락되었습니다")
	}
	attachments := []db.Attachment{}
	for i, f := range output.Files {
		suffix := strings.TrimPrefix(f.Name, "document")
		name := stem + suffix
		if input.Format != "pdf" && expected == 2 && i == 1 {
			name = stem + "_preview.pdf"
		}
		item, saveErr := s.media.SaveReader(bytes.NewReader(data[i]), name, f.MIME, media.MaxAttachmentBytes)
		if saveErr != nil {
			return registeredToolResult{}, saveErr
		}
		if i == 0 && (input.Format == "hwp" || input.Format == "hwpx") {
			if err := s.media.SaveDocument(item.ID, media.DocumentCache{Fingerprint: documentExtractionFingerprint, Text: output.Text, PageCount: output.PageCount}); err != nil {
				return registeredToolResult{}, err
			}
		}
		attachments = append(attachments, item)
	}
	result, _ := json.Marshal(map[string]any{"attachments": attachments, "preview_available": input.Format == "pdf" || expected == 2, "warning": output.Warning, "preview": "PDF generated from the same content; layout may differ from the Office original."})
	return registeredToolResult{Result: string(result), Attachments: attachments}, nil
}
