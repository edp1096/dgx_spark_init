package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/media"
	"sparktalk/internal/orchestrator"
)

func documentToolDefinition() llm.Tool {
	return llm.Tool{Type: "function", Function: llm.ToolFunction{Name: "document_generate", Description: "Create a downloadable DOCX report, PPTX presentation, XLSX spreadsheet, HWP/HWPX Korean document, or PDF. Office files include a separately rendered PDF when available; its layout may differ from the Office original. Warnings indicate PDF generation failure. Use only when the user wants an actual file. This tool cannot create HTML, CSS, JS or other source files. For a runnable web app, return complete html/css/javascript fenced code blocks for the artifact preview and HTML/ZIP download. Do not invent file_write. Supply finished document content, never code, paths or URLs. Send sections as a JSON array, not a JSON-encoded string. Titles/body may be Korean; output filenames use ASCII. For docx/pdf/hwp/hwpx provide sections, for pptx provide slides, for xlsx provide sheets with columns and rows. XLSX formulas are recalculated; use explicit formula objects and date objects. Use ordered blocks for rich content, tables, charts, images and mixed layouts. Existing image attachments work in all formats; PPTX also supports existing MP3/WAV/MP4 attachments. Use styles and page options instead of imitating tables with text. PPTX native tables accept table:{rows,widths,style} or table blocks; widths set relative column sizes. Cell objects also accept width in the same relative units; it sets the shared column width, and colspan widths sum the covered columns. Conflicting cell or table widths are rejected. Use returned presentation.slide_count and tables[].split/slide_numbers as facts, never guess. Attachment names are the actual output filenames. Set filename to an ASCII stem for a custom filename. Do not claim visual inspection from structure metadata. Follow format-specific block descriptions. PDF is a separate static preview; interactive features and complex layout may differ.", Parameters: orchestrator.DocumentToolSchema()}}
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
	switch strings.ToLower(filepath.Ext(input.Filename)) {
	case ".html", ".htm", ".css", ".js", ".mjs", ".ts", ".tsx", ".jsx", ".py", ".sh":
		return registeredToolResult{}, fmt.Errorf("document_generate는 코드 파일(%s)을 만들 수 없습니다. 웹 앱은 완성된 html/css/javascript 코드 블록으로 답변하면 아티팩트에서 실행·HTML/ZIP 다운로드할 수 있습니다. DOCX로 우회하거나 file_write를 호출하지 마세요", filepath.Ext(input.Filename))
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
		Warning      string          `json:"warning"`
		Text         string          `json:"text"`
		PageCount    int             `json:"page_count"`
		Presentation json.RawMessage `json:"presentation"`
		Files        []struct {
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
	result, _ := json.Marshal(map[string]any{"attachments": attachments, "presentation": output.Presentation, "verification": "Use attachment names and presentation metadata as generated facts. visually_verified=false means layout/legibility has not been visually inspected; do not claim you viewed the file.", "preview_available": input.Format == "pdf" || expected == 2, "warning": output.Warning, "preview": "PDF generated from the same content; layout may differ from the Office original."})
	return registeredToolResult{Result: string(result), Attachments: attachments}, nil
}
