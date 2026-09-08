package server

import (
	"context"
	"encoding/base64"
	"fmt"
	"net/http"
	"strings"
	"time"

	"sparktalk/internal/db"
	"sparktalk/internal/knowledge"
	"sparktalk/internal/llm"
)

const knowledgeOCRPrompt = `Transcribe every visible character from this document page exactly.
Preserve headings, paragraphs, table rows, numbers, units, punctuation, and the original language.
Do not summarize, translate, explain, or infer missing content.
Write [판독 불가] only where characters cannot be read.
Return plain text only.`

type knowledgeOCRRun struct {
	cancel context.CancelFunc
	done   chan struct{}
}

func (s *Server) ocrKnowledgeDocument(w http.ResponseWriter, _ *http.Request, document db.KnowledgeDocument) {
	if document.Status == "processing" && s.knowledgeOCRActive(document.ID) {
		writeJSON(w, http.StatusAccepted, document)
		return
	}
	if document.Status != "needs_ocr" && document.Status != "failed" && document.Status != "processing" {
		http.Error(w, "knowledge document does not require OCR", http.StatusConflict)
		return
	}
	if !knowledge.SupportsOCR(document.MIMEType) {
		detail := fmt.Sprintf("OCR does not support %s; the source must be imported again as a PDF or image", document.MIMEType)
		_ = s.db.UpdateKnowledgeDocumentStatus(document.ID, "failed", detail)
		http.Error(w, detail, http.StatusUnprocessableEntity)
		return
	}
	totalPages := document.OCRTotalPages
	if totalPages < 1 {
		totalPages = document.PageCount
	}
	if totalPages < 1 {
		totalPages = 1
	}
	resume := document.OCRProcessedPages > 0 && document.OCRProcessedPages < totalPages
	if err := s.db.BeginKnowledgeOCR(document.ID, totalPages, resume); err != nil {
		writeKnowledgeError(w, err)
		return
	}
	s.scheduleKnowledgeOCR(document.ID)
	updated, err := s.db.KnowledgeDocument(document.ID)
	if err != nil {
		writeKnowledgeError(w, err)
		return
	}
	writeJSON(w, http.StatusAccepted, updated)
}

func (s *Server) scheduleKnowledgeOCR(documentID string) {
	s.knowledgeOCRMu.Lock()
	if s.knowledgeOCRJobs == nil {
		s.knowledgeOCRJobs = make(map[string]*knowledgeOCRRun)
	}
	if s.knowledgeOCRSem == nil {
		s.knowledgeOCRSem = make(chan struct{}, 1)
	}
	if _, exists := s.knowledgeOCRJobs[documentID]; exists {
		s.knowledgeOCRMu.Unlock()
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	run := &knowledgeOCRRun{cancel: cancel, done: make(chan struct{})}
	s.knowledgeOCRJobs[documentID] = run
	s.knowledgeOCRMu.Unlock()
	go s.runKnowledgeOCR(ctx, documentID, run)
}

func (s *Server) knowledgeOCRActive(documentID string) bool {
	s.knowledgeOCRMu.Lock()
	defer s.knowledgeOCRMu.Unlock()
	_, active := s.knowledgeOCRJobs[documentID]
	return active
}

func (s *Server) stopKnowledgeOCR(documentID string, timeout time.Duration) bool {
	s.knowledgeOCRMu.Lock()
	run := s.knowledgeOCRJobs[documentID]
	s.knowledgeOCRMu.Unlock()
	if run == nil {
		return true
	}
	run.cancel()
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case <-run.done:
		return true
	case <-timer.C:
		return false
	}
}

func (s *Server) stopKnowledgeOCRs(documents []db.KnowledgeDocument, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for _, document := range documents {
		remaining := time.Until(deadline)
		if remaining <= 0 || !s.stopKnowledgeOCR(document.ID, remaining) {
			return false
		}
	}
	return true
}

func (s *Server) finishKnowledgeOCRRun(documentID string, run *knowledgeOCRRun) {
	s.knowledgeOCRMu.Lock()
	if current := s.knowledgeOCRJobs[documentID]; current == run {
		delete(s.knowledgeOCRJobs, documentID)
	}
	close(run.done)
	s.knowledgeOCRMu.Unlock()
}

func (s *Server) runKnowledgeOCR(ctx context.Context, documentID string, run *knowledgeOCRRun) {
	defer s.finishKnowledgeOCRRun(documentID, run)
	select {
	case s.knowledgeOCRSem <- struct{}{}:
		defer func() { <-s.knowledgeOCRSem }()
	case <-ctx.Done():
		return
	}
	document, err := s.db.KnowledgeDocument(documentID)
	if err != nil {
		return
	}
	path, err := s.knowledge.Path(document.StoragePath)
	if err != nil {
		_ = s.db.FinishKnowledgeOCR(documentID, "needs_ocr", compactKnowledgeError(err))
		return
	}
	cfg, client := s.snapshot()
	resumeFrom := document.OCRProcessedPages
	processed := resumeFrom
	pageCount, ocrErr := s.knowledgeIndex.RenderPages(path, document.MIMEType, func(page, total int, pngData []byte) error {
		if page <= resumeFrom {
			return nil
		}
		text, err := recognizeKnowledgePage(ctx, client, cfg.Model.DefaultModel, page, pngData)
		if err != nil {
			return err
		}
		chunks := knowledge.ChunkPages(documentID, []knowledge.Page{{Number: page, Text: text}})
		if len(chunks) == 0 {
			return fmt.Errorf("page %d: OCR returned no searchable text", page)
		}
		if err := s.db.ReplaceKnowledgeOCRPageChunks(documentID, page, chunks, total); err != nil {
			return fmt.Errorf("page %d: save OCR result: %w", page, err)
		}
		processed = page
		return nil
	})
	if ocrErr != nil {
		_ = s.db.FinishKnowledgeOCR(documentID, "needs_ocr", compactKnowledgeError(ocrErr))
		return
	}
	updated, err := s.db.KnowledgeDocument(documentID)
	if err != nil {
		return
	}
	if updated.ChunkCount == 0 || (processed < pageCount && resumeFrom < pageCount) {
		_ = s.db.FinishKnowledgeOCR(documentID, "needs_ocr", "OCR returned no searchable text")
		return
	}
	_ = s.db.FinishKnowledgeOCR(documentID, "ready", "")
}

func recognizeKnowledgePage(ctx context.Context, client *llm.Client, model string, page int, image []byte) (string, error) {
	mimeType := strings.TrimSpace(strings.Split(http.DetectContentType(image), ";")[0])
	if mimeType != "image/png" && mimeType != "image/jpeg" && mimeType != "image/webp" {
		return "", fmt.Errorf("page %d: renderer returned %s instead of an image", page, mimeType)
	}
	if client == nil {
		return "", fmt.Errorf("vision model is not configured")
	}
	dataURL := "data:" + mimeType + ";base64," + base64.StdEncoding.EncodeToString(image)
	messages := []llm.Message{
		{Role: "system", Content: knowledgeOCRPrompt},
		{Role: "user", Content: []map[string]any{
			{"type": "text", "text": fmt.Sprintf("Document page %d", page)},
			{"type": "image_url", "image_url": map[string]string{"url": dataURL}},
		}},
	}
	result, err := client.Stream(ctx, messages, model, "none", nil, func(string, string) error { return nil })
	if err != nil {
		return "", fmt.Errorf("page %d: %w", page, err)
	}
	text := knowledge.NormalizeOCRText(result.Content)
	if strings.TrimSpace(text) == "" {
		return "", fmt.Errorf("page %d: empty OCR response", page)
	}
	return text, nil
}
