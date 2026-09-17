package server

import (
	"context"
	"fmt"
	"strings"

	"sparktalk/internal/db"
	"sparktalk/internal/media"
)

const (
	documentExtractionFingerprint = "sparktalk-document-v1"
	maxDocumentPromptRunes        = 256000
	maxDocumentCacheRunes         = 4_000_000
)

func isDocumentAttachment(item db.Attachment) bool {
	return !strings.HasPrefix(item.MIME, "image/") && !strings.HasPrefix(item.MIME, "audio/") && !strings.HasPrefix(item.MIME, "video/")
}

func (s *Server) extractDocumentAttachment(ctx context.Context, item db.Attachment) (media.DocumentCache, error) {
	if cached, ok, err := s.media.LoadDocument(item.ID, documentExtractionFingerprint); err == nil && ok {
		return cached, nil
	}
	s.documentMu.Lock()
	defer s.documentMu.Unlock()
	if cached, ok, err := s.media.LoadDocument(item.ID, documentExtractionFingerprint); err == nil && ok {
		return cached, nil
	}
	if err := ctx.Err(); err != nil {
		return media.DocumentCache{}, err
	}
	file, err := s.media.Open(item)
	if err != nil {
		return media.DocumentCache{}, fmt.Errorf("open document %s: %w", item.Name, err)
	}
	path := file.Name()
	file.Close()
	pages, err := s.knowledgeIndex.Extract(path, item.MIME)
	if err != nil {
		return media.DocumentCache{}, fmt.Errorf("extract document %s: %w", item.Name, err)
	}
	var output strings.Builder
	for _, page := range pages {
		text := strings.TrimSpace(page.Text)
		if text == "" {
			continue
		}
		fmt.Fprintf(&output, "[page %d]\n%s\n\n", page.Number, text)
		if output.Len() > maxDocumentCacheRunes*4 {
			break
		}
	}
	text, _ := truncateCollectedText(output.String(), maxDocumentCacheRunes)
	if strings.TrimSpace(text) == "" {
		return media.DocumentCache{}, fmt.Errorf("document %s contains no usable text", item.Name)
	}
	cached := media.DocumentCache{Fingerprint: documentExtractionFingerprint, Text: text, PageCount: len(pages)}
	if err := s.media.SaveDocument(item.ID, cached); err != nil {
		return media.DocumentCache{}, fmt.Errorf("cache document %s: %w", item.Name, err)
	}
	return cached, nil
}

func documentAttachmentBlock(item db.Attachment, cached media.DocumentCache) string {
	text, truncated := truncateCollectedText(cached.Text, maxDocumentPromptRunes)
	guidance := ""
	if truncated {
		guidance = fmt.Sprintf("\n<attachment_read_required>Only the first %d characters are shown. Read the remainder with attachment_read(attachment_id=%q, offset=%d) and follow next_offset until has_more=false before judging file completeness or contents. Truncation here is NOT a missing or corrupt file.</attachment_read_required>", maxDocumentPromptRunes, item.ID, maxDocumentPromptRunes)
	}
	return fmt.Sprintf("<document_attachment attachment_id=%q filename=%q type=%q pages=%q truncated=%q>\n%s\n</document_attachment>%s",
		item.ID, item.Name, item.MIME, fmt.Sprint(cached.PageCount), fmt.Sprint(truncated), text, guidance)
}
