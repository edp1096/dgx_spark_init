package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"

	"sparktalk/internal/workflows"
)

// Conversion consumes committed task text, never a model-generated giant tool
// argument. Illustrations/layout are preserved by HTML; this is text export.
func (s *Server) convertDocumentTask(ctx context.Context, run workflows.Run, format string) (resultErr error) {
	if !s.docExportMu.TryLock() {
		return fmt.Errorf("document conversion already running")
	}
	defer s.docExportMu.Unlock()
	latest, err := s.db.DocumentRun(run.SessionID, run.ID)
	if err != nil {
		return err
	}
	run = latest
	if format != "docx" && format != "pdf" && format != "hwp" && format != "hwpx" {
		return fmt.Errorf("unsupported document format")
	}
	if run.Current < 2 {
		return fmt.Errorf("task output is not complete")
	}
	jobs, err := s.db.DocumentConversions(run.SessionID, run.ID)
	if err != nil {
		return err
	}
	for _, job := range jobs {
		if job.Current && job.Format == format {
			return nil
		}
	}
	if _, _, e := s.db.ConvertedDocumentOutput(run.SessionID, run.ID, "output."+format); e == nil {
		has := false
		for _, v := range run.Document.Formats {
			if v == format {
				has = true
			}
		}
		if !has {
			run.Document.Formats = append(run.Document.Formats, format)
			return s.saveDocumentConvertedFormats(&run)
		}
		if run.Document.Pipeline != "" {
			return s.saveDocumentConvertedFormats(&run)
		}
		return nil
	}
	text, _, e := s.db.DocumentOutput(run.SessionID, run.ID, "output.txt")
	if e != nil {
		return e
	}
	result, e := s.requestDocumentConversion(ctx, text, format)
	if e != nil {
		return e
	}
	found := false
	for _, f := range result.Files {
		ext := format
		if f.Name == "document.pdf" {
			ext = "pdf"
		}
		if f.Name != "document."+ext {
			return fmt.Errorf("unexpected converted filename")
		}
		data, err := base64.StdEncoding.DecodeString(f.Data)
		if err != nil || len(data) == 0 || len(data) > 24<<20 {
			return fmt.Errorf("invalid converted document data")
		}
		if ext == format {
			found = true
		}
		manifest := run.Document.Publication
		if manifest == nil {
			manifest = run.Document.DraftPublication
		}
		if ext != format && manifest != nil {
			published := false
			for _, file := range manifest.Files {
				published = published || file.Name == "output."+ext
			}
			// A converter may return a newly timestamped PDF alongside DOCX/HWP.
			// The existing PDF already belongs to this immutable text revision.
			if published {
				continue
			}
		}
		if err = s.db.SaveDocumentOutput(run.SessionID, run.ID, "output."+ext, f.MIME, data); err != nil {
			return err
		}
		has := false
		for _, v := range run.Document.Formats {
			if v == ext {
				has = true
			}
		}
		if !has {
			run.Document.Formats = append(run.Document.Formats, ext)
		}
	}
	if !found {
		return fmt.Errorf("requested document format missing")
	}
	warnings := run.Document.Warnings[:0]
	for _, w := range run.Document.Warnings {
		if !strings.HasPrefix(w, "requested "+format+" conversion failed:") {
			warnings = append(warnings, w)
		}
	}
	run.Document.Warnings = warnings
	if run.Document.Pipeline == "" && len(warnings) == 0 && run.Status == "needs_review" && run.Current >= 2 {
		run.Status = "completed"
	}
	if result.Warning != "" {
		run.Document.Warnings = append(run.Document.Warnings, result.Warning)
	}
	return s.saveDocumentConvertedFormats(&run)
}

func (s *Server) saveDocumentConvertedFormats(run *workflows.Run) error {
	if run.Document.Pipeline != "" && run.Document.Publication != nil {
		names := []string{}
		for _, file := range run.Document.Publication.Files {
			names = append(names, file.Name)
		}
		for _, format := range run.Document.Formats {
			names = append(names, "output."+format)
		}
		return s.db.PublishDocumentRevision(run, names, run.Document.Publication.TextHash)
	}
	if run.Document.Pipeline != "" && run.Document.DraftPublication != nil {
		names := []string{}
		for _, file := range run.Document.DraftPublication.Files {
			names = append(names, file.Name)
		}
		for _, format := range run.Document.Formats {
			names = append(names, "output."+format)
		}
		return s.db.PublishDocumentDraft(run, names, run.Document.DraftPublication.TextHash)
	}
	return s.db.SaveDocumentProgress(run)
}

type documentConvertedFile struct {
	Name string `json:"name"`
	MIME string `json:"mime"`
	Data string `json:"data"`
}
type documentConversionResult struct {
	Warning string                  `json:"warning"`
	Files   []documentConvertedFile `json:"files"`
}

func (s *Server) requestDocumentConversion(ctx context.Context, text []byte, format string) (result documentConversionResult, resultErr error) {
	ctx, cancel := context.WithTimeout(ctx, 310*time.Second)
	defer cancel()
	cfg, _ := s.snapshot()
	if !cfg.Extra.DocumentsEnabled {
		return result, fmt.Errorf("document conversion service is disabled")
	}
	if len(text) > 4<<20 || utf8.RuneCount(text) > 1_000_000 {
		return result, fmt.Errorf("문서 변환은 본문 100만 자·4 MiB까지 지원합니다. HTML·텍스트 결과는 그대로 받을 수 있습니다")
	}
	// The committed text already contains the document's authored title, if any.
	// Conversion must not prepend a synthetic heading or duplicate that title.
	payload, e := json.Marshal(map[string]any{"format": format, "text": string(text)})
	if e != nil {
		return result, e
	}
	if len(payload) > 8<<20 {
		return result, fmt.Errorf("document conversion payload exceeds 8 MiB; HTML/text results remain available")
	}
	release, e := s.acquireWorkload(ctx, "extra-documents")
	if e != nil {
		return result, e
	}
	defer finishWorkload(release, &resultErr)
	req, e := http.NewRequestWithContext(ctx, "POST", strings.TrimRight(cfg.Extra.DocumentsEndpoint, "/")+"/v1/documents/text", bytes.NewReader(payload))
	if e != nil {
		return result, e
	}
	req.Header.Set("Content-Type", "application/json")
	resp, e := http.DefaultClient.Do(req)
	if e != nil {
		return result, e
	}
	defer resp.Body.Close()
	raw, e := io.ReadAll(io.LimitReader(resp.Body, 70<<20))
	if e != nil {
		return result, e
	}
	if resp.StatusCode != 200 {
		if resp.StatusCode == http.StatusNotFound {
			return result, fmt.Errorf("문서 변환 서비스가 장문 변환을 지원하지 않습니다. Extra Documents를 0.5.3 이상으로 갱신해 주세요")
		}
		return result, fmt.Errorf("document conversion HTTP %d: %.500s", resp.StatusCode, raw)
	}

	if e = json.Unmarshal(raw, &result); e != nil {
		return result, e
	}
	if len(result.Files) < 1 || len(result.Files) > 2 {
		return result, fmt.Errorf("invalid document conversion output")
	}
	seen := map[string]bool{}
	for _, f := range result.Files {
		if seen[f.Name] || (f.Name != "document."+format && f.Name != "document.pdf") {
			return result, fmt.Errorf("unexpected or duplicate converted filename")
		}
		seen[f.Name] = true
		data, err := base64.StdEncoding.DecodeString(f.Data)
		if err != nil || len(data) == 0 || len(data) > 24<<20 {
			return result, fmt.Errorf("invalid converted document data")
		}
	}
	if !seen["document."+format] {
		return result, fmt.Errorf("requested document format missing")
	}
	return result, nil
}
