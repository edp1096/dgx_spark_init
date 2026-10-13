package server

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"strings"
	"unicode/utf8"

	"sparktalk/internal/workflows"
)

const documentTextConverter = "committed-text-v2"

func (s *Server) startDocumentReconversion(run workflows.Run, format string) (workflows.Run, error) {
	s.docTaskMu.Lock()
	defer s.docTaskMu.Unlock()
	jobs, err := s.db.DocumentConversions(run.SessionID, run.ID)
	if err != nil {
		return run, err
	}
	for _, job := range jobs {
		if job.Status == "running" && job.Format == format {
			return run, nil
		}
	}
	if s.docTasks[run.ID] != nil {
		return run, fmt.Errorf("문서 처리 완료 후 재변환해 주세요")
	}
	if !s.docExportMu.TryLock() {
		return run, fmt.Errorf("다른 문서 파일을 변환 중입니다. 완료 후 다시 시도해 주세요")
	}
	started := false
	defer func() {
		if !started {
			s.docExportMu.Unlock()
		}
	}()
	cfg, _ := s.snapshot()
	if !cfg.Extra.DocumentsEnabled {
		return run, fmt.Errorf("문서 변환 서비스가 비활성화되어 있습니다")
	}
	body, _, err := s.db.DocumentOutput(run.SessionID, run.ID, "output.txt")
	if err != nil {
		return run, err
	}
	if len(body) > 4<<20 || utf8.RuneCount(body) > 1_000_000 {
		return run, fmt.Errorf("문서 변환은 본문 100만 자·4 MiB까지 지원합니다")
	}
	ctx, finish, err := s.tasks.Track(context.Background())
	if err != nil {
		return run, err
	}
	job, body, err := s.db.BeginDocumentConversion(run.SessionID, run.ID, format, documentTextConverter)
	if err != nil {
		finish()
		return run, err
	}
	started = true
	go func() {
		defer finish()
		defer s.docExportMu.Unlock()
		result, err := s.requestDocumentConversion(ctx, body, format)
		if err == nil {
			for _, file := range result.Files {
				// Office conversions may return an additional PDF. Only the
				// requested format gets a new revision; other downloads stay put.
				if file.Name != "document."+format {
					continue
				}
				data, _ := base64.StdEncoding.DecodeString(file.Data)
				if err = validateReconvertedDocument(format, file.MIME, data); err == nil {
					err = s.db.CompleteDocumentConversion(run.SessionID, run.ID, job.ID, file.MIME, data)
				}
				break
			}
		}
		if err != nil {
			_ = s.db.FailDocumentConversion(run.SessionID, run.ID, job.ID, err)
		}
	}()
	return run, nil
}

func validateReconvertedDocument(format, mime string, data []byte) error {
	expected := map[string]string{"pdf": "application/pdf", "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "hwp": "application/x-hwp", "hwpx": "application/vnd.hancom.hwpx"}
	if mime != expected[format] {
		return fmt.Errorf("converted document MIME does not match %s", format)
	}
	valid := false
	switch format {
	case "pdf":
		valid = bytes.HasPrefix(data, []byte("%PDF-"))
	case "hwp":
		valid = bytes.HasPrefix(data, []byte{0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1}) && bytes.Contains(data, []byte("HWP Document File"))
	case "docx", "hwpx":
		z, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
		if err == nil && len(z.File) <= 10000 {
			for _, f := range z.File {
				if format == "docx" && f.Name == "word/document.xml" || format == "hwpx" && strings.HasPrefix(f.Name, "Contents/section") && strings.HasSuffix(f.Name, ".xml") {
					valid = true
				}
			}
		}
	}
	if !valid {
		return fmt.Errorf("invalid converted %s document", format)
	}
	return nil
}

func (s *Server) documentResultView(run workflows.Run) map[string]any {
	jobs, err := s.db.DocumentConversions(run.SessionID, run.ID)
	if err == nil {
		run.Conversions = jobs
	}
	return documentView(run)
}
