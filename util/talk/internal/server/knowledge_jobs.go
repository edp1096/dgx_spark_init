package server

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"sparktalk/internal/db"
	"sparktalk/internal/knowledge"
)

type knowledgeJobRun struct {
	cancel context.CancelFunc
	done   chan struct{}
}

type knowledgeJobListItem struct {
	db.KnowledgeJob
	Failures []db.KnowledgeJobItem `json:"failures,omitempty"`
}

func (s *Server) knowledgeJobList(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	items, err := s.db.KnowledgeJobs(100)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	result := make([]knowledgeJobListItem, 0, len(items))
	for _, item := range items {
		entry := knowledgeJobListItem{KnowledgeJob: item}
		if item.FailedItems > 0 {
			entry.Failures, _ = s.db.KnowledgeJobFailures(item.ID, 20)
		}
		result = append(result, entry)
	}
	writeJSON(w, http.StatusOK, result)
}

func (s *Server) knowledgeJobAction(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	parts := strings.Split(strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/knowledge/jobs/"), "/"), "/")
	if len(parts) != 2 || len(parts[0]) != 32 {
		http.Error(w, "invalid knowledge job action", http.StatusBadRequest)
		return
	}
	job, err := s.db.KnowledgeJob(parts[0])
	if err != nil {
		writeKnowledgeError(w, err)
		return
	}
	switch parts[1] {
	case "start", "resume", "retry":
		if job.Status == "completed" {
			http.Error(w, "knowledge job is already complete", http.StatusConflict)
			return
		}
		if job.Status == "queued" || job.Status == "running" {
			writeJSON(w, http.StatusOK, job)
			return
		}
		if err := s.db.ResetKnowledgeJobFailures(job.ID); err != nil {
			writeKnowledgeError(w, err)
			return
		}
		_ = s.db.UpdateKnowledgeDocumentStatus(job.DocumentID, "processing", "")
		s.scheduleKnowledgeJob(job.ID)
	case "pause":
		if !s.stopKnowledgeJob(job.ID, 5*time.Second) {
			http.Error(w, "knowledge job is still stopping", http.StatusConflict)
			return
		}
		if err := s.db.SetKnowledgeJobStatus(job.ID, "paused", "", job.CurrentItem); err != nil {
			writeKnowledgeError(w, err)
			return
		}
		_ = s.db.UpdateKnowledgeDocumentStatus(job.DocumentID, "paused", "")
	case "cancel":
		if !s.stopKnowledgeJob(job.ID, 5*time.Second) {
			http.Error(w, "knowledge job is still stopping", http.StatusConflict)
			return
		}
		if err := s.db.SetKnowledgeJobStatus(job.ID, "canceled", "", job.CurrentItem); err != nil {
			writeKnowledgeError(w, err)
			return
		}
		_ = s.db.UpdateKnowledgeDocumentStatus(job.DocumentID, "canceled", "")
	default:
		http.Error(w, "unknown knowledge job action", http.StatusBadRequest)
		return
	}
	updated, err := s.db.KnowledgeJob(job.ID)
	if err != nil {
		writeKnowledgeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, updated)
}

func (s *Server) scheduleKnowledgeJob(id string) {
	s.knowledgeJobMu.Lock()
	if s.knowledgeJobs == nil {
		s.knowledgeJobs = make(map[string]*knowledgeJobRun)
	}
	if s.knowledgeJobSem == nil {
		s.knowledgeJobSem = make(chan struct{}, 1)
	}
	if _, exists := s.knowledgeJobs[id]; exists {
		s.knowledgeJobMu.Unlock()
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	run := &knowledgeJobRun{cancel: cancel, done: make(chan struct{})}
	s.knowledgeJobs[id] = run
	s.knowledgeJobMu.Unlock()
	go s.runKnowledgeJob(ctx, id, run)
}

func (s *Server) stopKnowledgeJob(id string, timeout time.Duration) bool {
	s.knowledgeJobMu.Lock()
	run := s.knowledgeJobs[id]
	s.knowledgeJobMu.Unlock()
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

func (s *Server) stopKnowledgeJobs(ids []string, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for _, id := range ids {
		remaining := time.Until(deadline)
		if remaining <= 0 || !s.stopKnowledgeJob(id, remaining) {
			return false
		}
	}
	return true
}

func (s *Server) finishKnowledgeJobRun(id string, run *knowledgeJobRun) {
	s.knowledgeJobMu.Lock()
	if current := s.knowledgeJobs[id]; current == run {
		delete(s.knowledgeJobs, id)
	}
	close(run.done)
	s.knowledgeJobMu.Unlock()
}

func (s *Server) runKnowledgeJob(ctx context.Context, id string, run *knowledgeJobRun) {
	defer s.finishKnowledgeJobRun(id, run)
	select {
	case s.knowledgeJobSem <- struct{}{}:
		defer func() { <-s.knowledgeJobSem }()
	case <-ctx.Done():
		return
	}
	job, err := s.db.KnowledgeJob(id)
	if err != nil || job.Status != "queued" {
		return
	}
	if err := s.db.SetKnowledgeJobStatus(id, "running", "", job.CurrentItem); err != nil {
		return
	}
	items, err := s.db.KnowledgeJobItems(id, false)
	if err != nil {
		s.failKnowledgeJob(job, err)
		return
	}
	for _, item := range items {
		if ctx.Err() != nil {
			return
		}
		if err := s.db.SetKnowledgeJobStatus(id, "running", "", item.Ordinal); err != nil {
			return
		}
		_ = s.db.SetKnowledgeJobItemStatus(id, item.Ordinal, "running", "")
		if err := s.importKnowledgeJobItem(ctx, job, item); err != nil {
			if ctx.Err() != nil {
				_ = s.db.SetKnowledgeJobItemStatus(id, item.Ordinal, "pending", "")
				return
			}
			_ = s.db.SetKnowledgeJobItemStatus(id, item.Ordinal, "failed", compactKnowledgeError(err))
			continue
		}
		_ = s.db.SetKnowledgeJobItemStatus(id, item.Ordinal, "completed", "")
	}
	if ctx.Err() != nil {
		return
	}
	updated, err := s.db.KnowledgeJob(id)
	if err != nil {
		return
	}
	if updated.CompletedItems == 0 {
		detail := fmt.Sprintf("all %d pages failed", updated.FailedItems)
		_ = s.db.SetKnowledgeJobStatus(id, "failed", detail, updated.CurrentItem)
		_ = s.db.UpdateKnowledgeDocumentStatus(job.DocumentID, "failed", detail)
		return
	}
	status, detail := "completed", ""
	if updated.FailedItems > 0 {
		status = "completed_with_errors"
		detail = fmt.Sprintf("%d of %d pages failed", updated.FailedItems, updated.TotalItems)
	}
	_ = s.db.SetKnowledgeJobStatus(id, status, detail, updated.TotalItems)
	_ = s.db.UpdateKnowledgeDocumentStatus(job.DocumentID, "ready", detail)
}

func (s *Server) importKnowledgeJobItem(ctx context.Context, job db.KnowledgeJob, item db.KnowledgeJobItem) error {
	collected, err := s.collectorSnapshot().Collect(ctx, item.SourceURL, "auto", s.knowledge)
	if err != nil {
		return err
	}
	asset, oldPath, err := s.db.UpsertKnowledgeAsset(db.KnowledgeAsset{
		DocumentID: job.DocumentID, Kind: "page", Ordinal: item.Ordinal, SourceURL: item.SourceURL,
		MIMEType: collected.Source.MIMEType, SizeBytes: collected.Source.SizeBytes, SHA256: collected.Source.SHA256,
		StoragePath: collected.Source.StoragePath, Status: "ready",
	})
	if err != nil {
		s.removeUnreferencedKnowledgeObject(collected.Source.StoragePath)
		return err
	}
	if oldPath != "" && oldPath != asset.StoragePath {
		s.removeUnreferencedKnowledgeObject(oldPath)
	}
	var pages []knowledge.Page
	if strings.TrimSpace(collected.Text) != "" {
		pages = []knowledge.Page{{Number: item.Ordinal, Text: collected.Text}}
	} else {
		path, pathErr := s.knowledge.Path(asset.StoragePath)
		if pathErr != nil {
			return pathErr
		}
		pages, err = s.knowledgeIndex.Extract(path, asset.MIMEType)
		if err != nil {
			return err
		}
	}
	var text strings.Builder
	for _, page := range pages {
		if value := strings.TrimSpace(page.Text); value != "" {
			if text.Len() > 0 {
				text.WriteString("\n\n")
			}
			text.WriteString(value)
		}
	}
	if text.Len() == 0 {
		return errors.New("page contains no searchable text")
	}
	chunks := knowledge.ChunkPages(job.DocumentID, []knowledge.Page{{Number: item.Ordinal, Text: text.String()}})
	return s.db.ReplaceKnowledgePageChunks(job.DocumentID, item.Ordinal, chunks, job.TotalItems)
}

func (s *Server) failKnowledgeJob(job db.KnowledgeJob, err error) {
	detail := compactKnowledgeError(err)
	_ = s.db.SetKnowledgeJobStatus(job.ID, "failed", detail, job.CurrentItem)
	_ = s.db.UpdateKnowledgeDocumentStatus(job.DocumentID, "failed", detail)
}

func compactKnowledgeError(err error) string {
	if err == nil {
		return ""
	}
	value := strings.TrimSpace(err.Error())
	if len(value) > 1000 {
		value = value[:1000]
	}
	return value
}
