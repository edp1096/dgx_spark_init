package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"mime"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"sparktalk/internal/db"
	"sparktalk/internal/knowledge"
)

type knowledgeCollectionRequest struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Enabled     *bool  `json:"enabled,omitempty"`
}

func (s *Server) knowledgeCollections(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		items, err := s.db.KnowledgeCollections()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, http.StatusOK, items)
	case http.MethodPost:
		var input knowledgeCollectionRequest
		decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&input); err != nil {
			http.Error(w, "invalid knowledge collection", http.StatusBadRequest)
			return
		}
		if message := normalizeKnowledgeCollection(&input); message != "" {
			http.Error(w, message, http.StatusBadRequest)
			return
		}
		item, err := s.db.AddKnowledgeCollection(input.Name, input.Description)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, http.StatusCreated, item)
	default:
		methodNotAllowed(w)
	}
}

func (s *Server) knowledgeCollection(w http.ResponseWriter, r *http.Request) {
	rawID := strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/knowledge/collections/"), "/")
	id, err := strconv.ParseInt(rawID, 10, 64)
	if err != nil || id < 1 {
		http.Error(w, "invalid knowledge collection id", http.StatusBadRequest)
		return
	}
	switch r.Method {
	case http.MethodPut:
		var input knowledgeCollectionRequest
		decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&input); err != nil {
			http.Error(w, "invalid knowledge collection", http.StatusBadRequest)
			return
		}
		if message := normalizeKnowledgeCollection(&input); message != "" {
			http.Error(w, message, http.StatusBadRequest)
			return
		}
		enabled := true
		if input.Enabled != nil {
			enabled = *input.Enabled
		}
		item, err := s.db.UpdateKnowledgeCollection(id, input.Name, input.Description, enabled)
		if err != nil {
			writeKnowledgeError(w, err)
			return
		}
		writeJSON(w, http.StatusOK, item)
	case http.MethodDelete:
		jobIDs, err := s.db.KnowledgeJobIDsForCollection(id)
		if err != nil {
			writeKnowledgeError(w, err)
			return
		}
		if !s.stopKnowledgeJobs(jobIDs, 5*time.Second) {
			http.Error(w, "knowledge imports are still stopping", http.StatusConflict)
			return
		}
		documents, err := s.db.KnowledgeDocuments(id)
		if err != nil {
			writeKnowledgeError(w, err)
			return
		}
		if !s.stopKnowledgeOCRs(documents, 5*time.Second) {
			http.Error(w, "knowledge OCR is still stopping", http.StatusConflict)
			return
		}
		paths, err := s.db.DeleteKnowledgeCollection(id)
		if err != nil {
			writeKnowledgeError(w, err)
			return
		}
		for _, path := range paths {
			s.removeUnreferencedKnowledgeObject(path)
		}
		if collections, listErr := s.db.KnowledgeCollections(); listErr == nil && len(collections) == 0 {
			_, _ = s.db.AddKnowledgeCollection("내 지식", "문서와 수집 자료")
		}
		w.WriteHeader(http.StatusNoContent)
	default:
		methodNotAllowed(w)
	}
}

func normalizeKnowledgeCollection(input *knowledgeCollectionRequest) string {
	input.Name = strings.TrimSpace(input.Name)
	input.Description = strings.TrimSpace(input.Description)
	if input.Name == "" || utf8.RuneCountInString(input.Name) > 80 {
		return "knowledge collection name must be between 1 and 80 characters"
	}
	if utf8.RuneCountInString(input.Description) > 500 {
		return "knowledge collection description supports at most 500 characters"
	}
	return ""
}

func (s *Server) knowledgeDocuments(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		collectionID, _ := strconv.ParseInt(r.URL.Query().Get("collection_id"), 10, 64)
		items, err := s.db.KnowledgeDocuments(collectionID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, http.StatusOK, items)
	case http.MethodPost:
		s.uploadKnowledgeDocument(w, r)
	default:
		methodNotAllowed(w)
	}
}

func (s *Server) uploadKnowledgeDocument(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, knowledge.MaxSourceBytes+(1<<20))
	if err := r.ParseMultipartForm(8 << 20); err != nil {
		http.Error(w, "invalid knowledge upload or file is too large", http.StatusBadRequest)
		return
	}
	defer r.MultipartForm.RemoveAll()
	collectionID, err := strconv.ParseInt(strings.TrimSpace(r.FormValue("collection_id")), 10, 64)
	if err != nil || collectionID < 1 {
		http.Error(w, "knowledge collection is required", http.StatusBadRequest)
		return
	}
	file, header, err := r.FormFile("file")
	if err != nil {
		http.Error(w, "knowledge file is required", http.StatusBadRequest)
		return
	}
	_ = file.Close()
	source, err := s.knowledge.Save(header)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	id, err := knowledge.NewDocumentID()
	if err != nil {
		s.removeUnreferencedKnowledgeObject(source.StoragePath)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	title := strings.TrimSpace(r.FormValue("title"))
	if title == "" {
		title = knowledge.TitleFromName(source.Name)
	}
	if utf8.RuneCountInString(title) > 200 {
		s.removeUnreferencedKnowledgeObject(source.StoragePath)
		http.Error(w, "knowledge document title supports at most 200 characters", http.StatusBadRequest)
		return
	}
	document, duplicate, err := s.db.AddKnowledgeDocument(db.KnowledgeDocument{
		ID: id, CollectionID: collectionID, Title: title, SourceName: source.Name,
		SourceKind: "file", MIMEType: source.MIMEType, SizeBytes: source.SizeBytes,
		SHA256: source.SHA256, StoragePath: source.StoragePath, Status: "processing",
	})
	if err != nil {
		s.removeUnreferencedKnowledgeObject(source.StoragePath)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if duplicate {
		writeJSON(w, http.StatusOK, document)
		return
	}
	path, err := s.knowledge.Path(document.StoragePath)
	if err != nil {
		_ = s.db.ReplaceKnowledgeChunks(document.ID, nil, 0, "failed", err.Error())
		updated, queryErr := s.db.KnowledgeDocument(document.ID)
		if queryErr != nil {
			http.Error(w, queryErr.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, http.StatusCreated, updated)
		return
	}
	pages, extractErr := s.knowledgeIndex.Extract(path, document.MIMEType)
	updated, err := s.finishKnowledgeIndex(document, pages, extractErr)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusCreated, updated)
}

type knowledgeSourceRequest struct {
	CollectionID int64  `json:"collection_id"`
	URL          string `json:"url"`
	Mode         string `json:"mode"`
}

type knowledgeSourceResponse struct {
	Document    db.KnowledgeDocument            `json:"document"`
	Links       []knowledge.CollectedLink       `json:"links"`
	Publication *knowledge.CollectedPublication `json:"publication,omitempty"`
	Job         *db.KnowledgeJob                `json:"job,omitempty"`
}

func (s *Server) collectKnowledgeSource(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	var input knowledgeSourceRequest
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&input); err != nil {
		http.Error(w, "invalid knowledge source", http.StatusBadRequest)
		return
	}
	if err := normalizeKnowledgeSourceRequest(&input); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	result, status, err := s.importKnowledgeSource(r.Context(), input)
	if err != nil {
		http.Error(w, err.Error(), status)
		return
	}
	writeJSON(w, status, result)
}

func normalizeKnowledgeSourceRequest(input *knowledgeSourceRequest) error {
	input.URL = strings.TrimSpace(input.URL)
	input.Mode = strings.ToLower(strings.TrimSpace(input.Mode))
	if input.Mode == "" {
		input.Mode = "auto"
	}
	parsed, err := url.Parse(input.URL)
	if input.CollectionID < 1 || err != nil || parsed.Host == "" || parsed.Scheme != "http" && parsed.Scheme != "https" {
		return errors.New("a knowledge collection and HTTP(S) URL are required")
	}
	if input.Mode != "auto" && input.Mode != "direct" && input.Mode != "browser" {
		return errors.New("collection mode must be auto, direct, or browser")
	}
	return nil
}

// importKnowledgeSource is shared by the HTTP knowledge UI and the approved
// conversational import tool. Persisting external material must happen only
// after the caller has obtained explicit user intent or UI approval.
func (s *Server) importKnowledgeSource(ctx context.Context, input knowledgeSourceRequest) (knowledgeSourceResponse, int, error) {
	collected, err := s.collectorSnapshot().Collect(ctx, input.URL, input.Mode, s.knowledge)
	if err != nil {
		return knowledgeSourceResponse{}, http.StatusBadGateway, err
	}
	id, err := knowledge.NewDocumentID()
	if err != nil {
		s.removeUnreferencedKnowledgeObject(collected.Source.StoragePath)
		return knowledgeSourceResponse{}, http.StatusInternalServerError, err
	}
	title := trimKnowledgeTitle(collected.Manifest.Title)
	if title == "" {
		title = knowledge.TitleFromName(collected.Source.Name)
	}
	sourceURL := strings.TrimSpace(collected.Manifest.FinalURL)
	if sourceURL == "" {
		sourceURL = input.URL
	}
	document, duplicate, err := s.db.AddKnowledgeDocument(db.KnowledgeDocument{
		ID: id, CollectionID: input.CollectionID, Title: title, SourceName: collected.Source.Name,
		SourceURL: sourceURL, SourceKind: "url", MIMEType: collected.Source.MIMEType,
		SizeBytes: collected.Source.SizeBytes, SHA256: collected.Source.SHA256,
		StoragePath: collected.Source.StoragePath, Status: "processing",
	})
	if err != nil {
		s.removeUnreferencedKnowledgeObject(collected.Source.StoragePath)
		return knowledgeSourceResponse{}, http.StatusBadRequest, err
	}
	if duplicate {
		var job *db.KnowledgeJob
		if existing, jobErr := s.db.KnowledgeJobForDocument(document.ID); jobErr == nil {
			job = &existing
		}
		return knowledgeSourceResponse{Document: document, Links: collected.Links, Publication: collected.Publication, Job: job}, http.StatusOK, nil
	}
	if publication := collected.Publication; publication != nil {
		if err := s.db.ReplaceKnowledgeChunks(document.ID, nil, publication.PageCount, "paused", ""); err != nil {
			return knowledgeSourceResponse{}, http.StatusInternalServerError, err
		}
		jobID, idErr := knowledge.NewDocumentID()
		if idErr != nil {
			return knowledgeSourceResponse{}, http.StatusInternalServerError, idErr
		}
		items := make([]db.KnowledgeJobItem, 0, len(publication.Pages))
		for _, page := range publication.Pages {
			items = append(items, db.KnowledgeJobItem{JobID: jobID, Ordinal: page.Number, SourceURL: page.URL, MIMEType: page.MIMEType, Status: "pending"})
		}
		job, jobErr := s.db.AddKnowledgeJob(db.KnowledgeJob{
			ID: jobID, DocumentID: document.ID, CollectionID: input.CollectionID, SourceURL: sourceURL,
			Mode: input.Mode, Adapter: publication.Adapter, Title: publication.Title, Status: "paused",
		}, items)
		if jobErr != nil {
			_ = s.db.UpdateKnowledgeDocumentStatus(document.ID, "failed", jobErr.Error())
			return knowledgeSourceResponse{}, http.StatusInternalServerError, jobErr
		}
		updated, queryErr := s.db.KnowledgeDocument(document.ID)
		if queryErr != nil {
			return knowledgeSourceResponse{}, http.StatusInternalServerError, queryErr
		}
		return knowledgeSourceResponse{Document: updated, Links: collected.Links, Publication: publication, Job: &job}, http.StatusCreated, nil
	}
	var pages []knowledge.Page
	var extractErr error
	if strings.TrimSpace(collected.Text) != "" {
		pages = []knowledge.Page{{Number: 1, Text: collected.Text}}
	} else {
		path, pathErr := s.knowledge.Path(document.StoragePath)
		if pathErr != nil {
			extractErr = pathErr
		} else {
			pages, extractErr = s.knowledgeIndex.Extract(path, document.MIMEType)
		}
	}
	updated, err := s.finishKnowledgeIndex(document, pages, extractErr)
	if err != nil {
		return knowledgeSourceResponse{}, http.StatusInternalServerError, err
	}
	return knowledgeSourceResponse{Document: updated, Links: collected.Links, Publication: collected.Publication}, http.StatusCreated, nil
}

func trimKnowledgeTitle(value string) string {
	value = strings.TrimSpace(value)
	runes := []rune(value)
	if len(runes) > 200 {
		value = strings.TrimSpace(string(runes[:200]))
	}
	return value
}

func (s *Server) finishKnowledgeIndex(document db.KnowledgeDocument, pages []knowledge.Page, extractErr error) (db.KnowledgeDocument, error) {
	status, detail := "ready", ""
	if extractErr != nil {
		status, detail = "failed", extractErr.Error()
		if errors.Is(extractErr, knowledge.ErrOCRRequired) && knowledge.SupportsOCR(document.MIMEType) {
			status = "needs_ocr"
		} else if errors.Is(extractErr, knowledge.ErrOCRRequired) {
			detail = fmt.Sprintf("document has no usable text; OCR does not support %s", document.MIMEType)
		}
	}
	chunks := knowledge.ChunkPages(document.ID, pages)
	if extractErr == nil && len(chunks) == 0 {
		status, detail = "failed", "document contains no searchable text"
	}
	if err := s.db.ReplaceKnowledgeChunks(document.ID, chunks, len(pages), status, detail); err != nil {
		return db.KnowledgeDocument{}, err
	}
	updated, err := s.db.KnowledgeDocument(document.ID)
	if err != nil {
		return db.KnowledgeDocument{}, err
	}
	return updated, nil
}

func (s *Server) knowledgeDocument(w http.ResponseWriter, r *http.Request) {
	rest := strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/knowledge/documents/"), "/")
	parts := strings.Split(rest, "/")
	if len(parts) == 0 || len(parts[0]) != 32 {
		http.Error(w, "invalid knowledge document id", http.StatusBadRequest)
		return
	}
	document, err := s.db.KnowledgeDocument(parts[0])
	if err != nil {
		writeKnowledgeError(w, err)
		return
	}
	if len(parts) == 2 && parts[1] == "source" {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			methodNotAllowed(w)
			return
		}
		s.serveKnowledgeSource(w, r, document)
		return
	}
	if len(parts) == 2 && parts[1] == "ocr" {
		if r.Method != http.MethodPost {
			methodNotAllowed(w)
			return
		}
		s.ocrKnowledgeDocument(w, r, document)
		return
	}
	if len(parts) != 1 {
		http.NotFound(w, r)
		return
	}
	switch r.Method {
	case http.MethodGet:
		writeJSON(w, http.StatusOK, document)
	case http.MethodDelete:
		if job, jobErr := s.db.KnowledgeJobForDocument(document.ID); jobErr == nil && !s.stopKnowledgeJob(job.ID, 5*time.Second) {
			http.Error(w, "knowledge import is still stopping", http.StatusConflict)
			return
		}
		if !s.stopKnowledgeOCR(document.ID, 5*time.Second) {
			http.Error(w, "knowledge OCR is still stopping", http.StatusConflict)
			return
		}
		paths, pathErr := s.db.KnowledgeDocumentStoragePaths(document.ID)
		if pathErr != nil {
			writeKnowledgeError(w, pathErr)
			return
		}
		_, err := s.db.DeleteKnowledgeDocument(document.ID)
		if err != nil {
			writeKnowledgeError(w, err)
			return
		}
		for _, storagePath := range paths {
			s.removeUnreferencedKnowledgeObject(storagePath)
		}
		w.WriteHeader(http.StatusNoContent)
	default:
		methodNotAllowed(w)
	}
}

func (s *Server) serveKnowledgeSource(w http.ResponseWriter, r *http.Request, document db.KnowledgeDocument) {
	file, err := s.knowledge.Open(document.StoragePath)
	if err != nil {
		http.NotFound(w, r)
		return
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		http.NotFound(w, r)
		return
	}
	disposition := mime.FormatMediaType("inline", map[string]string{"filename": document.SourceName})
	w.Header().Set("Content-Type", document.MIMEType)
	w.Header().Set("Content-Disposition", disposition)
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Cache-Control", "private, max-age=3600")
	http.ServeContent(w, r, document.SourceName, info.ModTime(), file)
}

func (s *Server) searchKnowledge(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	query := strings.TrimSpace(r.URL.Query().Get("q"))
	if query == "" {
		http.Error(w, "knowledge search query is required", http.StatusBadRequest)
		return
	}
	collectionID, _ := strconv.ParseInt(r.URL.Query().Get("collection_id"), 10, 64)
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	items, err := s.db.SearchKnowledge(query, collectionID, limit)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *Server) removeUnreferencedKnowledgeObject(storagePath string) {
	referenced, err := s.db.KnowledgeStorageReferenced(storagePath)
	if err == nil && !referenced {
		_ = s.knowledge.Remove(storagePath)
	}
}

func writeKnowledgeError(w http.ResponseWriter, err error) {
	if db.IsKnowledgeNotFound(err) {
		http.Error(w, "knowledge item not found", http.StatusNotFound)
		return
	}
	http.Error(w, fmt.Sprintf("knowledge: %v", err), http.StatusInternalServerError)
}
