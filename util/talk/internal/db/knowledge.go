package db

import (
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"time"
)

type KnowledgeCollection struct {
	ID          int64     `json:"id"`
	Name        string    `json:"name"`
	Description string    `json:"description"`
	Enabled     bool      `json:"enabled"`
	Documents   int       `json:"documents"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

type KnowledgeDocument struct {
	ID                string    `json:"id"`
	CollectionID      int64     `json:"collection_id"`
	Title             string    `json:"title"`
	SourceName        string    `json:"source_name"`
	SourceURL         string    `json:"source_url,omitempty"`
	SourceKind        string    `json:"source_kind"`
	MIMEType          string    `json:"mime_type"`
	SizeBytes         int64     `json:"size_bytes"`
	SHA256            string    `json:"sha256"`
	StoragePath       string    `json:"-"`
	Status            string    `json:"status"`
	Error             string    `json:"error,omitempty"`
	PageCount         int       `json:"page_count"`
	ChunkCount        int       `json:"chunk_count"`
	OCRProcessedPages int       `json:"ocr_processed_pages"`
	OCRTotalPages     int       `json:"ocr_total_pages"`
	CreatedAt         time.Time `json:"created_at"`
	UpdatedAt         time.Time `json:"updated_at"`
}

type KnowledgeChunk struct {
	ID         int64     `json:"id"`
	DocumentID string    `json:"document_id"`
	Ordinal    int       `json:"ordinal"`
	PageStart  int       `json:"page_start"`
	PageEnd    int       `json:"page_end"`
	Heading    string    `json:"heading,omitempty"`
	Content    string    `json:"content"`
	CreatedAt  time.Time `json:"created_at"`
}

type KnowledgeSearchResult struct {
	ChunkID      int64  `json:"chunk_id"`
	Ordinal      int    `json:"ordinal"`
	DocumentID   string `json:"document_id"`
	CollectionID int64  `json:"collection_id"`
	Title        string `json:"title"`
	SourceURL    string `json:"source_url,omitempty"`
	Heading      string `json:"heading,omitempty"`
	Content      string `json:"content"`
	PageStart    int    `json:"page_start"`
	PageEnd      int    `json:"page_end"`
}

func (d *DB) ReadyKnowledgeDocumentCount() (int, error) {
	var count int
	err := d.conn.QueryRow(`
		SELECT count(*) FROM knowledge_documents AS document
		JOIN knowledge_collections AS collection ON collection.id=document.collection_id
		WHERE document.status='ready' AND collection.enabled=1`).Scan(&count)
	return count, err
}

func (d *DB) KnowledgeCollectionIDByName(name string) (int64, error) {
	var id int64
	err := d.conn.QueryRow(`SELECT id FROM knowledge_collections WHERE enabled=1 AND lower(name)=lower(?) ORDER BY id LIMIT 1`, strings.TrimSpace(name)).Scan(&id)
	return id, err
}

func (d *DB) KnowledgeCollections() ([]KnowledgeCollection, error) {
	rows, err := d.conn.Query(`
		SELECT collection.id,collection.name,collection.description,collection.enabled,
			count(document.id),collection.created_at,collection.updated_at
		FROM knowledge_collections AS collection
		LEFT JOIN knowledge_documents AS document ON document.collection_id=collection.id
		GROUP BY collection.id ORDER BY collection.updated_at DESC,collection.id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	items := []KnowledgeCollection{}
	for rows.Next() {
		var item KnowledgeCollection
		if err := rows.Scan(&item.ID, &item.Name, &item.Description, &item.Enabled, &item.Documents, &item.CreatedAt, &item.UpdatedAt); err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	return items, rows.Err()
}

func (d *DB) AddKnowledgeCollection(name, description string) (KnowledgeCollection, error) {
	now := time.Now()
	result, err := d.conn.Exec(`INSERT INTO knowledge_collections(name,description,enabled,created_at,updated_at) VALUES(?,?,1,?,?)`, name, description, now, now)
	if err != nil {
		return KnowledgeCollection{}, err
	}
	id, _ := result.LastInsertId()
	return KnowledgeCollection{ID: id, Name: name, Description: description, Enabled: true, CreatedAt: now, UpdatedAt: now}, nil
}

func (d *DB) UpdateKnowledgeCollection(id int64, name, description string, enabled bool) (KnowledgeCollection, error) {
	now := time.Now()
	result, err := d.conn.Exec(`UPDATE knowledge_collections SET name=?,description=?,enabled=?,updated_at=? WHERE id=?`, name, description, enabled, now, id)
	if err != nil {
		return KnowledgeCollection{}, err
	}
	if changed, _ := result.RowsAffected(); changed == 0 {
		return KnowledgeCollection{}, sql.ErrNoRows
	}
	var item KnowledgeCollection
	err = d.conn.QueryRow(`
		SELECT collection.id,collection.name,collection.description,collection.enabled,count(document.id),collection.created_at,collection.updated_at
		FROM knowledge_collections AS collection LEFT JOIN knowledge_documents AS document ON document.collection_id=collection.id
		WHERE collection.id=? GROUP BY collection.id`, id).
		Scan(&item.ID, &item.Name, &item.Description, &item.Enabled, &item.Documents, &item.CreatedAt, &item.UpdatedAt)
	return item, err
}

func (d *DB) DeleteKnowledgeCollection(id int64) ([]string, error) {
	rows, err := d.conn.Query(`
		SELECT storage_path FROM knowledge_documents WHERE collection_id=?
		UNION
		SELECT asset.storage_path FROM knowledge_assets AS asset
		JOIN knowledge_documents AS document ON document.id=asset.document_id
		WHERE document.collection_id=? AND asset.storage_path<>''`, id, id)
	if err != nil {
		return nil, err
	}
	paths := []string{}
	for rows.Next() {
		var path string
		if err := rows.Scan(&path); err != nil {
			rows.Close()
			return nil, err
		}
		paths = append(paths, path)
	}
	if err := rows.Close(); err != nil {
		return nil, err
	}
	result, err := d.conn.Exec(`DELETE FROM knowledge_collections WHERE id=?`, id)
	if err != nil {
		return nil, err
	}
	if changed, _ := result.RowsAffected(); changed == 0 {
		return nil, sql.ErrNoRows
	}
	return paths, nil
}

func (d *DB) KnowledgeDocuments(collectionID int64) ([]KnowledgeDocument, error) {
	statement := `SELECT id,collection_id,title,source_name,source_url,source_kind,mime_type,size_bytes,sha256,storage_path,status,error,page_count,chunk_count,ocr_processed_pages,ocr_total_pages,created_at,updated_at FROM knowledge_documents`
	args := []any{}
	if collectionID > 0 {
		statement += ` WHERE collection_id=?`
		args = append(args, collectionID)
	}
	statement += ` ORDER BY updated_at DESC,id`
	rows, err := d.conn.Query(statement, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	items := []KnowledgeDocument{}
	for rows.Next() {
		item, err := scanKnowledgeDocument(rows)
		if err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	return items, rows.Err()
}

func (d *DB) KnowledgeDocument(id string) (KnowledgeDocument, error) {
	row := d.conn.QueryRow(`SELECT id,collection_id,title,source_name,source_url,source_kind,mime_type,size_bytes,sha256,storage_path,status,error,page_count,chunk_count,ocr_processed_pages,ocr_total_pages,created_at,updated_at FROM knowledge_documents WHERE id=?`, id)
	return scanKnowledgeDocument(row)
}

type knowledgeDocumentScanner interface{ Scan(...any) error }

func scanKnowledgeDocument(row knowledgeDocumentScanner) (KnowledgeDocument, error) {
	var item KnowledgeDocument
	err := row.Scan(&item.ID, &item.CollectionID, &item.Title, &item.SourceName, &item.SourceURL, &item.SourceKind, &item.MIMEType, &item.SizeBytes, &item.SHA256, &item.StoragePath, &item.Status, &item.Error, &item.PageCount, &item.ChunkCount, &item.OCRProcessedPages, &item.OCRTotalPages, &item.CreatedAt, &item.UpdatedAt)
	return item, err
}

func (d *DB) AddKnowledgeDocument(item KnowledgeDocument) (KnowledgeDocument, bool, error) {
	now := time.Now()
	_, err := d.conn.Exec(`INSERT INTO knowledge_documents(id,collection_id,title,source_name,source_url,source_kind,mime_type,size_bytes,sha256,storage_path,status,error,page_count,chunk_count,ocr_processed_pages,ocr_total_pages,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`,
		item.ID, item.CollectionID, item.Title, item.SourceName, item.SourceURL, item.SourceKind, item.MIMEType, item.SizeBytes, item.SHA256, item.StoragePath, item.Status, item.Error, item.PageCount, item.ChunkCount, item.OCRProcessedPages, item.OCRTotalPages, now, now)
	if err == nil {
		item.CreatedAt, item.UpdatedAt = now, now
		return item, false, nil
	}
	existing, queryErr := d.knowledgeDocumentByHash(item.CollectionID, item.SHA256)
	if queryErr == nil {
		return existing, true, nil
	}
	return KnowledgeDocument{}, false, err
}

func (d *DB) knowledgeDocumentByHash(collectionID int64, sha256 string) (KnowledgeDocument, error) {
	row := d.conn.QueryRow(`SELECT id,collection_id,title,source_name,source_url,source_kind,mime_type,size_bytes,sha256,storage_path,status,error,page_count,chunk_count,ocr_processed_pages,ocr_total_pages,created_at,updated_at FROM knowledge_documents WHERE collection_id=? AND sha256=?`, collectionID, sha256)
	return scanKnowledgeDocument(row)
}

func (d *DB) ReplaceKnowledgeChunks(documentID string, chunks []KnowledgeChunk, pageCount int, status, detail string) error {
	tx, err := d.conn.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if _, err := tx.Exec(`DELETE FROM knowledge_chunks WHERE document_id=?`, documentID); err != nil {
		return err
	}
	now := time.Now()
	for _, chunk := range chunks {
		if _, err := tx.Exec(`INSERT INTO knowledge_chunks(document_id,ordinal,page_start,page_end,heading,content,created_at) VALUES(?,?,?,?,?,?,?)`, documentID, chunk.Ordinal, chunk.PageStart, chunk.PageEnd, chunk.Heading, chunk.Content, now); err != nil {
			return err
		}
	}
	result, err := tx.Exec(`UPDATE knowledge_documents SET status=?,error=?,page_count=?,chunk_count=?,updated_at=? WHERE id=?`, status, detail, pageCount, len(chunks), now, documentID)
	if err != nil {
		return err
	}
	if changed, _ := result.RowsAffected(); changed == 0 {
		return sql.ErrNoRows
	}
	return tx.Commit()
}

func (d *DB) UpdateKnowledgeDocumentStatus(documentID, status, detail string) error {
	result, err := d.conn.Exec(`UPDATE knowledge_documents SET status=?,error=?,updated_at=? WHERE id=?`, status, detail, time.Now(), documentID)
	if err != nil {
		return err
	}
	if changed, _ := result.RowsAffected(); changed == 0 {
		return sql.ErrNoRows
	}
	return nil
}

func (d *DB) BeginKnowledgeOCR(documentID string, totalPages int, resume bool) error {
	tx, err := d.conn.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if !resume {
		if _, err := tx.Exec(`DELETE FROM knowledge_chunks WHERE document_id=?`, documentID); err != nil {
			return err
		}
	}
	processedExpression := "ocr_processed_pages"
	if !resume {
		processedExpression = "0"
	}
	result, err := tx.Exec(`UPDATE knowledge_documents SET status='processing',error='',ocr_processed_pages=`+processedExpression+`,ocr_total_pages=?,chunk_count=(SELECT count(*) FROM knowledge_chunks WHERE document_id=?),updated_at=? WHERE id=?`, totalPages, documentID, time.Now(), documentID)
	if err != nil {
		return err
	}
	if changed, _ := result.RowsAffected(); changed == 0 {
		return sql.ErrNoRows
	}
	return tx.Commit()
}

func (d *DB) ReplaceKnowledgeOCRPageChunks(documentID string, page int, chunks []KnowledgeChunk, totalPages int) error {
	tx, err := d.conn.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	if _, err := tx.Exec(`DELETE FROM knowledge_chunks WHERE document_id=? AND page_start=?`, documentID, page); err != nil {
		return err
	}
	now := time.Now()
	for index, chunk := range chunks {
		ordinal := (page-1)*1000 + index
		if _, err := tx.Exec(`INSERT INTO knowledge_chunks(document_id,ordinal,page_start,page_end,heading,content,created_at) VALUES(?,?,?,?,?,?,?)`, documentID, ordinal, page, page, chunk.Heading, chunk.Content, now); err != nil {
			return err
		}
	}
	result, err := tx.Exec(`UPDATE knowledge_documents SET page_count=?,ocr_processed_pages=?,ocr_total_pages=?,chunk_count=(SELECT count(*) FROM knowledge_chunks WHERE document_id=?),updated_at=? WHERE id=?`, totalPages, page, totalPages, documentID, now, documentID)
	if err != nil {
		return err
	}
	if changed, _ := result.RowsAffected(); changed == 0 {
		return sql.ErrNoRows
	}
	return tx.Commit()
}

func (d *DB) FinishKnowledgeOCR(documentID, status, detail string) error {
	result, err := d.conn.Exec(`UPDATE knowledge_documents SET status=?,error=?,ocr_processed_pages=CASE WHEN ?='ready' THEN ocr_total_pages ELSE ocr_processed_pages END,chunk_count=(SELECT count(*) FROM knowledge_chunks WHERE document_id=?),updated_at=? WHERE id=?`, status, detail, status, documentID, time.Now(), documentID)
	if err != nil {
		return err
	}
	if changed, _ := result.RowsAffected(); changed == 0 {
		return sql.ErrNoRows
	}
	return nil
}

func (d *DB) RecoverKnowledgeOCR() error {
	_, err := d.conn.Exec(`UPDATE knowledge_documents SET status='needs_ocr',error='OCR 작업이 서버 재시작으로 중단되었습니다. 다시 실행하세요.',updated_at=? WHERE status='processing' AND ocr_total_pages>0`, time.Now())
	return err
}

func (d *DB) DeleteKnowledgeDocument(id string) (string, error) {
	document, err := d.KnowledgeDocument(id)
	if err != nil {
		return "", err
	}
	result, err := d.conn.Exec(`DELETE FROM knowledge_documents WHERE id=?`, id)
	if err != nil {
		return "", err
	}
	if changed, _ := result.RowsAffected(); changed == 0 {
		return "", sql.ErrNoRows
	}
	return document.StoragePath, nil
}

func (d *DB) KnowledgeDocumentStoragePaths(id string) ([]string, error) {
	rows, err := d.conn.Query(`SELECT storage_path FROM knowledge_documents WHERE id=? UNION SELECT storage_path FROM knowledge_assets WHERE document_id=? AND storage_path<>''`, id, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	paths := []string{}
	for rows.Next() {
		var path string
		if err := rows.Scan(&path); err != nil {
			return nil, err
		}
		paths = append(paths, path)
	}
	return paths, rows.Err()
}

func (d *DB) KnowledgeStorageReferenced(storagePath string) (bool, error) {
	var count int
	err := d.conn.QueryRow(`SELECT
		(SELECT count(*) FROM knowledge_documents WHERE storage_path=?) +
		(SELECT count(*) FROM knowledge_assets WHERE storage_path=?)`, storagePath, storagePath).Scan(&count)
	return count > 0, err
}

func (d *DB) SearchKnowledge(query string, collectionID int64, limit int) ([]KnowledgeSearchResult, error) {
	query = strings.TrimSpace(query)
	if query == "" {
		return []KnowledgeSearchResult{}, nil
	}
	if limit < 1 || limit > 50 {
		limit = 12
	}
	match := ftsMatchQuery(manualSearchTerms(query))
	if match == "" {
		return d.searchKnowledgeSubstring(query, collectionID, limit)
	}
	where := []string{"knowledge_search MATCH ?", "document.status='ready'", "collection.enabled=1"}
	args := []any{match}
	if collectionID > 0 {
		where = append(where, "document.collection_id=?")
		args = append(args, collectionID)
	}
	args = append(args, limit)
	rows, err := d.conn.Query(fmt.Sprintf(`
		SELECT chunk.id,chunk.ordinal,document.id,document.collection_id,document.title,document.source_url,chunk.heading,
			snippet(knowledge_search,4,'','',' … ',48),chunk.page_start,chunk.page_end
		FROM knowledge_search
		JOIN knowledge_chunks AS chunk ON chunk.id=knowledge_search.rowid
		JOIN knowledge_documents AS document ON document.id=chunk.document_id
		JOIN knowledge_collections AS collection ON collection.id=document.collection_id
		WHERE %s ORDER BY bm25(knowledge_search),chunk.id LIMIT ?`, strings.Join(where, " AND ")), args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	items := []KnowledgeSearchResult{}
	for rows.Next() {
		var item KnowledgeSearchResult
		if err := rows.Scan(&item.ChunkID, &item.Ordinal, &item.DocumentID, &item.CollectionID, &item.Title, &item.SourceURL, &item.Heading, &item.Content, &item.PageStart, &item.PageEnd); err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	return items, rows.Err()
}

func (d *DB) searchKnowledgeSubstring(query string, collectionID int64, limit int) ([]KnowledgeSearchResult, error) {
	where := []string{"document.status='ready'", "collection.enabled=1", "(instr(lower(chunk.content),lower(?))>0 OR instr(lower(document.title),lower(?))>0)"}
	args := []any{query, query}
	if collectionID > 0 {
		where = append(where, "document.collection_id=?")
		args = append(args, collectionID)
	}
	args = append(args, limit)
	rows, err := d.conn.Query(fmt.Sprintf(`
		SELECT chunk.id,chunk.ordinal,document.id,document.collection_id,document.title,document.source_url,chunk.heading,chunk.content,chunk.page_start,chunk.page_end
		FROM knowledge_chunks AS chunk
		JOIN knowledge_documents AS document ON document.id=chunk.document_id
		JOIN knowledge_collections AS collection ON collection.id=document.collection_id
		WHERE %s ORDER BY document.updated_at DESC,chunk.ordinal LIMIT ?`, strings.Join(where, " AND ")), args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	items := []KnowledgeSearchResult{}
	for rows.Next() {
		var item KnowledgeSearchResult
		if err := rows.Scan(&item.ChunkID, &item.Ordinal, &item.DocumentID, &item.CollectionID, &item.Title, &item.SourceURL, &item.Heading, &item.Content, &item.PageStart, &item.PageEnd); err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	return items, rows.Err()
}

func (d *DB) KnowledgeChunksAround(documentID string, ordinal, radius int) (KnowledgeDocument, []KnowledgeChunk, error) {
	if radius < 0 {
		radius = 0
	}
	if radius > 2 {
		radius = 2
	}
	document, err := d.KnowledgeDocument(documentID)
	if err != nil {
		return KnowledgeDocument{}, nil, err
	}
	var enabled bool
	if err := d.conn.QueryRow(`SELECT enabled FROM knowledge_collections WHERE id=?`, document.CollectionID).Scan(&enabled); err != nil {
		return KnowledgeDocument{}, nil, err
	}
	if !enabled || document.Status != "ready" {
		return KnowledgeDocument{}, nil, sql.ErrNoRows
	}
	rows, err := d.conn.Query(`
		SELECT id,document_id,ordinal,page_start,page_end,heading,content,created_at
		FROM knowledge_chunks WHERE document_id=? AND ordinal<=? ORDER BY ordinal DESC LIMIT ?`,
		documentID, ordinal, radius+1)
	if err != nil {
		return KnowledgeDocument{}, nil, err
	}
	defer rows.Close()
	before := []KnowledgeChunk{}
	for rows.Next() {
		var chunk KnowledgeChunk
		if err := rows.Scan(&chunk.ID, &chunk.DocumentID, &chunk.Ordinal, &chunk.PageStart, &chunk.PageEnd, &chunk.Heading, &chunk.Content, &chunk.CreatedAt); err != nil {
			return KnowledgeDocument{}, nil, err
		}
		before = append(before, chunk)
	}
	if err := rows.Err(); err != nil {
		return KnowledgeDocument{}, nil, err
	}
	rows.Close()
	if len(before) == 0 || before[0].Ordinal != ordinal {
		return KnowledgeDocument{}, nil, sql.ErrNoRows
	}
	chunks := make([]KnowledgeChunk, 0, radius*2+1)
	for index := len(before) - 1; index >= 0; index-- {
		chunks = append(chunks, before[index])
	}
	rows, err = d.conn.Query(`
		SELECT id,document_id,ordinal,page_start,page_end,heading,content,created_at
		FROM knowledge_chunks WHERE document_id=? AND ordinal>? ORDER BY ordinal LIMIT ?`, documentID, ordinal, radius)
	if err != nil {
		return KnowledgeDocument{}, nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var chunk KnowledgeChunk
		if err := rows.Scan(&chunk.ID, &chunk.DocumentID, &chunk.Ordinal, &chunk.PageStart, &chunk.PageEnd, &chunk.Heading, &chunk.Content, &chunk.CreatedAt); err != nil {
			return KnowledgeDocument{}, nil, err
		}
		chunks = append(chunks, chunk)
	}
	if err := rows.Err(); err != nil {
		return KnowledgeDocument{}, nil, err
	}
	return document, chunks, nil
}

func IsKnowledgeNotFound(err error) bool { return errors.Is(err, sql.ErrNoRows) }
