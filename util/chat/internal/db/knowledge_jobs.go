package db

import (
	"database/sql"
	"time"
)

type KnowledgeAsset struct {
	ID          int64     `json:"id"`
	DocumentID  string    `json:"document_id"`
	Kind        string    `json:"kind"`
	Ordinal     int       `json:"ordinal"`
	SourceURL   string    `json:"source_url,omitempty"`
	MIMEType    string    `json:"mime_type,omitempty"`
	SizeBytes   int64     `json:"size_bytes"`
	SHA256      string    `json:"sha256,omitempty"`
	StoragePath string    `json:"-"`
	Status      string    `json:"status"`
	Error       string    `json:"error,omitempty"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

type KnowledgeJob struct {
	ID             string    `json:"id"`
	DocumentID     string    `json:"document_id"`
	CollectionID   int64     `json:"collection_id"`
	SourceURL      string    `json:"source_url"`
	Mode           string    `json:"mode"`
	Adapter        string    `json:"adapter"`
	Title          string    `json:"title"`
	Status         string    `json:"status"`
	TotalItems     int       `json:"total_items"`
	CompletedItems int       `json:"completed_items"`
	FailedItems    int       `json:"failed_items"`
	CurrentItem    int       `json:"current_item"`
	Error          string    `json:"error,omitempty"`
	CreatedAt      time.Time `json:"created_at"`
	UpdatedAt      time.Time `json:"updated_at"`
}

type KnowledgeJobItem struct {
	JobID     string    `json:"job_id"`
	Ordinal   int       `json:"ordinal"`
	SourceURL string    `json:"source_url"`
	MIMEType  string    `json:"mime_type,omitempty"`
	Status    string    `json:"status"`
	Error     string    `json:"error,omitempty"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

func (d *DB) AddKnowledgeJob(job KnowledgeJob, items []KnowledgeJobItem) (KnowledgeJob, error) {
	tx, err := d.conn.Begin()
	if err != nil {
		return KnowledgeJob{}, err
	}
	defer tx.Rollback()
	now := time.Now()
	job.TotalItems = len(items)
	job.CreatedAt, job.UpdatedAt = now, now
	if _, err := tx.Exec(`INSERT INTO knowledge_jobs(id,document_id,collection_id,source_url,mode,adapter,title,status,total_items,completed_items,failed_items,current_item,error,created_at,updated_at)
		VALUES(?,?,?,?,?,?,?,?,?,0,0,0,'',?,?)`, job.ID, job.DocumentID, job.CollectionID, job.SourceURL, job.Mode, job.Adapter, job.Title, job.Status, job.TotalItems, now, now); err != nil {
		return KnowledgeJob{}, err
	}
	for _, item := range items {
		if _, err := tx.Exec(`INSERT INTO knowledge_job_items(job_id,ordinal,source_url,mime_type,status,error,created_at,updated_at) VALUES(?,?,?,?,?,'',?,?)`,
			job.ID, item.Ordinal, item.SourceURL, item.MIMEType, "pending", now, now); err != nil {
			return KnowledgeJob{}, err
		}
	}
	if err := tx.Commit(); err != nil {
		return KnowledgeJob{}, err
	}
	return job, nil
}

func (d *DB) KnowledgeJobs(limit int) ([]KnowledgeJob, error) {
	if limit < 1 || limit > 200 {
		limit = 50
	}
	rows, err := d.conn.Query(`SELECT id,document_id,collection_id,source_url,mode,adapter,title,status,total_items,completed_items,failed_items,current_item,error,created_at,updated_at
		FROM knowledge_jobs ORDER BY updated_at DESC,id LIMIT ?`, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	items := []KnowledgeJob{}
	for rows.Next() {
		item, err := scanKnowledgeJob(rows)
		if err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	return items, rows.Err()
}

func (d *DB) KnowledgeJob(id string) (KnowledgeJob, error) {
	return scanKnowledgeJob(d.conn.QueryRow(`SELECT id,document_id,collection_id,source_url,mode,adapter,title,status,total_items,completed_items,failed_items,current_item,error,created_at,updated_at FROM knowledge_jobs WHERE id=?`, id))
}

func (d *DB) KnowledgeJobForDocument(documentID string) (KnowledgeJob, error) {
	return scanKnowledgeJob(d.conn.QueryRow(`SELECT id,document_id,collection_id,source_url,mode,adapter,title,status,total_items,completed_items,failed_items,current_item,error,created_at,updated_at FROM knowledge_jobs WHERE document_id=? ORDER BY created_at DESC LIMIT 1`, documentID))
}

func (d *DB) KnowledgeJobIDsForCollection(collectionID int64) ([]string, error) {
	rows, err := d.conn.Query(`SELECT id FROM knowledge_jobs WHERE collection_id=? AND status IN ('queued','running') ORDER BY created_at`, collectionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	return ids, rows.Err()
}

type knowledgeJobScanner interface{ Scan(...any) error }

func scanKnowledgeJob(row knowledgeJobScanner) (KnowledgeJob, error) {
	var item KnowledgeJob
	err := row.Scan(&item.ID, &item.DocumentID, &item.CollectionID, &item.SourceURL, &item.Mode, &item.Adapter, &item.Title, &item.Status, &item.TotalItems, &item.CompletedItems, &item.FailedItems, &item.CurrentItem, &item.Error, &item.CreatedAt, &item.UpdatedAt)
	return item, err
}

func (d *DB) KnowledgeJobItems(id string, retryFailed bool) ([]KnowledgeJobItem, error) {
	where := "status='pending'"
	if retryFailed {
		where = "status IN ('pending','failed','running')"
	}
	rows, err := d.conn.Query(`SELECT job_id,ordinal,source_url,mime_type,status,error,created_at,updated_at FROM knowledge_job_items WHERE job_id=? AND `+where+` ORDER BY ordinal`, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	items := []KnowledgeJobItem{}
	for rows.Next() {
		var item KnowledgeJobItem
		if err := rows.Scan(&item.JobID, &item.Ordinal, &item.SourceURL, &item.MIMEType, &item.Status, &item.Error, &item.CreatedAt, &item.UpdatedAt); err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	return items, rows.Err()
}

func (d *DB) KnowledgeJobFailures(id string, limit int) ([]KnowledgeJobItem, error) {
	if limit < 1 || limit > 100 {
		limit = 20
	}
	rows, err := d.conn.Query(`SELECT job_id,ordinal,source_url,mime_type,status,error,created_at,updated_at FROM knowledge_job_items WHERE job_id=? AND status='failed' ORDER BY ordinal LIMIT ?`, id, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	items := []KnowledgeJobItem{}
	for rows.Next() {
		var item KnowledgeJobItem
		if err := rows.Scan(&item.JobID, &item.Ordinal, &item.SourceURL, &item.MIMEType, &item.Status, &item.Error, &item.CreatedAt, &item.UpdatedAt); err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	return items, rows.Err()
}

func (d *DB) SetKnowledgeJobStatus(id, status, detail string, current int) error {
	result, err := d.conn.Exec(`UPDATE knowledge_jobs SET status=?,error=?,current_item=?,updated_at=? WHERE id=?`, status, detail, current, time.Now(), id)
	if err != nil {
		return err
	}
	if changed, _ := result.RowsAffected(); changed == 0 {
		return sql.ErrNoRows
	}
	return nil
}

func (d *DB) SetKnowledgeJobItemStatus(jobID string, ordinal int, status, detail string) error {
	tx, err := d.conn.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	now := time.Now()
	if _, err := tx.Exec(`UPDATE knowledge_job_items SET status=?,error=?,updated_at=? WHERE job_id=? AND ordinal=?`, status, detail, now, jobID, ordinal); err != nil {
		return err
	}
	if _, err := tx.Exec(`UPDATE knowledge_jobs SET
		completed_items=(SELECT count(*) FROM knowledge_job_items WHERE job_id=? AND status='completed'),
		failed_items=(SELECT count(*) FROM knowledge_job_items WHERE job_id=? AND status='failed'),updated_at=? WHERE id=?`, jobID, jobID, now, jobID); err != nil {
		return err
	}
	return tx.Commit()
}

func (d *DB) ResetKnowledgeJobFailures(id string) error {
	tx, err := d.conn.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	now := time.Now()
	if _, err := tx.Exec(`UPDATE knowledge_job_items SET status='pending',error='',updated_at=? WHERE job_id=? AND status IN ('failed','running')`, now, id); err != nil {
		return err
	}
	if _, err := tx.Exec(`UPDATE knowledge_jobs SET status='queued',failed_items=0,error='',updated_at=? WHERE id=?`, now, id); err != nil {
		return err
	}
	return tx.Commit()
}

func (d *DB) RecoverKnowledgeJobs() ([]KnowledgeJob, error) {
	if _, err := d.conn.Exec(`UPDATE knowledge_job_items SET status='pending',error='' WHERE status='running'`); err != nil {
		return nil, err
	}
	if _, err := d.conn.Exec(`UPDATE knowledge_jobs SET status='queued',error='recovered after restart',updated_at=? WHERE status IN ('running','queued')`, time.Now()); err != nil {
		return nil, err
	}
	rows, err := d.conn.Query(`SELECT id,document_id,collection_id,source_url,mode,adapter,title,status,total_items,completed_items,failed_items,current_item,error,created_at,updated_at FROM knowledge_jobs WHERE status='queued' ORDER BY created_at`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	items := []KnowledgeJob{}
	for rows.Next() {
		item, err := scanKnowledgeJob(rows)
		if err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	return items, rows.Err()
}

func (d *DB) UpsertKnowledgeAsset(item KnowledgeAsset) (KnowledgeAsset, string, error) {
	var old string
	_ = d.conn.QueryRow(`SELECT storage_path FROM knowledge_assets WHERE document_id=? AND kind=? AND ordinal=?`, item.DocumentID, item.Kind, item.Ordinal).Scan(&old)
	now := time.Now()
	_, err := d.conn.Exec(`INSERT INTO knowledge_assets(document_id,kind,ordinal,source_url,mime_type,size_bytes,sha256,storage_path,status,error,created_at,updated_at)
		VALUES(?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(document_id,kind,ordinal) DO UPDATE SET source_url=excluded.source_url,mime_type=excluded.mime_type,size_bytes=excluded.size_bytes,sha256=excluded.sha256,storage_path=excluded.storage_path,status=excluded.status,error=excluded.error,updated_at=excluded.updated_at`,
		item.DocumentID, item.Kind, item.Ordinal, item.SourceURL, item.MIMEType, item.SizeBytes, item.SHA256, item.StoragePath, item.Status, item.Error, now, now)
	if err != nil {
		return KnowledgeAsset{}, old, err
	}
	row := d.conn.QueryRow(`SELECT id,document_id,kind,ordinal,source_url,mime_type,size_bytes,sha256,storage_path,status,error,created_at,updated_at FROM knowledge_assets WHERE document_id=? AND kind=? AND ordinal=?`, item.DocumentID, item.Kind, item.Ordinal)
	err = row.Scan(&item.ID, &item.DocumentID, &item.Kind, &item.Ordinal, &item.SourceURL, &item.MIMEType, &item.SizeBytes, &item.SHA256, &item.StoragePath, &item.Status, &item.Error, &item.CreatedAt, &item.UpdatedAt)
	return item, old, err
}

func (d *DB) ReplaceKnowledgePageChunks(documentID string, page int, chunks []KnowledgeChunk, totalPages int) error {
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
	result, err := tx.Exec(`UPDATE knowledge_documents SET page_count=?,chunk_count=(SELECT count(*) FROM knowledge_chunks WHERE document_id=?),updated_at=? WHERE id=?`, totalPages, documentID, now, documentID)
	if err != nil {
		return err
	}
	if changed, _ := result.RowsAffected(); changed == 0 {
		return sql.ErrNoRows
	}
	return tx.Commit()
}
