package db

import "database/sql"

func migrateKnowledge(conn *sql.DB) error {
	if _, err := conn.Exec(`
		CREATE TABLE IF NOT EXISTS knowledge_collections (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			name TEXT NOT NULL,
			description TEXT NOT NULL DEFAULT '',
			enabled INTEGER NOT NULL DEFAULT 1,
			created_at DATETIME NOT NULL,
			updated_at DATETIME NOT NULL
		);
		CREATE INDEX IF NOT EXISTS idx_knowledge_collections_updated
			ON knowledge_collections(enabled, updated_at DESC);
		CREATE TABLE IF NOT EXISTS knowledge_documents (
			id TEXT PRIMARY KEY,
			collection_id INTEGER NOT NULL REFERENCES knowledge_collections(id) ON DELETE CASCADE,
			title TEXT NOT NULL,
			source_name TEXT NOT NULL DEFAULT '',
			source_url TEXT NOT NULL DEFAULT '',
			source_kind TEXT NOT NULL DEFAULT 'file',
			mime_type TEXT NOT NULL DEFAULT '',
			size_bytes INTEGER NOT NULL DEFAULT 0,
			sha256 TEXT NOT NULL,
			storage_path TEXT NOT NULL,
			status TEXT NOT NULL DEFAULT 'processing',
			error TEXT NOT NULL DEFAULT '',
			page_count INTEGER NOT NULL DEFAULT 0,
			chunk_count INTEGER NOT NULL DEFAULT 0,
			ocr_processed_pages INTEGER NOT NULL DEFAULT 0,
			ocr_total_pages INTEGER NOT NULL DEFAULT 0,
			created_at DATETIME NOT NULL,
			updated_at DATETIME NOT NULL,
			UNIQUE(collection_id, sha256)
		);
		CREATE INDEX IF NOT EXISTS idx_knowledge_documents_collection
			ON knowledge_documents(collection_id, updated_at DESC);
		CREATE INDEX IF NOT EXISTS idx_knowledge_documents_storage
			ON knowledge_documents(storage_path);
		CREATE TABLE IF NOT EXISTS knowledge_chunks (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			document_id TEXT NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
			ordinal INTEGER NOT NULL,
			page_start INTEGER NOT NULL DEFAULT 0,
			page_end INTEGER NOT NULL DEFAULT 0,
			heading TEXT NOT NULL DEFAULT '',
			content TEXT NOT NULL,
			created_at DATETIME NOT NULL,
			UNIQUE(document_id, ordinal)
		);
		CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_document
			ON knowledge_chunks(document_id, ordinal);
		CREATE TABLE IF NOT EXISTS knowledge_assets (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			document_id TEXT NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
			kind TEXT NOT NULL,
			ordinal INTEGER NOT NULL DEFAULT 0,
			source_url TEXT NOT NULL DEFAULT '',
			mime_type TEXT NOT NULL DEFAULT '',
			size_bytes INTEGER NOT NULL DEFAULT 0,
			sha256 TEXT NOT NULL DEFAULT '',
			storage_path TEXT NOT NULL DEFAULT '',
			status TEXT NOT NULL DEFAULT 'pending',
			error TEXT NOT NULL DEFAULT '',
			created_at DATETIME NOT NULL,
			updated_at DATETIME NOT NULL,
			UNIQUE(document_id, kind, ordinal)
		);
		CREATE INDEX IF NOT EXISTS idx_knowledge_assets_document
			ON knowledge_assets(document_id, kind, ordinal);
		CREATE INDEX IF NOT EXISTS idx_knowledge_assets_storage
			ON knowledge_assets(storage_path);
		CREATE TABLE IF NOT EXISTS knowledge_jobs (
			id TEXT PRIMARY KEY,
			document_id TEXT NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
			collection_id INTEGER NOT NULL REFERENCES knowledge_collections(id) ON DELETE CASCADE,
			source_url TEXT NOT NULL,
			mode TEXT NOT NULL DEFAULT 'auto',
			adapter TEXT NOT NULL DEFAULT '',
			title TEXT NOT NULL DEFAULT '',
			status TEXT NOT NULL DEFAULT 'paused',
			total_items INTEGER NOT NULL DEFAULT 0,
			completed_items INTEGER NOT NULL DEFAULT 0,
			failed_items INTEGER NOT NULL DEFAULT 0,
			current_item INTEGER NOT NULL DEFAULT 0,
			error TEXT NOT NULL DEFAULT '',
			created_at DATETIME NOT NULL,
			updated_at DATETIME NOT NULL
		);
		CREATE INDEX IF NOT EXISTS idx_knowledge_jobs_updated
			ON knowledge_jobs(status, updated_at DESC);
		CREATE TABLE IF NOT EXISTS knowledge_job_items (
			job_id TEXT NOT NULL REFERENCES knowledge_jobs(id) ON DELETE CASCADE,
			ordinal INTEGER NOT NULL,
			source_url TEXT NOT NULL,
			mime_type TEXT NOT NULL DEFAULT '',
			status TEXT NOT NULL DEFAULT 'pending',
			error TEXT NOT NULL DEFAULT '',
			created_at DATETIME NOT NULL,
			updated_at DATETIME NOT NULL,
			PRIMARY KEY(job_id, ordinal)
		);
		CREATE INDEX IF NOT EXISTS idx_knowledge_job_items_status
			ON knowledge_job_items(job_id, status, ordinal);
		CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_search USING fts5(
			document_id UNINDEXED, chunk_id UNINDEXED, title, heading, content,
			tokenize='trigram'
		);
		CREATE TRIGGER IF NOT EXISTS knowledge_chunks_search_insert AFTER INSERT ON knowledge_chunks BEGIN
			INSERT INTO knowledge_search(rowid,document_id,chunk_id,title,heading,content)
				SELECT NEW.id,NEW.document_id,NEW.id,document.title,NEW.heading,NEW.content
				FROM knowledge_documents AS document WHERE document.id=NEW.document_id;
		END;
		CREATE TRIGGER IF NOT EXISTS knowledge_chunks_search_update AFTER UPDATE ON knowledge_chunks BEGIN
			DELETE FROM knowledge_search WHERE rowid=OLD.id;
			INSERT INTO knowledge_search(rowid,document_id,chunk_id,title,heading,content)
				SELECT NEW.id,NEW.document_id,NEW.id,document.title,NEW.heading,NEW.content
				FROM knowledge_documents AS document WHERE document.id=NEW.document_id;
		END;
		CREATE TRIGGER IF NOT EXISTS knowledge_chunks_search_delete AFTER DELETE ON knowledge_chunks BEGIN
			DELETE FROM knowledge_search WHERE rowid=OLD.id;
		END;
		CREATE TRIGGER IF NOT EXISTS knowledge_documents_search_title AFTER UPDATE OF title ON knowledge_documents BEGIN
			UPDATE knowledge_search SET title=NEW.title WHERE document_id=NEW.id;
		END;
		CREATE TABLE IF NOT EXISTS session_knowledge (
			session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
			collection_id INTEGER NOT NULL REFERENCES knowledge_collections(id) ON DELETE CASCADE,
			created_at DATETIME NOT NULL,
			PRIMARY KEY(session_id, collection_id)
		);
	`); err != nil {
		return err
	}
	// Additive migration for knowledge databases created before OCR progress
	// became persistent. Duplicate-column errors are intentionally ignored.
	_, _ = conn.Exec(`ALTER TABLE knowledge_documents ADD COLUMN ocr_processed_pages INTEGER NOT NULL DEFAULT 0`)
	_, _ = conn.Exec(`ALTER TABLE knowledge_documents ADD COLUMN ocr_total_pages INTEGER NOT NULL DEFAULT 0`)
	if _, err := conn.Exec(`
		INSERT INTO knowledge_search(rowid,document_id,chunk_id,title,heading,content)
		SELECT chunk.id,chunk.document_id,chunk.id,document.title,chunk.heading,chunk.content
		FROM knowledge_chunks AS chunk
		JOIN knowledge_documents AS document ON document.id=chunk.document_id
		WHERE NOT EXISTS(SELECT 1 FROM knowledge_search WHERE rowid=chunk.id);
	`); err != nil {
		return err
	}
	var count int
	if err := conn.QueryRow(`SELECT count(*) FROM knowledge_collections`).Scan(&count); err != nil {
		return err
	}
	if count == 0 {
		_, err := conn.Exec(`INSERT INTO knowledge_collections(name,description,enabled,created_at,updated_at) VALUES('내 지식','문서와 수집 자료',1,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)`)
		return err
	}
	return nil
}
