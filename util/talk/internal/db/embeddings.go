package db

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/binary"
	"fmt"
	"math"
	"sort"
	"time"
)

// Original rows remain authoritative. Jobs invalidate vectors immediately, and
// their generation prevents an in-flight encoder from publishing stale content.
func migrateEmbeddings(conn *sql.DB) error {
	tx, err := conn.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	_, err = tx.Exec(`
CREATE TABLE IF NOT EXISTS retrieval_vectors (
 kind TEXT NOT NULL, source_id INTEGER NOT NULL, profile TEXT NOT NULL,
 segment INTEGER NOT NULL, content_hash TEXT NOT NULL, content TEXT NOT NULL,
 dimensions INTEGER NOT NULL, vector BLOB NOT NULL,
 PRIMARY KEY(kind,source_id,profile,segment)
);
CREATE INDEX IF NOT EXISTS retrieval_vectors_profile ON retrieval_vectors(profile,kind,source_id);
CREATE TABLE IF NOT EXISTS retrieval_jobs (
 kind TEXT NOT NULL, source_id INTEGER NOT NULL, generation INTEGER NOT NULL DEFAULT 1,
 retry_after INTEGER NOT NULL DEFAULT 0, error TEXT NOT NULL DEFAULT '',
 PRIMARY KEY(kind,source_id)
);
DROP VIEW IF EXISTS retrieval_sources;
CREATE VIEW retrieval_sources AS
 SELECT 'memory' kind,id source_id,title,content,source_session_id session_id,0 collection_id,
 '' document_id,0 ordinal,0 page_start,0 page_end,'' source_url,priority,
 updated_at created_at,(enabled=1 AND kind='memory' AND length(trim(content))>0) enabled,'' role,source_message_id reference_message_id FROM memories
 UNION ALL
 SELECT 'message',m.id,s.title,m.content,m.session_id,0,'',0,0,0,'','reference',
 m.created_at,(m.status='completed' AND m.role IN ('user','assistant') AND length(trim(m.content))>0),m.role,m.id
 FROM messages m JOIN sessions s ON s.id=m.session_id
 UNION ALL
 SELECT 'knowledge',k.id,d.title,k.content,'',d.collection_id,d.id,k.ordinal,
 k.page_start,k.page_end,d.source_url,'reference',d.updated_at,
 (d.status='ready' AND c.enabled=1 AND length(trim(k.content))>0),'',0
 FROM knowledge_chunks k JOIN knowledge_documents d ON d.id=k.document_id
 JOIN knowledge_collections c ON c.id=d.collection_id;
`)
	if err != nil {
		return err
	}
	for _, entry := range []struct{ table, kind, updates string }{
		{"memories", "memory", "content,title,enabled,kind,priority"},
		{"messages", "message", "content,status,role"},
		{"knowledge_chunks", "knowledge", "content,heading,page_start,page_end"},
	} {
		for _, event := range []string{"INSERT", "UPDATE OF " + entry.updates} {
			name := "insert"
			if event != "INSERT" {
				name = "update"
			}
			_, err = tx.Exec(fmt.Sprintf(`CREATE TRIGGER IF NOT EXISTS retrieval_%s_%s AFTER %s ON %s BEGIN
 INSERT INTO retrieval_jobs(kind,source_id) VALUES('%s',NEW.id)
 ON CONFLICT(kind,source_id) DO UPDATE SET generation=generation+1,retry_after=0,error='';
 END;`, entry.kind, name, event, entry.table, entry.kind))
			if err != nil {
				return err
			}
		}
		_, err = tx.Exec(fmt.Sprintf(`CREATE TRIGGER IF NOT EXISTS retrieval_%s_delete AFTER DELETE ON %s BEGIN
 DELETE FROM retrieval_vectors WHERE kind='%s' AND source_id=OLD.id;
 DELETE FROM retrieval_jobs WHERE kind='%s' AND source_id=OLD.id;
 END;`, entry.kind, entry.table, entry.kind, entry.kind))
		if err != nil {
			return err
		}
	}
	_, err = tx.Exec(`
CREATE TRIGGER IF NOT EXISTS retrieval_session_title AFTER UPDATE OF title ON sessions BEGIN
 INSERT INTO retrieval_jobs(kind,source_id) SELECT 'message',id FROM messages WHERE session_id=NEW.id
 ON CONFLICT(kind,source_id) DO UPDATE SET generation=generation+1,retry_after=0,error='';
END;
CREATE TRIGGER IF NOT EXISTS retrieval_document_title AFTER UPDATE OF title ON knowledge_documents BEGIN
 INSERT INTO retrieval_jobs(kind,source_id) SELECT 'knowledge',id FROM knowledge_chunks WHERE document_id=NEW.id
 ON CONFLICT(kind,source_id) DO UPDATE SET generation=generation+1,retry_after=0,error='';
END;`)
	if err != nil {
		return err
	}
	return tx.Commit()
}

type EmbeddingSource struct {
	Kind                        string
	ID                          int64
	Generation                  int64
	Title                       string
	Content                     string
	SessionID                   string
	CollectionID                int64
	DocumentID                  string
	Ordinal, PageStart, PageEnd int
	SourceURL, Priority         string
	Role                        string
	ReferenceMessageID          int64
	CreatedAt                   time.Time
}
type EmbeddingSegment struct {
	Text   string
	Vector []float32
}
type SemanticMatch struct {
	Source     EmbeddingSource
	Similarity float64
}

func (d *DB) EnsureEmbeddingJobs(profile string) error {
	_, err := d.conn.Exec(`INSERT INTO retrieval_jobs(kind,source_id)
 SELECT s.kind,s.source_id FROM retrieval_sources s WHERE s.enabled=1
 AND NOT EXISTS(SELECT 1 FROM retrieval_vectors v WHERE v.kind=s.kind AND v.source_id=s.source_id AND v.profile=?)
 ON CONFLICT(kind,source_id) DO NOTHING`, profile)
	return err
}

func (d *DB) PendingEmbeddings(limit int) ([]EmbeddingSource, error) {
	rows, err := d.conn.Query(`SELECT s.kind,s.source_id,j.generation,s.title,s.content
 FROM retrieval_jobs j JOIN retrieval_sources s ON s.kind=j.kind AND s.source_id=j.source_id
 WHERE s.enabled=1 AND j.retry_after<=? ORDER BY CASE s.kind WHEN 'memory' THEN 0 WHEN 'knowledge' THEN 1 ELSE 2 END,s.source_id LIMIT ?`, time.Now().Unix(), limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []EmbeddingSource
	for rows.Next() {
		var s EmbeddingSource
		if err = rows.Scan(&s.Kind, &s.ID, &s.Generation, &s.Title, &s.Content); err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return out, rows.Err()
}

func normalizeVector(values []float32) ([]byte, error) {
	if len(values) != 768 {
		return nil, fmt.Errorf("embedding dimension must be 768")
	}
	norm := 0.0
	for _, v := range values {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("non-finite embedding")
		}
		norm += float64(v) * float64(v)
	}
	if norm <= 0 {
		return nil, fmt.Errorf("zero embedding")
	}
	norm = math.Sqrt(norm)
	b := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(float32(float64(v)/norm)))
	}
	return b, nil
}

func (d *DB) CompleteEmbedding(s EmbeddingSource, profile string, segments []EmbeddingSegment) error {
	if len(segments) == 0 {
		return fmt.Errorf("empty embedding result")
	}
	blobs := make([][]byte, len(segments))
	for i, p := range segments {
		var err error
		blobs[i], err = normalizeVector(p.Vector)
		if err != nil {
			return err
		}
	}
	tx, err := d.conn.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	var generation int64
	if err = tx.QueryRow(`SELECT generation FROM retrieval_jobs WHERE kind=? AND source_id=?`, s.Kind, s.ID).Scan(&generation); err != nil {
		return err
	}
	if generation != s.Generation {
		return fmt.Errorf("embedding source changed during encoding")
	}
	var title, content string
	var enabled bool
	if err = tx.QueryRow(`SELECT title,content,enabled FROM retrieval_sources WHERE kind=? AND source_id=?`, s.Kind, s.ID).Scan(&title, &content, &enabled); err != nil {
		return err
	}
	if !enabled || title != s.Title || content != s.Content {
		return fmt.Errorf("embedding source is stale")
	}
	if _, err = tx.Exec(`DELETE FROM retrieval_vectors WHERE kind=? AND source_id=?`, s.Kind, s.ID); err != nil {
		return err
	}
	hash := fmt.Sprintf("%x", sha256.Sum256([]byte(title+"\x00"+content)))
	for i, p := range segments {
		if _, err = tx.Exec(`INSERT INTO retrieval_vectors(kind,source_id,profile,segment,content_hash,content,dimensions,vector) VALUES(?,?,?,?,?,?,768,?)`, s.Kind, s.ID, profile, i, hash, p.Text, blobs[i]); err != nil {
			return err
		}
	}
	if _, err = tx.Exec(`DELETE FROM retrieval_jobs WHERE kind=? AND source_id=? AND generation=?`, s.Kind, s.ID, s.Generation); err != nil {
		return err
	}
	return tx.Commit()
}

func (d *DB) FailEmbedding(s EmbeddingSource, detail string) error {
	_, err := d.conn.Exec(`UPDATE retrieval_jobs SET retry_after=?,error=? WHERE kind=? AND source_id=? AND generation=?`, time.Now().Add(time.Minute).Unix(), detail, s.Kind, s.ID, s.Generation)
	return err
}

// A bounded top-k is retained while SQLite rows are streamed. Vectors never
// accumulate in a process-wide cache, and filtering precedes candidate ranking.
func (d *DB) SearchSemantic(ctx context.Context, profile, kind, sessionID string, through int64, collectionID int64, query []float32, minSimilarity float64, limit int) ([]SemanticMatch, error) {
	q, err := normalizeVector(query)
	if err != nil {
		return nil, err
	}
	rows, err := d.conn.QueryContext(ctx, `SELECT s.kind,s.source_id,s.title,v.content,s.session_id,s.collection_id,s.document_id,s.ordinal,s.page_start,s.page_end,s.source_url,s.priority,s.created_at,s.role,s.reference_message_id,v.vector
 FROM retrieval_vectors v JOIN retrieval_sources s ON s.kind=v.kind AND s.source_id=v.source_id
 WHERE v.profile=? AND v.dimensions=768 AND s.kind=? AND s.enabled=1
 AND NOT EXISTS(SELECT 1 FROM retrieval_jobs j WHERE j.kind=s.kind AND j.source_id=s.source_id)
 AND (?<1 OR s.collection_id=?)
 AND (s.kind!='message' OR ((?=0 AND s.session_id!=?) OR (? >0 AND s.session_id=? AND s.source_id<=?)))
 ORDER BY s.source_id,v.segment`, profile, kind, collectionID, collectionID, through, sessionID, through, sessionID, through)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []SemanticMatch{}
	for rows.Next() {
		var m SemanticMatch
		var b []byte
		s := &m.Source
		if err = rows.Scan(&s.Kind, &s.ID, &s.Title, &s.Content, &s.SessionID, &s.CollectionID, &s.DocumentID, &s.Ordinal, &s.PageStart, &s.PageEnd, &s.SourceURL, &s.Priority, &s.CreatedAt, &s.Role, &s.ReferenceMessageID, &b); err != nil {
			return nil, err
		}
		if len(b) != len(q) {
			continue
		}
		score := 0.0
		for i := 0; i < len(b); i += 4 {
			score += float64(math.Float32frombits(binary.LittleEndian.Uint32(b[i:]))) * float64(math.Float32frombits(binary.LittleEndian.Uint32(q[i:])))
		}
		if math.IsNaN(score) || math.IsInf(score, 0) || score < minSimilarity {
			continue
		}
		m.Similarity = score
		found := -1
		for i, x := range out {
			if x.Source.ID == s.ID {
				found = i
				break
			}
		}
		if found >= 0 {
			if out[found].Similarity >= score {
				continue
			}
			out[found] = m
		} else {
			out = append(out, m)
		}
		sort.SliceStable(out, func(i, j int) bool {
			if out[i].Similarity == out[j].Similarity {
				return out[i].Source.ID < out[j].Source.ID
			}
			return out[i].Similarity > out[j].Similarity
		})
		if len(out) > limit {
			out = out[:limit]
		}
	}
	return out, rows.Err()
}

func (d *DB) HasEmbeddingVectors(profile string) (bool, error) {
	var found bool
	err := d.conn.QueryRow(`SELECT EXISTS(SELECT 1 FROM retrieval_vectors v JOIN retrieval_sources s ON s.kind=v.kind AND s.source_id=v.source_id WHERE v.profile=? AND s.enabled=1 AND NOT EXISTS(SELECT 1 FROM retrieval_jobs j WHERE j.kind=s.kind AND j.source_id=s.source_id))`, profile).Scan(&found)
	return found, err
}
func (d *DB) EmbeddingCounts(profile string) (map[string]int, error) {
	result := map[string]int{}
	for _, kind := range []string{"memory", "message", "knowledge"} {
		var total, ready int
		if err := d.conn.QueryRow(`SELECT count(*) FROM retrieval_sources WHERE kind=? AND enabled=1`, kind).Scan(&total); err != nil {
			return nil, err
		}
		if err := d.conn.QueryRow(`SELECT count(*) FROM retrieval_sources s WHERE s.kind=? AND s.enabled=1 AND EXISTS(SELECT 1 FROM retrieval_vectors v WHERE v.kind=s.kind AND v.source_id=s.source_id AND v.profile=?) AND NOT EXISTS(SELECT 1 FROM retrieval_jobs j WHERE j.kind=s.kind AND j.source_id=s.source_id)`, kind, profile).Scan(&ready); err != nil {
			return nil, err
		}
		result[kind+"_total"] = total
		result[kind+"_ready"] = ready
	}
	return result, nil
}
