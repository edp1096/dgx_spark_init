package db

import (
	"encoding/json"
	"errors"
	"path/filepath"
	"strings"
	"testing"

	"sparktalk/internal/workflows"
)

func TestDocumentConversionStartupCompactsPreviousHistory(t *testing.T) {
	d, path := conversionFixture(t, false)
	j, _, err := d.BeginDocumentConversion("s", "r", "pdf", "test")
	if err != nil {
		t.Fatal(err)
	}
	if err = d.CompleteDocumentConversion("s", "r", j.ID, "application/pdf", []byte("older")); err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"latest-success", "old-failure", "latest-failure"} {
		item := j
		item.ID = id
		item.Status = "failed"
		var content []byte
		if id == "latest-success" {
			item.Status = "completed"
			content = []byte("latest")
			item.Bytes = len(content)
			item.Hash = documentContentHash(content)
			item.MIME = "application/pdf"
		}
		raw, _ := json.Marshal(item)
		if _, err = d.conn.Exec(`INSERT INTO document_conversions(id,run_id,format,status,data,content) VALUES(?,'r','pdf',?,?,?)`, id, item.Status, string(raw), content); err != nil {
			t.Fatal(err)
		}
	}
	d.Close()
	reopened, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	jobs, err := reopened.DocumentConversions("s", "r")
	if err != nil || len(jobs) != 2 || jobs[0].ID != "latest-failure" || jobs[1].ID != "latest-success" {
		t.Fatal("history was not compacted", jobs, err)
	}
	data, _, err := reopened.ConvertedDocumentOutput("s", "r", "output.pdf")
	if err != nil || string(data) != "latest" {
		t.Fatal("migration changed current file", err, string(data))
	}
}

func conversionFixture(t *testing.T, versioned bool) (*DB, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "test.db")
	d, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { d.Close() })
	if _, err = d.CreateSession("s", "test", "model", "low"); err != nil {
		t.Fatal(err)
	}
	r := workflows.Run{ID: "r", SessionID: "s", Status: "completed", Current: 2, Document: &workflows.DocumentTask{Kind: "translate", Calls: 9, ElapsedSeconds: 123, Formats: []string{"txt", "pdf"}}}
	if versioned {
		r.Document.Pipeline = workflows.DocumentPipelineVersion
		r.Document.ExecutionID = "test-execution"
	}
	if err = d.CreateDocumentRun(&r, nil); err != nil {
		t.Fatal(err)
	}
	for name, body := range map[string]string{"output.txt": "작품 제목\n\n본문 ［＃改頁］", "output.pdf": "original PDF"} {
		if err = d.SaveDocumentOutput("s", "r", name, "application/pdf", []byte(body)); err != nil {
			t.Fatal(err)
		}
	}
	if versioned {
		if err = d.PublishDocumentRevision(&r, []string{"output.txt", "output.pdf"}, "original-hash"); err != nil {
			t.Fatal(err)
		}
	}
	return d, path
}

func TestDocumentConversionPreservesPublicationAndReplacesCurrentFile(t *testing.T) {
	for _, versioned := range []bool{false, true} {
		t.Run(map[bool]string{false: "legacy", true: "published"}[versioned], func(t *testing.T) {
			d, _ := conversionFixture(t, versioned)
			var before string
			d.conn.QueryRow(`SELECT data FROM workflow_runs WHERE id='r'`).Scan(&before)
			first, _, err := d.BeginDocumentConversion("s", "r", "pdf", "test")
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err = d.BeginDocumentConversion("s", "r", "pdf", "test"); err == nil {
				t.Fatal("duplicate active conversion accepted")
			}
			if err = d.CompleteDocumentConversion("s", "r", first.ID, "application/pdf", []byte("first")); err != nil {
				t.Fatal(err)
			}
			second, _, err := d.BeginDocumentConversion("s", "r", "pdf", "test")
			if err != nil {
				t.Fatal(err)
			}
			if err = d.FailDocumentConversion("s", "r", second.ID, errors.New("offline")); err != nil {
				t.Fatal(err)
			}
			data, _, err := d.ConvertedDocumentOutput("s", "r", "output.pdf")
			if err != nil || string(data) != "first" {
				t.Fatal("failure replaced current file", err, string(data))
			}
			third, _, err := d.BeginDocumentConversion("s", "r", "pdf", "test")
			if err != nil {
				t.Fatal(err)
			}
			if err = d.CompleteDocumentConversion("s", "r", third.ID, "application/pdf", []byte("third")); err != nil {
				t.Fatal(err)
			}
			data, _, err = d.ConvertedDocumentOutput("s", "r", "output.pdf")
			if err != nil || string(data) != "third" {
				t.Fatal("latest file missing", err)
			}
			if _, _, err = d.ConvertedDocumentOutput("other", "r", "output.pdf"); err == nil {
				t.Fatal("cross-session download")
			}
			if _, _, err = d.ConvertedDocumentOutput("s", "r", "output.docx"); err == nil {
				t.Fatal("cross-format download")
			}
			var after string
			d.conn.QueryRow(`SELECT data FROM workflow_runs WHERE id='r'`).Scan(&after)
			if after != before {
				t.Fatal("translation/publication state changed")
			}
			jobs, err := d.DocumentConversions("s", "r")
			if err != nil || len(jobs) != 1 || !jobs[0].Current {
				t.Fatal(jobs, err)
			}
			if err = d.CompleteDocumentConversion("s", "r", first.ID, "application/pdf", []byte("overwrite")); err == nil {
				t.Fatal("immutable file overwritten")
			}
		})
	}
}

func TestDocumentConversionRejectsChangedTextAndRecoversInterruptedJob(t *testing.T) {
	d, path := conversionFixture(t, false)
	j, _, err := d.BeginDocumentConversion("s", "r", "pdf", "test")
	if err != nil {
		t.Fatal(err)
	}
	if err = d.SaveDocumentOutput("s", "r", "output.txt", "text/plain", []byte("changed")); err != nil {
		t.Fatal(err)
	}
	if err = d.CompleteDocumentConversion("s", "r", j.ID, "application/pdf", []byte("stale")); err == nil {
		t.Fatal("stale text published")
	}
	d.Close()
	reopened, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer reopened.Close()
	jobs, err := reopened.DocumentConversions("s", "r")
	if err != nil || len(jobs) != 1 || jobs[0].Status != "failed" || !strings.Contains(jobs[0].Error, "재시작") {
		t.Fatal(jobs, err)
	}
	data, _, err := reopened.ConvertedDocumentOutput("s", "r", "output.pdf")
	if err != nil || string(data) != "original PDF" {
		t.Fatal(err, string(data))
	}
	if _, _, err = reopened.BeginDocumentConversion("s", "r", "pdf", "test"); err != nil {
		t.Fatal("interrupted job blocked retry", err)
	}
}

func TestDocumentConversionCountsStorageAndRollsBackOnWriteFailure(t *testing.T) {
	d, _ := conversionFixture(t, false)
	j, _, err := d.BeginDocumentConversion("s", "r", "pdf", "test")
	if err != nil {
		t.Fatal(err)
	}
	if err = d.CompleteDocumentConversion("s", "r", j.ID, "application/pdf", []byte("kept")); err != nil {
		t.Fatal(err)
	}
	j, _, err = d.BeginDocumentConversion("s", "r", "pdf", "test")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = d.conn.Exec(`CREATE TRIGGER fail_conversion BEFORE UPDATE OF content ON document_conversions BEGIN SELECT RAISE(ABORT,'disk write failure'); END`); err != nil {
		t.Fatal(err)
	}
	if err = d.CompleteDocumentConversion("s", "r", j.ID, "application/pdf", []byte("new")); err == nil {
		t.Fatal("write failure ignored")
	}
	jobs, _ := d.DocumentConversions("s", "r")
	if jobs[0].Status != "running" {
		t.Fatal("partial status escaped failed write")
	}
	data, _, err := d.ConvertedDocumentOutput("s", "r", "output.pdf")
	if err != nil || string(data) != "kept" {
		t.Fatal("pruning escaped failed transaction", err)
	}
	if _, err = d.conn.Exec(`DROP TRIGGER fail_conversion`); err != nil {
		t.Fatal(err)
	}
	if err = d.CompleteDocumentConversion("s", "r", j.ID, "application/pdf", make([]byte, 1<<20)); err != nil {
		t.Fatal(err)
	}
	tx, err := d.conn.Begin()
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback()
	if err = checkDocumentStorage(tx, "s", 512<<20-(1<<19)); err == nil {
		t.Fatal("current conversion omitted from quota")
	}
}

func TestDocumentConversionRejectsFirstExportAndUnpublishedDraft(t *testing.T) {
	d, _ := conversionFixture(t, false)
	if _, _, err := d.BeginDocumentConversion("s", "r", "docx", "test"); err == nil {
		t.Fatal("first export silently treated as reconversion")
	}
	r, err := d.DocumentRun("s", "r")
	if err != nil {
		t.Fatal(err)
	}
	r.Document.Pipeline = workflows.DocumentPipelineVersion
	r.Document.DraftPublication = &workflows.DocumentPublication{}
	if err = d.SaveWorkflowRun(&r); err != nil {
		t.Fatal(err)
	}
	if _, _, err = d.BeginDocumentConversion("s", "r", "pdf", "test"); err == nil {
		t.Fatal("unfinished publication bypassed by reconversion")
	}
}
