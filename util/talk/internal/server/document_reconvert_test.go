package server

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"sparktalk/internal/workflows"
)

func reconversionFixture(t *testing.T) (*Server, workflows.Run) {
	t.Helper()
	s, _ := documentFixture(t, func(w http.ResponseWriter, r *http.Request) { t.Error("reconversion must never call a model") })
	run := workflows.Run{ID: "r", SessionID: "s", Status: "completed", Current: 2, Document: &workflows.DocumentTask{Kind: "translate", Phase: "completed", Calls: 11, ElapsedSeconds: 321, Total: 1, Completed: 1, Formats: []string{"txt", "pdf"}}}
	if err := s.db.CreateDocumentRun(&run, nil); err != nil {
		t.Fatal(err)
	}
	for name, body := range map[string]string{"output.txt": "인간실격\n\n다자이 오사무\n\n［＃改頁］", "output.pdf": "%PDF-1.7\nold"} {
		if err := s.db.SaveDocumentOutput("s", "r", name, "application/pdf", []byte(body)); err != nil {
			t.Fatal(err)
		}
	}
	return s, run
}

func waitReconversion(t *testing.T, s *Server, status string) []workflows.DocumentConversion {
	t.Helper()
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		jobs, err := s.db.DocumentConversions("s", "r")
		if err != nil {
			t.Fatal(err)
		}
		if len(jobs) > 0 && jobs[0].Status == status {
			return jobs
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("conversion did not reach", status)
	return nil
}

func TestDocumentReconversionSurvivesDisconnectDeduplicatesAndPreservesOldDownload(t *testing.T) {
	s, run := reconversionFixture(t)
	before, _ := json.Marshal(run)
	gate := make(chan struct{})
	entered := make(chan struct{}, 1)
	var calls atomic.Int32
	converter := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var input map[string]any
		json.NewDecoder(r.Body).Decode(&input)
		if input["text"] != "인간실격\n\n다자이 오사무\n\n［＃改頁］" || input["title"] != nil {
			t.Error("saved text changed", input)
		}
		entered <- struct{}{}
		select {
		case <-gate:
		case <-r.Context().Done():
			return
		}
		json.NewEncoder(w).Encode(map[string]any{"files": []documentConvertedFile{{Name: "document.pdf", MIME: "application/pdf", Data: base64.StdEncoding.EncodeToString([]byte("%PDF-1.7\nnew"))}}})
	}))
	defer converter.Close()
	defer close(gate)
	s.cfg.Extra.DocumentsEnabled = true
	s.cfg.Extra.DocumentsEndpoint = converter.URL
	ctx, cancel := context.WithCancel(context.Background())
	req := httptest.NewRequest("POST", "/api/document-tasks/r/reexport?session_id=s", strings.NewReader(`{"format":"pdf"}`)).WithContext(ctx)
	w := httptest.NewRecorder()
	s.documentTaskAPI(w, req)
	cancel()
	if w.Code != http.StatusAccepted {
		t.Fatal(w.Code, w.Body.String())
	}
	select {
	case <-entered:
	case <-time.After(time.Second):
		t.Fatal("conversion did not start")
	}
	if _, err := s.documentAction(context.Background(), "s", documentStart{Action: "reexport", RunID: "r", Format: "pdf"}); err != nil {
		t.Fatal(err)
	}
	jobs, _ := s.db.DocumentConversions("s", "r")
	if len(jobs) != 1 || calls.Load() != 1 {
		t.Fatal("duplicate conversion", jobs, calls.Load())
	}
	download := func(session, id string) *httptest.ResponseRecorder {
		w := httptest.NewRecorder()
		s.documentTaskAPI(w, httptest.NewRequest("GET", "/api/document-tasks/r/output.pdf?session_id="+session+"&conversion="+id, nil))
		return w
	}
	if w := download("s", ""); w.Code != 200 || w.Body.String() != "%PDF-1.7\nold" {
		t.Fatal("old file unavailable during conversion", w.Code, w.Body.String())
	}
	gate <- struct{}{}
	jobs = waitReconversion(t, s, "completed")
	if w := download("s", ""); w.Code != 200 || w.Body.String() != "%PDF-1.7\nnew" || w.Header().Get("Cache-Control") != "no-store" {
		t.Fatal(w.Code, w.Body.String())
	}
	if w := download("s", "original"); w.Body.String() != "%PDF-1.7\nnew" {
		t.Fatal("obsolete history link did not resolve current file")
	}
	if w := download("other", jobs[0].ID); w.Code != 404 {
		t.Fatal("cross-session file leak", w.Code)
	}
	if w := download("s", "does-not-exist"); w.Code != 200 || w.Body.String() != "%PDF-1.7\nnew" {
		t.Fatal("obsolete revision parameter changed the download", w.Code)
	}
	latest, _ := s.db.DocumentRun("s", "r")
	after, _ := json.Marshal(latest)
	if !bytes.Equal(before, after) {
		t.Fatal("translation state changed")
	}
	view := s.documentResultView(latest)
	if view["outputs"].(map[string]string)["pdf"] != "/api/document-tasks/r/output.pdf?session_id=s" {
		t.Fatal("new download URL missing", view)
	}
}

func TestDocumentReconversionRejectsMalformedOutputWithoutReplacingCurrentFile(t *testing.T) {
	for _, kind := range []string{"http-error", "duplicate", "missing", "bad-base64", "wrong-mime", "bad-signature", "unexpected-second"} {
		t.Run(kind, func(t *testing.T) {
			s, run := reconversionFixture(t)
			converter := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if kind == "http-error" {
					http.Error(w, "offline", 503)
					return
				}
				file := documentConvertedFile{Name: "document.pdf", MIME: "application/pdf", Data: base64.StdEncoding.EncodeToString([]byte("%PDF-1.7\nnew"))}
				files := []documentConvertedFile{file}
				switch kind {
				case "duplicate":
					files = append(files, file)
				case "missing":
					files = nil
				case "bad-base64":
					files[0].Data = "!!!"
				case "wrong-mime":
					files[0].MIME = "text/html"
				case "bad-signature":
					files[0].Data = base64.StdEncoding.EncodeToString([]byte("oops"))
				case "unexpected-second":
					file.Name = "../../other.pdf"
					files = append(files, file)
				}
				json.NewEncoder(w).Encode(map[string]any{"files": files})
			}))
			defer converter.Close()
			s.cfg.Extra.DocumentsEnabled = true
			s.cfg.Extra.DocumentsEndpoint = converter.URL
			if _, err := s.startDocumentReconversion(run, "pdf"); err != nil {
				t.Fatal(err)
			}
			jobs := waitReconversion(t, s, "failed")
			if jobs[0].Error == "" {
				t.Fatal("missing failure detail")
			}
			data, _, err := s.db.ConvertedDocumentOutput("s", "r", "output.pdf")
			if err != nil || string(data) != "%PDF-1.7\nold" {
				t.Fatal("failed conversion replaced old file", err)
			}
		})
	}
}

func TestDocumentReconversionOnlyPublishesRequestedFormat(t *testing.T) {
	s, run := reconversionFixture(t)
	var zipped bytes.Buffer
	z := zip.NewWriter(&zipped)
	f, _ := z.Create("word/document.xml")
	fmt.Fprint(f, "<document/>")
	z.Close()
	if err := s.db.SaveDocumentOutput("s", "r", "output.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", zipped.Bytes()); err != nil {
		t.Fatal(err)
	}
	converter := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"files": []documentConvertedFile{
			{Name: "document.docx", MIME: "application/vnd.openxmlformats-officedocument.wordprocessingml.document", Data: base64.StdEncoding.EncodeToString(zipped.Bytes())},
			{Name: "document.pdf", MIME: "application/pdf", Data: base64.StdEncoding.EncodeToString([]byte("%PDF-1.7\nsidecar"))},
		}})
	}))
	defer converter.Close()
	s.cfg.Extra.DocumentsEnabled = true
	s.cfg.Extra.DocumentsEndpoint = converter.URL
	if _, err := s.startDocumentReconversion(run, "docx"); err != nil {
		t.Fatal(err)
	}
	waitReconversion(t, s, "completed")
	data, _, err := s.db.ConvertedDocumentOutput("s", "r", "output.pdf")
	if err != nil || string(data) != "%PDF-1.7\nold" {
		t.Fatal("DOCX changed PDF", err)
	}
	w := httptest.NewRecorder()
	s.documentTaskAPI(w, httptest.NewRequest("GET", "/api/document-tasks/r/output.docx?session_id=s", nil))
	if w.Code != 200 || !bytes.Equal(w.Body.Bytes(), zipped.Bytes()) {
		t.Fatal("new format download missing", w.Code)
	}
}
