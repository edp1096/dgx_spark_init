package server

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/knowledge"
	"sparktalk/internal/llm"
)

func TestKnowledgeURLCollectionPreservesSourceAndIndexesNormalizedText(t *testing.T) {
	collector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var input map[string]any
		if r.URL.Path != "/v1/collect" || json.NewDecoder(r.Body).Decode(&input) != nil || input["mode"] != "auto" {
			t.Fatalf("unexpected collector request: path=%s input=%+v", r.URL.Path, input)
		}
		var output bytes.Buffer
		archive := zip.NewWriter(&output)
		manifest, _ := archive.Create("manifest.json")
		_ = json.NewEncoder(manifest).Encode(map[string]any{
			"version": 1, "requested_url": input["url"], "final_url": "https://example.com/final",
			"title": "수집 자료", "method": "browser", "content_type": "text/html",
			"raw_path": "raw/page.html", "fetched_at": time.Now(),
		})
		raw, _ := archive.Create("raw/page.html")
		_, _ = raw.Write([]byte("<html><body>보존할 원문</body></html>"))
		text, _ := archive.Create("normalized/text.txt")
		_, _ = text.Write([]byte("동적 페이지에서 정규화한 검색 본문"))
		links, _ := archive.Create("normalized/links.json")
		_, _ = links.Write([]byte(`[{"text":"PDF","url":"https://example.com/data.pdf"}]`))
		_ = archive.Close()
		w.Header().Set("Content-Type", "application/zip")
		_, _ = w.Write(output.Bytes())
	}))
	defer collector.Close()

	databasePath := filepath.Join(t.TempDir(), "sparktalk.db")
	store, err := db.Open(databasePath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	fileStore, _ := knowledge.New(databasePath)
	server := &Server{db: store, knowledge: fileStore, knowledgeIndex: &knowledge.Extractor{}, collector: knowledge.NewCollectorClient(collector.URL)}
	body := strings.NewReader(`{"collection_id":1,"url":"https://example.com/start","mode":"auto"}`)
	request := httptest.NewRequest(http.MethodPost, "/api/knowledge/sources", body)
	response := httptest.NewRecorder()
	server.collectKnowledgeSource(response, request)
	if response.Code != http.StatusCreated {
		t.Fatalf("collect status=%d body=%s", response.Code, response.Body.String())
	}
	var collected knowledgeSourceResponse
	if err := json.Unmarshal(response.Body.Bytes(), &collected); err != nil {
		t.Fatal(err)
	}
	document := collected.Document
	if document.Status != "ready" || document.SourceKind != "url" || document.SourceURL != "https://example.com/final" {
		t.Fatalf("unexpected URL document: %+v", document)
	}
	if len(collected.Links) != 1 || collected.Links[0].URL != "https://example.com/data.pdf" {
		t.Fatalf("discovered links missing: %+v", collected.Links)
	}
	results, err := store.SearchKnowledge("정규화 검색", 1, 5)
	if err != nil || len(results) != 1 {
		t.Fatalf("collected text was not indexed: results=%+v err=%v", results, err)
	}
	toolResult, err := server.executeKnowledgeSearch(context.Background(), llm.ToolCall{Function: llm.FunctionCall{Name: "knowledge_search", Arguments: `{"query":"정규화 검색"}`}})
	if err != nil || !strings.Contains(toolResult, `"source_url":"https://example.com/final"`) || !strings.Contains(toolResult, `"stored_source_url":"/api/knowledge/`) {
		t.Fatalf("collected source attribution is incomplete: result=%s err=%v", toolResult, err)
	}
	source := httptest.NewRecorder()
	server.knowledgeDocument(source, httptest.NewRequest(http.MethodGet, "/api/knowledge/documents/"+document.ID+"/source", nil))
	if source.Code != http.StatusOK || !strings.Contains(source.Body.String(), "보존할 원문") {
		t.Fatalf("collected raw source missing: status=%d body=%q", source.Code, source.Body.String())
	}
}

func TestKnowledgeUploadSearchSourceAndToolRegistry(t *testing.T) {
	databasePath := filepath.Join(t.TempDir(), "sparktalk.db")
	store, err := db.Open(databasePath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	fileStore, err := knowledge.New(databasePath)
	if err != nil {
		t.Fatal(err)
	}
	server := &Server{db: store, knowledge: fileStore, knowledgeIndex: &knowledge.Extractor{}}
	collections, err := store.KnowledgeCollections()
	if err != nil || len(collections) != 1 {
		t.Fatalf("collections=%+v err=%v", collections, err)
	}

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	if err := writer.WriteField("collection_id", "1"); err != nil {
		t.Fatal(err)
	}
	part, err := writer.CreateFormFile("file", "manual.md")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := part.Write([]byte("# 장치 안내\n\n스파크톡 지식 도구는 원문 위치를 인용합니다.")); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPost, "/api/knowledge/documents", &body)
	request.Header.Set("Content-Type", writer.FormDataContentType())
	response := httptest.NewRecorder()
	server.knowledgeDocuments(response, request)
	if response.Code != http.StatusCreated {
		t.Fatalf("upload status=%d body=%s", response.Code, response.Body.String())
	}
	var document db.KnowledgeDocument
	if err := json.Unmarshal(response.Body.Bytes(), &document); err != nil {
		t.Fatal(err)
	}
	if document.Status != "ready" || document.ChunkCount != 1 {
		t.Fatalf("unexpected document: %+v", document)
	}

	registry := newCompletionToolRegistry(server, "session", config.ToolsConfig{}, false, nil)
	if _, ok := registry.handlers["knowledge_search"]; !ok {
		t.Fatalf("knowledge tools were not registered: %+v", registry.handlers)
	}
	search, err := registry.execute(context.Background(), llm.ToolCall{Function: llm.FunctionCall{Name: "knowledge_search", Arguments: `{"query":"원문 위치"}`}}, nil, nil)
	if err != nil || !strings.Contains(search.Result, document.ID) || !strings.Contains(search.Result, "source_url") {
		t.Fatalf("search=%s err=%v", search.Result, err)
	}
	read, err := registry.execute(context.Background(), llm.ToolCall{Function: llm.FunctionCall{Name: "knowledge_read", Arguments: `{"document_id":"` + document.ID + `","chunk":0}`}}, nil, nil)
	if err != nil || !strings.Contains(read.Result, "스파크톡 지식 도구") {
		t.Fatalf("read=%s err=%v", read.Result, err)
	}

	sourceRequest := httptest.NewRequest(http.MethodGet, "/api/knowledge/documents/"+document.ID+"/source", nil)
	sourceResponse := httptest.NewRecorder()
	server.knowledgeDocument(sourceResponse, sourceRequest)
	if sourceResponse.Code != http.StatusOK || !strings.Contains(sourceResponse.Body.String(), "스파크톡 지식 도구") {
		t.Fatalf("source status=%d body=%q", sourceResponse.Code, sourceResponse.Body.String())
	}
}

func TestKnowledgeImageOCRUsesVisionModelAndIndexesResult(t *testing.T) {
	modelServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatal(err)
		}
		encoded, _ := json.Marshal(payload["messages"])
		if !strings.Contains(string(encoded), "image_url") || !strings.Contains(string(encoded), "Transcribe every visible character") {
			t.Fatalf("OCR request is missing image or exact-transcription prompt: %s", encoded)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":"스캔 문서의 정확한 글자"}}]}`)
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer modelServer.Close()

	databasePath := filepath.Join(t.TempDir(), "sparktalk.db")
	store, err := db.Open(databasePath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	fileStore, err := knowledge.New(databasePath)
	if err != nil {
		t.Fatal(err)
	}
	server := &Server{
		db: store, knowledge: fileStore, knowledgeIndex: &knowledge.Extractor{},
		cfg: config.Config{Model: config.ModelConfig{Endpoint: modelServer.URL, DefaultModel: "vision"}},
		llm: llm.New(modelServer.URL, "vision", ""),
	}
	var imageData bytes.Buffer
	if err := png.Encode(&imageData, image.NewRGBA(image.Rect(0, 0, 2, 2))); err != nil {
		t.Fatal(err)
	}
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	_ = writer.WriteField("collection_id", "1")
	part, _ := writer.CreateFormFile("file", "scan.png")
	_, _ = part.Write(imageData.Bytes())
	_ = writer.Close()
	upload := httptest.NewRequest(http.MethodPost, "/api/knowledge/documents", &body)
	upload.Header.Set("Content-Type", writer.FormDataContentType())
	uploadResponse := httptest.NewRecorder()
	server.knowledgeDocuments(uploadResponse, upload)
	var document db.KnowledgeDocument
	if uploadResponse.Code != http.StatusCreated || json.Unmarshal(uploadResponse.Body.Bytes(), &document) != nil || document.Status != "needs_ocr" {
		t.Fatalf("image upload status=%d document=%+v body=%s", uploadResponse.Code, document, uploadResponse.Body.String())
	}
	requestContext, cancelRequest := context.WithCancel(context.Background())
	cancelRequest()
	ocrRequest := httptest.NewRequest(http.MethodPost, "/api/knowledge/documents/"+document.ID+"/ocr", nil).WithContext(requestContext)
	ocrResponse := httptest.NewRecorder()
	server.knowledgeDocument(ocrResponse, ocrRequest)
	if ocrResponse.Code != http.StatusAccepted {
		t.Fatalf("OCR status=%d body=%s", ocrResponse.Code, ocrResponse.Body.String())
	}
	deadline := time.Now().Add(3 * time.Second)
	for {
		updated, err := store.KnowledgeDocument(document.ID)
		if err != nil {
			t.Fatal(err)
		}
		if updated.Status == "ready" {
			if updated.OCRProcessedPages != 1 || updated.OCRTotalPages != 1 {
				t.Fatalf("OCR progress was not completed: %+v", updated)
			}
			break
		}
		if updated.Status == "needs_ocr" || time.Now().After(deadline) {
			t.Fatalf("OCR did not complete: %+v", updated)
		}
		time.Sleep(10 * time.Millisecond)
	}
	results, err := store.SearchKnowledge("정확한 글자", 1, 5)
	if err != nil || len(results) != 1 || !strings.Contains(results[0].Content, "스캔 문서") {
		t.Fatalf("OCR was not indexed: results=%+v err=%v", results, err)
	}
}

func TestKnowledgeOCRRejectsHTMLBeforeCallingVisionModel(t *testing.T) {
	_, err := recognizeKnowledgePage(context.Background(), nil, "vision", 1, []byte("<html><body>not an image</body></html>"))
	if err == nil || !strings.Contains(err.Error(), "instead of an image") {
		t.Fatalf("HTML was accepted as an OCR image: %v", err)
	}
}

func TestKnowledgePublicationCreatesResumableJobAndIndexesPages(t *testing.T) {
	var collector *httptest.Server
	collector = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var input struct {
			URL string `json:"url"`
		}
		if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
			t.Fatal(err)
		}
		var output bytes.Buffer
		archive := zip.NewWriter(&output)
		manifest, _ := archive.Create("manifest.json")
		_ = json.NewEncoder(manifest).Encode(map[string]any{
			"version": 1, "requested_url": input.URL, "final_url": input.URL,
			"title": "시험 전자책", "method": "direct", "content_type": "text/plain",
			"raw_path": "raw/page.txt", "fetched_at": time.Now(),
		})
		raw, _ := archive.Create("raw/page.txt")
		_, _ = raw.Write([]byte("보존 원문 " + input.URL))
		if strings.HasSuffix(input.URL, "/viewer") {
			publication, _ := archive.Create("normalized/publication.json")
			_ = json.NewEncoder(publication).Encode(map[string]any{
				"adapter": "test-viewer", "title": "시험 전자책", "page_count": 2,
				"pages": []map[string]any{
					{"number": 1, "url": collector.URL + "/page-1", "mime_type": "text/plain"},
					{"number": 2, "url": collector.URL + "/page-2", "mime_type": "text/plain"},
				},
			})
		} else {
			text, _ := archive.Create("normalized/text.txt")
			_, _ = text.Write([]byte("검색할 전자책 " + strings.TrimPrefix(input.URL, collector.URL+"/")))
		}
		_ = archive.Close()
		w.Header().Set("Content-Type", "application/zip")
		_, _ = w.Write(output.Bytes())
	}))
	defer collector.Close()

	databasePath := filepath.Join(t.TempDir(), "sparktalk.db")
	store, err := db.Open(databasePath)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	fileStore, _ := knowledge.New(databasePath)
	srv := &Server{db: store, knowledge: fileStore, knowledgeIndex: &knowledge.Extractor{}, collector: knowledge.NewCollectorClient(collector.URL)}
	request := httptest.NewRequest(http.MethodPost, "/api/knowledge/sources", strings.NewReader(`{"collection_id":1,"url":"`+collector.URL+`/viewer","mode":"auto"}`))
	response := httptest.NewRecorder()
	srv.collectKnowledgeSource(response, request)
	if response.Code != http.StatusCreated {
		t.Fatalf("collect status=%d body=%s", response.Code, response.Body.String())
	}
	var result knowledgeSourceResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil || result.Job == nil || result.Job.Status != "paused" || result.Job.TotalItems != 2 {
		t.Fatalf("publication job missing: result=%+v err=%v", result, err)
	}
	start := httptest.NewRecorder()
	srv.knowledgeJobAction(start, httptest.NewRequest(http.MethodPost, "/api/knowledge/jobs/"+result.Job.ID+"/resume", nil))
	if start.Code != http.StatusOK {
		t.Fatalf("resume status=%d body=%s", start.Code, start.Body.String())
	}
	deadline := time.Now().Add(3 * time.Second)
	for {
		job, err := store.KnowledgeJob(result.Job.ID)
		if err != nil {
			t.Fatal(err)
		}
		if job.Status == "completed" {
			break
		}
		if job.Status == "failed" || time.Now().After(deadline) {
			t.Fatalf("job did not complete: %+v", job)
		}
		time.Sleep(10 * time.Millisecond)
	}
	results, err := store.SearchKnowledge("page-2", 1, 5)
	foundPageTwo := false
	for _, item := range results {
		foundPageTwo = foundPageTwo || item.PageStart == 2
	}
	if err != nil || !foundPageTwo {
		t.Fatalf("publication page was not indexed: results=%+v err=%v", results, err)
	}
}
