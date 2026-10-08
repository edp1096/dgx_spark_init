package db

import (
	"path/filepath"
	"testing"
)

func TestKnowledgeCollectionsIndexAndCascade(t *testing.T) {
	store, err := Open(filepath.Join(t.TempDir(), "knowledge.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	collections, err := store.KnowledgeCollections()
	if err != nil {
		t.Fatal(err)
	}
	if len(collections) != 1 || collections[0].Name != "내 지식" {
		t.Fatalf("default collection missing: %+v", collections)
	}
	collection, err := store.AddKnowledgeCollection("제품 자료", "시험")
	if err != nil {
		t.Fatal(err)
	}
	document, duplicate, err := store.AddKnowledgeDocument(KnowledgeDocument{
		ID: "0123456789abcdef0123456789abcdef", CollectionID: collection.ID,
		Title: "스파크톡 안내서", SourceName: "guide.txt", SourceKind: "file",
		MIMEType: "text/plain", SizeBytes: 10, SHA256: "hash", StoragePath: "objects/hash", Status: "processing",
	})
	if err != nil || duplicate {
		t.Fatalf("add document: duplicate=%v err=%v", duplicate, err)
	}
	if err := store.ReplaceKnowledgeChunks(document.ID, []KnowledgeChunk{{DocumentID: document.ID, Ordinal: 0, PageStart: 2, PageEnd: 2, Content: "스파크톡 지식 검색 시험"}}, 2, "ready", ""); err != nil {
		t.Fatal(err)
	}
	results, err := store.SearchKnowledge("지식 검색", collection.ID, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].Title != "스파크톡 안내서" || results[0].PageStart != 2 {
		t.Fatalf("unexpected search results: %+v", results)
	}
	duplicateDocument, duplicate, err := store.AddKnowledgeDocument(KnowledgeDocument{
		ID: "ffffffffffffffffffffffffffffffff", CollectionID: collection.ID,
		Title: "duplicate", SourceName: "copy.txt", SourceKind: "file",
		MIMEType: "text/plain", SHA256: "hash", StoragePath: "objects/hash", Status: "processing",
	})
	if err != nil || !duplicate || duplicateDocument.ID != document.ID {
		t.Fatalf("duplicate was not resolved: doc=%+v duplicate=%v err=%v", duplicateDocument, duplicate, err)
	}
	paths, err := store.DeleteKnowledgeCollection(collection.ID)
	if err != nil || len(paths) != 1 {
		t.Fatalf("delete collection: paths=%v err=%v", paths, err)
	}
	results, err = store.SearchKnowledge("지식 검색", 0, 10)
	if err != nil || len(results) != 0 {
		t.Fatalf("cascade left searchable chunks: results=%+v err=%v", results, err)
	}
}

func TestKnowledgeJobProgressRecoveryAndPageIndex(t *testing.T) {
	store, err := Open(filepath.Join(t.TempDir(), "knowledge-jobs.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	document, duplicate, err := store.AddKnowledgeDocument(KnowledgeDocument{
		ID: "11111111111111111111111111111111", CollectionID: 1, Title: "전자책",
		SourceName: "viewer.html", SourceKind: "url", MIMEType: "text/html",
		SHA256: "viewer-hash", StoragePath: "objects/viewer", Status: "paused",
	})
	if err != nil || duplicate {
		t.Fatalf("add document: duplicate=%v err=%v", duplicate, err)
	}
	job, err := store.AddKnowledgeJob(KnowledgeJob{
		ID: "22222222222222222222222222222222", DocumentID: document.ID, CollectionID: 1,
		SourceURL: "https://example.com/viewer", Adapter: "test-viewer", Title: "전자책", Status: "paused",
	}, []KnowledgeJobItem{
		{Ordinal: 1, SourceURL: "https://example.com/1.pdf"},
		{Ordinal: 2, SourceURL: "https://example.com/2.pdf"},
	})
	if err != nil || job.TotalItems != 2 {
		t.Fatalf("add job: job=%+v err=%v", job, err)
	}
	if err := store.ResetKnowledgeJobFailures(job.ID); err != nil {
		t.Fatal(err)
	}
	if err := store.SetKnowledgeJobItemStatus(job.ID, 1, "completed", ""); err != nil {
		t.Fatal(err)
	}
	if _, _, err := store.UpsertKnowledgeAsset(KnowledgeAsset{
		DocumentID: document.ID, Kind: "page", Ordinal: 1, SourceURL: "https://example.com/1.pdf",
		MIMEType: "application/pdf", SHA256: "page-1", StoragePath: "objects/page-1", Status: "ready",
	}); err != nil {
		t.Fatal(err)
	}
	if err := store.ReplaceKnowledgePageChunks(document.ID, 1, []KnowledgeChunk{
		{Heading: "첫 쪽", Content: "전자책 첫 페이지 검색 본문"},
	}, 2); err != nil {
		t.Fatal(err)
	}
	if err := store.ReplaceKnowledgePageChunks(document.ID, 2, []KnowledgeChunk{
		{Heading: "둘째 쪽", Content: "전자책 둘째 페이지 검색 본문"},
	}, 2); err != nil {
		t.Fatal(err)
	}
	if err := store.UpdateKnowledgeDocumentStatus(document.ID, "ready", ""); err != nil {
		t.Fatal(err)
	}
	_, around, err := store.KnowledgeChunksAround(document.ID, 1000, 1)
	if err != nil || len(around) != 2 || around[0].PageStart != 1 || around[1].PageStart != 2 {
		t.Fatalf("sparse page ordinals lost neighbors: chunks=%+v err=%v", around, err)
	}
	results, err := store.SearchKnowledge("첫 페이지", 1, 5)
	foundPageOne := false
	for _, item := range results {
		foundPageOne = foundPageOne || item.PageStart == 1
	}
	if err != nil || !foundPageOne {
		t.Fatalf("page chunks were not indexed: results=%+v err=%v", results, err)
	}
	if err := store.SetKnowledgeJobItemStatus(job.ID, 2, "running", ""); err != nil {
		t.Fatal(err)
	}
	if err := store.SetKnowledgeJobStatus(job.ID, "running", "", 2); err != nil {
		t.Fatal(err)
	}
	recovered, err := store.RecoverKnowledgeJobs()
	if err != nil || len(recovered) != 1 || recovered[0].Status != "queued" {
		t.Fatalf("recover jobs: jobs=%+v err=%v", recovered, err)
	}
	pending, err := store.KnowledgeJobItems(job.ID, false)
	if err != nil || len(pending) != 1 || pending[0].Ordinal != 2 {
		t.Fatalf("running item was not reset: items=%+v err=%v", pending, err)
	}
	paths, err := store.KnowledgeDocumentStoragePaths(document.ID)
	if err != nil || len(paths) != 2 {
		t.Fatalf("document assets are not tracked: paths=%v err=%v", paths, err)
	}
}

func TestKnowledgeOCRProgressAndRecovery(t *testing.T) {
	store, err := Open(filepath.Join(t.TempDir(), "knowledge-ocr.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	document, duplicate, err := store.AddKnowledgeDocument(KnowledgeDocument{
		ID: "33333333333333333333333333333333", CollectionID: 1, Title: "스캔 문서",
		SourceName: "scan.pdf", SourceKind: "file", MIMEType: "application/pdf",
		SHA256: "scan-hash", StoragePath: "objects/scan", Status: "needs_ocr", PageCount: 3,
	})
	if err != nil || duplicate {
		t.Fatalf("add document: duplicate=%v err=%v", duplicate, err)
	}
	if err := store.BeginKnowledgeOCR(document.ID, 3, false); err != nil {
		t.Fatal(err)
	}
	if err := store.ReplaceKnowledgeOCRPageChunks(document.ID, 1, []KnowledgeChunk{{Content: "첫째 쪽"}}, 3); err != nil {
		t.Fatal(err)
	}
	progress, err := store.KnowledgeDocument(document.ID)
	if err != nil || progress.Status != "processing" || progress.OCRProcessedPages != 1 || progress.OCRTotalPages != 3 {
		t.Fatalf("unexpected OCR progress: document=%+v err=%v", progress, err)
	}
	if err := store.RecoverKnowledgeOCR(); err != nil {
		t.Fatal(err)
	}
	recovered, err := store.KnowledgeDocument(document.ID)
	if err != nil || recovered.Status != "needs_ocr" || recovered.OCRProcessedPages != 1 {
		t.Fatalf("OCR recovery lost progress: document=%+v err=%v", recovered, err)
	}
}
