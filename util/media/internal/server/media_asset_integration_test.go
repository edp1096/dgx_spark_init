package server

import (
	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestMediaAssetProxyPreservesRangeAndJobDeleteRemovesAsset(t *testing.T) {
	const assetID = "0123456789abcdef0123456789abcdef"
	deleted := false
	artifactsDeleted := false
	mediaWorker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete && r.URL.Path == "/v1/media/jobs/media-job" {
			artifactsDeleted = true
			w.WriteHeader(http.StatusNoContent)
			return
		}
		if r.URL.Path != "/v1/media/assets/"+assetID {
			http.NotFound(w, r)
			return
		}
		switch r.Method {
		case http.MethodGet:
			if got := r.Header.Get("Range"); got != "bytes=2-4" {
				t.Fatalf("Range = %q", got)
			}
			w.Header().Set("Accept-Ranges", "bytes")
			w.Header().Set("Content-Range", "bytes 2-4/6")
			w.Header().Set("Content-Type", "video/mp4")
			w.WriteHeader(http.StatusPartialContent)
			_, _ = w.Write([]byte("cde"))
		case http.MethodDelete:
			deleted = true
			w.WriteHeader(http.StatusNoContent)
		default:
			w.WriteHeader(http.StatusMethodNotAllowed)
		}
	}))
	defer mediaWorker.Close()

	cfg := config.Config{DataDir: t.TempDir(), Engines: map[string]config.Engine{"media": {Endpoint: mediaWorker.URL}}}
	store, err := jobs.New(cfg.DataDir)
	if err != nil {
		t.Fatal(err)
	}
	job := jobs.Job{ID: "media-job", Kind: "recognition", Status: "completed", MediaAssetID: assetID, CreatedAt: time.Now()}
	if err := store.Save(job); err != nil {
		t.Fatal(err)
	}
	handler := New(cfg, store, nil).Handler()

	streamReq := httptest.NewRequest(http.MethodGet, "/api/media/assets/"+assetID, nil)
	streamReq.Header.Set("Range", "bytes=2-4")
	streamRes := httptest.NewRecorder()
	handler.ServeHTTP(streamRes, streamReq)
	if streamRes.Code != http.StatusPartialContent || streamRes.Body.String() != "cde" {
		t.Fatalf("stream status=%d body=%q", streamRes.Code, streamRes.Body.String())
	}
	if got := streamRes.Header().Get("Content-Range"); got != "bytes 2-4/6" {
		t.Fatalf("Content-Range = %q", got)
	}

	deleteReq := httptest.NewRequest(http.MethodDelete, "/api/jobs/"+job.ID, nil)
	deleteRes := httptest.NewRecorder()
	handler.ServeHTTP(deleteRes, deleteReq)
	if deleteRes.Code != http.StatusNoContent || !deleted || !artifactsDeleted {
		t.Fatalf("delete status=%d asset deleted=%v artifacts deleted=%v", deleteRes.Code, deleted, artifactsDeleted)
	}
	if _, ok := store.Get(job.ID); ok {
		t.Fatal("job remains after delete")
	}
}
