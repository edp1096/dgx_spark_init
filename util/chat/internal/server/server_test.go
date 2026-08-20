package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"testing/fstest"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
)

func TestFallbackTitle(t *testing.T) {
	if got := fallbackTitle("  간단한   테스트 제목  "); got != "간단한 테스트 제목" {
		t.Fatalf("unexpected title: %q", got)
	}
	long := fallbackTitle("이 문장은 자동 제목의 최대 길이를 확실하게 넘어가도록 충분히 길게 작성한 테스트 문장입니다")
	if []rune(long)[len([]rune(long))-1] != '…' {
		t.Fatalf("long title was not shortened: %q", long)
	}
}

func TestSessionContextRoute(t *testing.T) {
	modelServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"data": []map[string]any{{"max_model_len": 32768}}})
	}))
	defer modelServer.Close()
	store, err := db.Open(t.TempDir() + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.CreateSession("route-test", "route test", "test-model", "low"); err != nil {
		t.Fatal(err)
	}
	cfg, _, err := config.Load(t.TempDir() + "/defaults.yaml")
	if err != nil {
		t.Fatal(err)
	}
	cfg.Server.Database = t.TempDir() + "/media.db"
	cfg.Model.Endpoint = modelServer.URL
	embedded := fstest.MapFS{"web/dist/index.html": {Data: []byte("ok")}}
	srv, err := New(cfg, t.TempDir()+"/sparktalk.yaml", store, llm.New(modelServer.URL, "test-model", ""), embedded)
	if err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodGet, "/api/sessions/route-test/context", nil)
	rec := httptest.NewRecorder()
	srv.server.Handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("context route status = %d, body = %s", rec.Code, rec.Body.String())
	}
	var state contextState
	if err := json.Unmarshal(rec.Body.Bytes(), &state); err != nil {
		t.Fatal(err)
	}
	if state.WindowTokens != 32768 {
		t.Fatalf("window tokens = %d", state.WindowTokens)
	}
}
