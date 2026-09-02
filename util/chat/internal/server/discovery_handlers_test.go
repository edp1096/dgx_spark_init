package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
)

func TestDiscoveryHandlers(t *testing.T) {
	store, err := db.Open(t.TempDir() + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	if _, err := store.CreateSession("old", "메모리 실험", "model", "low"); err != nil {
		t.Fatal(err)
	}
	if _, err := store.AddMessage("old", "assistant", "메모리 검색 결과", "", nil, nil); err != nil {
		t.Fatal(err)
	}
	if err := store.AddToolAudit("old", "web_search", "", "execute", "executed", ""); err != nil {
		t.Fatal(err)
	}
	server := &Server{db: store, cfg: config.Config{Tools: config.ToolsConfig{Enabled: true, SkillsEnabled: true}}}

	search := httptest.NewRecorder()
	server.searchConversations(search, httptest.NewRequest(http.MethodGet, "/api/search?q=메모리", nil))
	if search.Code != http.StatusOK || !strings.Contains(search.Body.String(), `"session_id":"old"`) {
		t.Fatalf("search status=%d body=%s", search.Code, search.Body.String())
	}
	skills := httptest.NewRecorder()
	server.skillCatalog(skills, httptest.NewRequest(http.MethodGet, "/api/skills", nil))
	if skills.Code != http.StatusOK || !strings.Contains(skills.Body.String(), "web-research") {
		t.Fatalf("skills status=%d body=%s", skills.Code, skills.Body.String())
	}
	audits := httptest.NewRecorder()
	server.toolAudits(audits, httptest.NewRequest(http.MethodGet, "/api/tool-audit", nil))
	if audits.Code != http.StatusOK || !strings.Contains(audits.Body.String(), "web_search") {
		t.Fatalf("audits status=%d body=%s", audits.Code, audits.Body.String())
	}
	page := httptest.NewRecorder()
	server.searchConversationPage(page, httptest.NewRequest(http.MethodGet, "/api/search/page?q=메모리&limit=1&sort=relevance&scope=all", nil))
	if page.Code != http.StatusOK {
		t.Fatalf("page status=%d body=%s", page.Code, page.Body.String())
	}
	var result struct {
		Items []db.RecallItem `json:"items"`
	}
	if err := json.Unmarshal(page.Body.Bytes(), &result); err != nil || len(result.Items) != 1 {
		t.Fatalf("page result=%+v err=%v", result, err)
	}
}
