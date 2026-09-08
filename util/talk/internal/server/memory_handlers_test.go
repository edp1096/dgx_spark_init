package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"sparktalk/internal/db"
)

func TestMemoryHandlersCRUD(t *testing.T) {
	store, err := db.Open(t.TempDir() + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	server := &Server{db: store}

	create := httptest.NewRecorder()
	server.memories(create, httptest.NewRequest(http.MethodPost, "/api/memories", strings.NewReader(`{"kind":"user","title":"말투","content":"항상 존댓말을 사용한다"}`)))
	if create.Code != http.StatusCreated {
		t.Fatalf("create status = %d, body = %s", create.Code, create.Body.String())
	}
	var item db.Memory
	if err := json.Unmarshal(create.Body.Bytes(), &item); err != nil {
		t.Fatal(err)
	}
	if item.Priority != "preferred" {
		t.Fatalf("default priority = %q", item.Priority)
	}

	list := httptest.NewRecorder()
	server.memories(list, httptest.NewRequest(http.MethodGet, "/api/memories", nil))
	if list.Code != http.StatusOK || !strings.Contains(list.Body.String(), "항상 존댓말") {
		t.Fatalf("list status = %d, body = %s", list.Code, list.Body.String())
	}

	update := httptest.NewRecorder()
	server.memory(update, httptest.NewRequest(http.MethodPut, "/api/memories/"+strconv.FormatInt(item.ID, 10), strings.NewReader(`{"kind":"user","priority":"reference","title":"말투","content":"간결한 존댓말을 사용한다","enabled":false}`)))
	if update.Code != http.StatusOK || !strings.Contains(update.Body.String(), `"enabled":false`) || !strings.Contains(update.Body.String(), `"priority":"reference"`) {
		t.Fatalf("update status = %d, body = %s", update.Code, update.Body.String())
	}

	remove := httptest.NewRecorder()
	server.memory(remove, httptest.NewRequest(http.MethodDelete, "/api/memories/"+strconv.FormatInt(item.ID, 10), nil))
	if remove.Code != http.StatusNoContent {
		t.Fatalf("delete status = %d, body = %s", remove.Code, remove.Body.String())
	}
}
