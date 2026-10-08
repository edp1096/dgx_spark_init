package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/orchestrator"
)

func TestVideoToolPersistsBeforeEventAndRejectsUnknownFields(t *testing.T) {
	s, _ := testImageServer(t)
	var calls atomic.Int32
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"events":[],"next":0}`))
			return
		}
		calls.Add(1)
		w.Header().Set("Content-Type", "video/mp4")
		w.Write(importTestVideo)
	}))
	defer backend.Close()
	catalog, err := orchestrator.LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	for i := range catalog.Components {
		if catalog.Components[i].ID == "qwim-mmh3" {
			catalog.Components[i].Endpoint = backend.URL
			catalog.Components[i].HealthURL = backend.URL + "/health"
		}
	}
	catalog, err = orchestrator.ValidateCatalog(catalog)
	if err != nil {
		t.Fatal(err)
	}
	s.cfg = config.Config{Runtime: config.RuntimeConfig{Mode: "external", Bundle: "qwen38fn_exl3", Catalog: &catalog}}
	reg := newCompletionToolRegistry(s, "session", config.ToolsConfig{}, false, s.persistMediaAttachments(1, 0, nil, nil))
	history, _ := s.db.Messages("session")
	sink := s.persistMediaAttachments(history[0].ID, 0, nil, nil)
	reg = newCompletionToolRegistry(s, "session", config.ToolsConfig{}, false, sink)
	handler, ok := reg.handlers["video_generate"]
	if !ok {
		t.Fatal("video tool missing")
	}
	events := 0
	result, err := handler(context.Background(), llm.ToolCall{Function: llm.FunctionCall{Name: "video_generate", Arguments: `{"prompt":"A red car rolls","seed":1}`}}, nil, func(event string, payload any) error {
		if event != "media_attached" {
			return nil
		}
		events++
		messages, _ := s.db.Messages("session")
		found := false
		for _, file := range messages[0].Attachments {
			if file.MIME == "video/mp4" {
				found = true
			}
		}
		if !found {
			t.Fatal("video event emitted before durable persistence")
		}
		return nil
	})
	if err != nil || events != 1 || !result.AttachmentEmitted {
		t.Fatalf("%+v %v events=%d", result, err, events)
	}
	var info struct {
		Status     string        `json:"status"`
		Attachment db.Attachment `json:"attachment"`
	}
	json.Unmarshal([]byte(result.Result), &info)
	if info.Status != "saved" || info.Attachment.Size != int64(len(importTestVideo)) {
		t.Fatalf("bad save: %s", result.Result)
	}
	_, err = handler(context.Background(), llm.ToolCall{Function: llm.FunctionCall{Arguments: `{"prompt":"car","duration":60}`}}, nil, nil)
	if err == nil || calls.Load() != 1 {
		t.Fatal("unknown video argument invoked backend")
	}
	if !strings.Contains(result.Result, "video/mp4") {
		t.Fatal("no original MP4 attachment")
	}
}
