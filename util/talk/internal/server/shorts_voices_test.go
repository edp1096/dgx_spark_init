package server

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
)

func TestVoiceListShowsActualCatalogAndReusesPreviewAttachments(t *testing.T) {
	downloads := 0
	s, reg := shortsToolTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/health":
			io.WriteString(w, `{"status":"ok"}`)
		case strings.HasSuffix(r.URL.Path, "/preview.wav"):
			downloads++
			w.Header().Set("Content-Type", "audio/wav")
			io.WriteString(w, "RIFF\x00\x00\x00\x00WAVEfmt ")
		case r.URL.Path == "/v1/voices":
			var a map[string]string
			json.NewDecoder(r.Body).Decode(&a)
			if a["session_id"] != "session" || a["action"] != "list" {
				t.Errorf("wrong arguments: %v", a)
			}
			io.WriteString(w, `{"voices":[{"id":"ryan","name":"Ryan","description":"남성","native_language":"영어"},{"id":"sohee","name":"Sohee","description":"여성","native_language":"한국어"}],"selected_voice":"ryan"}`)
		default:
			t.Errorf("unexpected generation request: %s", r.URL.Path)
			http.NotFound(w, r)
		}
	})
	handler := reg.handlers["shorts_voices"]
	call := llm.ToolCall{Function: llm.FunctionCall{Arguments: `{"action":"list"}`}}
	result, err := handler(context.Background(), call, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if downloads != 2 || len(result.Attachments) != 2 || !strings.Contains(result.FinalContent, "2개") || !strings.Contains(result.FinalContent, "Ryan") {
		t.Fatalf("bad list: %+v downloads=%d", result, downloads)
	}
	if _, err = s.db.AddMessage("session", "assistant", result.FinalContent, "", nil, result.Attachments); err != nil {
		t.Fatal(err)
	}
	if _, err = handler(context.Background(), call, nil, nil); err != nil {
		t.Fatal(err)
	}
	if downloads != 2 {
		t.Fatal("preview was needlessly copied again", downloads)
	}
	// The next voice question must not send assistant audio to the ASR/model.
	messages, err := s.llmMessages(context.Background(), []db.Message{{Role: "assistant", Content: result.FinalContent, Attachments: result.Attachments}, {Role: "user", Content: "Ryan을 선택해"}}, config.Config{})
	if err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal(messages)
	if strings.Contains(string(raw), "data:audio") || !strings.Contains(string(raw), "media_reference") {
		t.Fatal("historical previews not reduced to references", string(raw))
	}
}
