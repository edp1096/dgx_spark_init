package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestGenerateTitleUsesNamedToolCall(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Tools  []Tool `json:"tools"`
			Choice struct {
				Function struct {
					Name string `json:"name"`
				} `json:"function"`
			} `json:"tool_choice"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		if len(body.Tools) != 1 || body.Tools[0].Function.Name != "session_title" || body.Choice.Function.Name != "session_title" {
			t.Errorf("missing forced title call: %+v", body)
		}
		fmt.Fprintln(w, `{"choices":[{"message":{"content":"","tool_calls":[{"type":"function","function":{"name":"session_title","arguments":"{\"title\":\"음성 전사 구성\"}"}}]}}]}`)
	}))
	defer backend.Close()
	title, err := New(backend.URL, "model", "").GenerateTitle(context.Background(), "model", "영상에서 말을 글로 바꿔줘")
	if err != nil || title != "음성 전사 구성" {
		t.Fatalf("title=%q err=%v", title, err)
	}
}
