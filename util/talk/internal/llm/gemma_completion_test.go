package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
)

func TestGemma31CompletionPolicyPreservesPersonaAndExplicitBrevity(t *testing.T) {
	original := []Message{{Role: "system", Content: "사용자를 오빠로 부르고 생각은 짧게 한다."}, {Role: "user", Content: "궁궐을 한 문장으로 설명해 줘."}}
	before := append([]Message(nil), original...)
	for _, test := range []struct {
		family, model string
		want          bool
	}{
		{"gemma4", "Huihui-gemma-4-31B-it-abliterated-v2-NVFP4", true},
		{"gemma4-vllm", "google/gemma-4-31B-it", true},
		{"gemma4", "google/gemma-4-26B-A4B-it", false},
		{"qwen38fn_exl3", "qwen38fn_exl3_q4", false},
	} {
		t.Run(test.family+test.model, func(t *testing.T) {
			var received []Message
			backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var payload struct {
					Messages []Message      `json:"messages"`
					Params   map[string]any `json:"custom_params"`
				}
				if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
					t.Error(err)
				}
				received = payload.Messages
				if _, forced := payload.Params["thinking_budget"]; forced {
					t.Error("unrestricted thinking unexpectedly forced a boundary")
				}
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprint(w, "data: [DONE]\n\n")
			}))
			defer backend.Close()
			client := New(backend.URL, test.model, "", test.family)
			prepared := client.InputMessages(original)
			_, err := client.Stream(context.Background(), prepared, test.model, "on", nil, func(string, string) error { return nil })
			if err != nil {
				t.Fatal(err)
			}
			if len(received) != 2 || received[1].Content != original[1].Content {
				t.Fatal("user's explicit length instruction changed")
			}
			prompt := received[0].Content.(string)
			if !strings.HasSuffix(prompt, original[0].Content.(string)) {
				t.Fatal("persona was replaced")
			}
			want := 0
			if test.want {
				want = 1
			}
			if strings.Count(prompt, gemma31CompletionPolicy) != want {
				t.Fatalf("completion policy count: %q", prompt)
			}
			if !reflect.DeepEqual(original, before) {
				t.Fatal("caller-owned conversation mutated")
			}
		})
	}
}
