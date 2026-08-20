package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"sparktalk/internal/config"
	"sparktalk/internal/llm"
)

func TestCompletionLoopExecutesToolAndContinues(t *testing.T) {
	var requests atomic.Int32
	var sawSystemPrompt atomic.Bool
	modelServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err == nil && len(request.Messages) > 0 && request.Messages[0].Role == "system" &&
			strings.Contains(request.Messages[0].Content, "Always use polite Korean") && strings.Contains(request.Messages[0].Content, "web_search") {
			sawSystemPrompt.Store(true)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		if requests.Add(1) == 1 {
			fmt.Fprintln(w, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_test","type":"function","function":{"name":"not_allowed","arguments":"{}"}}]}}]}`)
		} else {
			fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":"도구 오류를 확인했습니다."}}]}`)
		}
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer modelServer.Close()

	var events []string
	result, err := runCompletionLoop(
		context.Background(), llm.New(modelServer.URL, "test-model", ""),
		[]llm.Message{{Role: "user", Content: "test"}}, "test-model", "low",
		"Always use polite Korean.",
		config.ToolsConfig{Enabled: true, MaxRounds: 3, SearchResults: 5, Timeout: "1s"}, true,
		func(event string, _ any) error {
			events = append(events, event)
			return nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if requests.Load() != 2 || result.Content != "도구 오류를 확인했습니다." {
		t.Fatalf("unexpected loop result: requests=%d result=%+v", requests.Load(), result)
	}
	if !sawSystemPrompt.Load() {
		t.Fatal("custom and web tool system prompts were not merged")
	}
	if len(result.ToolTrace) != 1 || !strings.Contains(result.ToolTrace[0].Error, "unknown tool") {
		t.Fatalf("tool failure was not recorded: %+v", result.ToolTrace)
	}
	if strings.Join(events, ",") != "tool_start,tool_result,delta" {
		t.Fatalf("unexpected events: %v", events)
	}
}

func TestCompletionLoopUsesCustomSystemPromptWithoutTools(t *testing.T) {
	var firstRole, firstContent string
	modelServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err == nil && len(request.Messages) > 0 {
			firstRole = request.Messages[0].Role
			firstContent = request.Messages[0].Content
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":"네, 알겠습니다."}}]}`)
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer modelServer.Close()

	result, err := runCompletionLoop(
		context.Background(), llm.New(modelServer.URL, "test-model", ""),
		[]llm.Message{{Role: "user", Content: "test"}}, "test-model", "medium",
		"모든 답변은 존댓말로 작성한다.", config.ToolsConfig{Enabled: true}, false,
		func(string, any) error { return nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	if result.Content != "네, 알겠습니다." || firstRole != "system" || firstContent != "모든 답변은 존댓말로 작성한다." {
		t.Fatalf("custom system prompt was not sent: role=%q content=%q result=%+v", firstRole, firstContent, result)
	}
}

func TestCompletionLoopMergesContextCheckpointIntoFirstSystemMessage(t *testing.T) {
	var roles []string
	var systemContent string
	modelServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		for _, message := range request.Messages {
			roles = append(roles, message.Role)
			if message.Role == "system" {
				systemContent = message.Content
			}
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":"ok"}}]}`)
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer modelServer.Close()

	_, err := runCompletionLoop(
		context.Background(), llm.New(modelServer.URL, "test-model", ""),
		[]llm.Message{
			{Role: "system", Content: "Conversation checkpoint: old facts"},
			{Role: "user", Content: "new question"},
		},
		"test-model", "none", "global instruction", config.ToolsConfig{}, false,
		func(string, any) error { return nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(roles, ",") != "system,user" || !strings.Contains(systemContent, "global instruction") || !strings.Contains(systemContent, "Conversation checkpoint") {
		t.Fatalf("system messages were not merged: roles=%v content=%q", roles, systemContent)
	}
}
