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

func TestCompletionLoopHidesToolProtocolAfterRoundLimit(t *testing.T) {
	var requests atomic.Int32
	var finalInstructionSeen atomic.Bool
	modelServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Messages []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
			Tools []any `json:"tools"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		if requests.Add(1) == 1 {
			fmt.Fprintln(w, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_limit","type":"function","function":{"name":"not_allowed","arguments":"{}"}}]}}]}`)
		} else {
			if len(request.Tools) == 0 && len(request.Messages) > 0 && strings.Contains(request.Messages[len(request.Messages)-1].Content, "tool execution limit") {
				finalInstructionSeen.Store(true)
			}
			fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":"<tool_call><function=ssh_exec>leaked</function></tool_call>"}}]}`)
		}
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer modelServer.Close()

	result, err := runCompletionLoop(
		context.Background(), llm.New(modelServer.URL, "test-model", ""),
		[]llm.Message{{Role: "user", Content: "test"}}, "test-model", "low", "",
		config.ToolsConfig{Enabled: true, MaxRounds: 1, SearchResults: 1, Timeout: "1s"}, true,
		func(string, any) error { return nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	if !finalInstructionSeen.Load() || strings.Contains(result.Content, "tool_call") || !strings.Contains(result.Content, "실행 한도") {
		t.Fatalf("instruction=%v result=%+v", finalInstructionSeen.Load(), result)
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

func TestRetainLatestVideoInputRemovesOlderRawVideos(t *testing.T) {
	messages := []llm.Message{
		{Role: "user", Content: []map[string]any{
			{"type": "video_url", "video_url": map[string]string{"url": "data:video/mp4;base64,old"}},
			{"type": "text", "text": "이전 영상 질문"},
		}},
		{Role: "assistant", Content: "이전 답변"},
		{Role: "user", Content: []map[string]any{
			{"type": "video_url", "video_url": map[string]string{"url": "data:video/mp4;base64,new"}},
			{"type": "text", "text": "현재 영상 분석해라"},
		}},
	}
	filtered := retainLatestVideoInput(messages)
	payload, _ := json.Marshal(filtered)
	text := string(payload)
	if strings.Count(text, `"type":"video_url"`) != 1 || strings.Contains(text, "base64,old") || !strings.Contains(text, "base64,new") {
		t.Fatalf("unexpected video filtering: %s", text)
	}
	for _, instruction := range []string{"이전 영상 질문", "현재 영상 분석해라"} {
		if !strings.Contains(text, instruction) {
			t.Fatalf("text instruction %q was removed: %s", instruction, text)
		}
	}
}

func TestDeepSeekPreservesToolReasoningThroughFinalRequest(t *testing.T) {
	for _, maxRounds := range []int{1, 3} {
		t.Run(fmt.Sprintf("max_rounds_%d", maxRounds), func(t *testing.T) {
			var requests atomic.Int32
			modelServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var request struct {
					Messages []llm.Message  `json:"messages"`
					Options  map[string]any `json:"chat_template_kwargs"`
					Tools    []any          `json:"tools"`
				}
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Error(err)
				}
				w.Header().Set("Content-Type", "text/event-stream")
				if requests.Add(1) == 1 {
					fmt.Fprintln(w, `data: {"choices":[{"delta":{"reasoning":"Need a lookup first."}}]}`)
					fmt.Fprintln(w, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_reasoning","type":"function","function":{"name":"not_allowed","arguments":"{}"}}]}}]}`)
				} else {
					found := false
					for _, m := range request.Messages {
						if m.Role == "assistant" && len(m.ToolCalls) > 0 {
							found = true
							if m.ReasoningContent != "Need a lookup first." {
								t.Errorf("lost tool reasoning: %q", m.ReasoningContent)
							}
						}
					}
					if !found || request.Options["drop_thinking"] != false {
						t.Errorf("encoder must retain tool reasoning: %+v", request.Options)
					}
					if request.Options["reasoning_effort"] != "max" {
						t.Error("changed user thinking preference")
					}
					if maxRounds == 1 && (len(request.Tools) != 0 || request.Messages[len(request.Messages)-1].Content != toolLimitFinalInstruction) {
						t.Error("round-limit path not exercised")
					}
					fmt.Fprintln(w, `data: {"choices":[{"delta":{"reasoning":"The lookup failed."}}]}`)
					fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":"Lookup unavailable."}}]}`)
				}
				fmt.Fprintln(w, "data: [DONE]")
			}))
			defer modelServer.Close()
			result, err := runCompletionLoop(context.Background(), llm.New(modelServer.URL, "test-model", "", "deepseek-v4"), []llm.Message{{Role: "user", Content: "test"}}, "test-model", "max", "", config.ToolsConfig{Enabled: true, MaxRounds: maxRounds, SearchResults: 1, Timeout: "1s"}, true, func(string, any) error { return nil })
			if err != nil {
				t.Fatal(err)
			}
			if requests.Load() != 2 || result.Content != "Lookup unavailable." || result.Reasoning != "Need a lookup first.\n\nThe lookup failed." {
				t.Fatalf("unexpected result: %+v", result)
			}
		})
	}
}

func TestCompletionLoopRetriesReasoningOnlyFinalWithoutThinking(t *testing.T) {
	var requests atomic.Int32
	modelServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Messages []llm.Message  `json:"messages"`
			Options  map[string]any `json:"chat_template_kwargs"`
			Tools    []any          `json:"tools"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		if requests.Add(1) == 1 {
			fmt.Fprintln(w, `data: {"choices":[{"delta":{"reasoning":"Long reasoning that reached EOS."}}]}`)
		} else {
			if request.Options["thinking"] != false || request.Options["reasoning_effort"] != nil {
				t.Errorf("retry still enabled thinking: %+v", request.Options)
			}
			if len(request.Tools) != 0 || request.Messages[len(request.Messages)-1].Content != emptyFinalRetryInstruction {
				t.Errorf("unexpected retry request: %+v", request)
			}
			fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":"Recovered final answer."}}]}`)
		}
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer modelServer.Close()

	result, err := runCompletionLoop(
		context.Background(), llm.New(modelServer.URL, "test-model", "", "deepseek-v4"),
		[]llm.Message{{Role: "user", Content: "test"}}, "test-model", "max", "",
		config.ToolsConfig{}, false, func(string, any) error { return nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	if requests.Load() != 2 || result.Content != "Recovered final answer." || result.Reasoning != "Long reasoning that reached EOS." {
		t.Fatalf("unexpected recovery result: requests=%d result=%+v", requests.Load(), result)
	}
}

func TestCompletionLoopRetriesReasoningOnlyRoundLimitFinal(t *testing.T) {
	var requests atomic.Int32
	modelServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Messages []llm.Message  `json:"messages"`
			Options  map[string]any `json:"chat_template_kwargs"`
			Tools    []any          `json:"tools"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Error(err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		switch requests.Add(1) {
		case 1:
			fmt.Fprintln(w, `data: {"choices":[{"delta":{"reasoning":"Need a tool."}}]}`)
			fmt.Fprintln(w, `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_retry","type":"function","function":{"name":"not_allowed","arguments":"{}"}}]}}]}`)
		case 2:
			if len(request.Tools) != 0 || request.Messages[len(request.Messages)-1].Content != toolLimitFinalInstruction {
				t.Errorf("round-limit final request is malformed: %+v", request)
			}
			fmt.Fprintln(w, `data: {"choices":[{"delta":{"reasoning":"I should now answer."}}]}`)
		case 3:
			if len(request.Tools) != 0 || request.Options["thinking"] != false || request.Messages[len(request.Messages)-1].Content != emptyFinalRetryInstruction {
				t.Errorf("empty-final retry is malformed: %+v", request)
			}
			fmt.Fprintln(w, `data: {"choices":[{"delta":{"content":"Final after retry."}}]}`)
		default:
			t.Fatalf("unexpected request %d", requests.Load())
		}
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer modelServer.Close()

	result, err := runCompletionLoop(
		context.Background(), llm.New(modelServer.URL, "test-model", "", "deepseek-v4"),
		[]llm.Message{{Role: "user", Content: "test"}}, "test-model", "max", "",
		config.ToolsConfig{Enabled: true, MaxRounds: 1, SearchResults: 1, Timeout: "1s"}, true,
		func(string, any) error { return nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	wantReasoning := "Need a tool.\n\nI should now answer."
	if requests.Load() != 3 || result.Content != "Final after retry." || result.Reasoning != wantReasoning {
		t.Fatalf("unexpected recovery result: requests=%d result=%+v", requests.Load(), result)
	}
}

func TestCompletionLoopRejectsRepeatedReasoningOnlyFinal(t *testing.T) {
	modelServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintln(w, `data: {"choices":[{"delta":{"reasoning":"Still no answer."}}]}`)
		fmt.Fprintln(w, "data: [DONE]")
	}))
	defer modelServer.Close()

	result, err := runCompletionLoop(
		context.Background(), llm.New(modelServer.URL, "test-model", "", "deepseek-v4"),
		[]llm.Message{{Role: "user", Content: "test"}}, "test-model", "max", "",
		config.ToolsConfig{}, false, func(string, any) error { return nil },
	)
	if err == nil || !strings.Contains(err.Error(), "no final answer") || result.Content != "" || result.Reasoning != "Still no answer.\n\nStill no answer." {
		t.Fatalf("reasoning-only retry should fail visibly: result=%+v err=%v", result, err)
	}
}
