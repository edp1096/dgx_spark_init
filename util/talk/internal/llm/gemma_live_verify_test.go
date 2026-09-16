package llm

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"
)

func TestGemmaLiveToolContinuation(t *testing.T) {
	if os.Getenv("GEMMA_LIVE_VERIFY") != "1" {
		t.Skip("explicit live check only")
	}
	c := New("http://127.0.0.1:8000", "", "", "gemma4")
	c.thinkingBudget = 512
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()
	ctx = WithOutputLimit(ctx, 1024)
	model, err := c.Model(ctx)
	if err != nil {
		t.Fatal(err)
	}
	tools := []Tool{
		{Type: "function", Function: ToolFunction{Name: "web_search", Description: "Search weather information", Parameters: json.RawMessage(`{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}`)}},
		{Type: "function", Function: ToolFunction{Name: "web_fetch", Description: "Read the forecast page", Parameters: json.RawMessage(`{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}`)}},
	}
	var records []map[string]any
	defer func() {
		b, _ := json.MarshalIndent(records, "", "  ")
		_ = os.WriteFile("/tmp/gemma-live-preserve-verification.json", b, 0600)
	}()
	for trial := 0; trial < 3; trial++ {
		messages := []Message{{Role: "system", Content: "도구 연동 시험이다. 검색 요약에 온도가 없으면 상세 페이지를 열어라. 도구 결과 없이 수치를 만들거나 확인했다고 주장하지 마라."}, {Role: "user", Content: "내일 잠실 날씨 최저 최고 기온을 확인해라."}}
		names := []string{}
		final := ""
		loop := false
		var trace []StreamResult
		for round := 0; round < 4; round++ {
			r, e := c.Stream(ctx, messages, model, "on", tools, func(string, string) error { return nil })
			trace = append(trace, r)
			combined := strings.ToLower(r.Content + r.Reasoning)
			compact := strings.NewReplacer("_", "", " ", "", "\n", "").Replace(combined)
			loop = loop || strings.Contains(compact, "getgetgetgetget")
			t.Logf("trial=%d round=%d calls=%d finish=%s get_loop=%v err=%v", trial+1, round+1, len(r.ToolCalls), r.FinishReason, loop, e)
			if e != nil || len(r.ToolCalls) == 0 {
				final = r.Content
				break
			}
			messages = append(messages, Message{Role: "assistant", Content: r.Content, ReasoningContent: r.Reasoning, ToolCalls: r.ToolCalls})
			for _, call := range r.ToolCalls {
				names = append(names, call.Function.Name)
				result := `{"error":"unknown tool"}`
				if call.Function.Name == "web_search" {
					result = `{"url":"https://example.invalid/jamsil","snippet":"기온은 상세 페이지에서 확인하세요."}`
				}
				if call.Function.Name == "web_fetch" {
					result = `{"fixture":true,"location":"잠실","min_celsius":17,"max_celsius":28}`
				}
				messages = append(messages, Message{Role: "tool", ToolCallID: call.ID, Content: result})
			}
		}
		passed := strings.Contains(strings.Join(names, ","), "web_fetch") && strings.Contains(final, "17") && strings.Contains(final, "28") && !loop
		records = append(records, map[string]any{"trial": trial + 1, "passed": passed, "get_loop": loop, "calls": names, "final": final, "trace": trace})
		t.Logf("RESULT trial=%d passed=%v calls=%v get_loop=%v", trial+1, passed, names, loop)
		if !passed {
			t.Errorf("trial %d: tool continuation failed; calls=%v get_loop=%v", trial+1, names, loop)
		}
	}
}
