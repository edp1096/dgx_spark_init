package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

type Message struct {
	Role       string     `json:"role"`
	Content    any        `json:"content,omitempty"`
	ToolCalls  []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
}

type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type StreamResult struct {
	Content   string
	Reasoning string
	ToolCalls []ToolCall
	Usage     Usage
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type toolCallAccum struct {
	ID        string
	Type      string
	Name      string
	Arguments strings.Builder
}

type Client struct {
	endpoint string
	model    string
	apiKey   string
	http     *http.Client
}

func New(endpoint, model, apiKey string) *Client {
	return &Client{
		endpoint: strings.TrimRight(endpoint, "/"), model: model, apiKey: apiKey,
		http: &http.Client{Timeout: 0},
	}
}

func (c *Client) Models(ctx context.Context) ([]string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.endpoint+"/v1/models", nil)
	if err != nil {
		return nil, err
	}
	c.authorize(req)
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("models: HTTP %d", resp.StatusCode)
	}
	var payload struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return nil, err
	}
	models := make([]string, 0, len(payload.Data))
	for _, item := range payload.Data {
		if item.ID != "" {
			models = append(models, item.ID)
		}
	}
	if len(models) == 0 {
		return nil, fmt.Errorf("models: empty model list")
	}
	return models, nil
}

func (c *Client) Model(ctx context.Context) (string, error) {
	if c.model != "" {
		return c.model, nil
	}
	models, err := c.Models(ctx)
	if err != nil {
		return "", err
	}
	return models[0], nil
}

func (c *Client) Stream(ctx context.Context, messages []Message, model, reasoningEffort string, tools []Tool, emit func(kind, text string) error) (StreamResult, error) {
	if model == "" {
		var err error
		model, err = c.Model(ctx)
		if err != nil {
			return StreamResult{}, err
		}
	}
	payload := map[string]any{
		"model": model, "messages": messages, "stream": true, "temperature": 0.7,
		"separate_reasoning": true, "stream_reasoning": true,
		"stream_options": map[string]bool{"include_usage": true},
	}
	if len(tools) > 0 {
		payload["tools"] = tools
		payload["tool_choice"] = "auto"
	}
	if effort := reasoningValue(reasoningEffort); effort != nil {
		payload["reasoning_effort"] = effort
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return StreamResult{}, err
	}
	resp, err := c.post(ctx, body)
	if err != nil {
		return StreamResult{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		detail, _ := io.ReadAll(io.LimitReader(resp.Body, 8192))
		return StreamResult{}, fmt.Errorf("chat completion: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(detail)))
	}

	var answer, reasoning strings.Builder
	var usage Usage
	toolCalls := make(map[int]*toolCallAccum)
	var toolOrder []int
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" || data == "[DONE]" {
			continue
		}
		var chunk struct {
			Usage   Usage `json:"usage"`
			Choices []struct {
				Delta struct {
					Content          string `json:"content"`
					ReasoningContent string `json:"reasoning_content"`
					Reasoning        string `json:"reasoning"`
					ToolCalls        []struct {
						Index    int    `json:"index"`
						ID       string `json:"id"`
						Type     string `json:"type"`
						Function struct {
							Name      string `json:"name"`
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if chunk.Usage.TotalTokens > 0 || chunk.Usage.PromptTokens > 0 {
			usage = chunk.Usage
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		delta := chunk.Choices[0].Delta
		reasoningDelta := delta.ReasoningContent
		if reasoningDelta == "" {
			reasoningDelta = delta.Reasoning
		}
		if reasoningDelta != "" {
			reasoning.WriteString(reasoningDelta)
			if err := emit("reasoning", reasoningDelta); err != nil {
				return StreamResult{Content: answer.String(), Reasoning: reasoning.String(), ToolCalls: assembleToolCalls(toolOrder, toolCalls)}, err
			}
		}
		if delta.Content != "" {
			answer.WriteString(delta.Content)
			if err := emit("delta", delta.Content); err != nil {
				return StreamResult{Content: answer.String(), Reasoning: reasoning.String(), ToolCalls: assembleToolCalls(toolOrder, toolCalls)}, err
			}
		}
		for _, call := range delta.ToolCalls {
			acc, ok := toolCalls[call.Index]
			if !ok {
				acc = &toolCallAccum{}
				toolCalls[call.Index] = acc
				toolOrder = append(toolOrder, call.Index)
			}
			if call.ID != "" {
				acc.ID = call.ID
			}
			if call.Type != "" {
				acc.Type = call.Type
			}
			acc.Name += call.Function.Name
			acc.Arguments.WriteString(call.Function.Arguments)
		}
	}
	if err := scanner.Err(); err != nil {
		return StreamResult{Content: answer.String(), Reasoning: reasoning.String(), ToolCalls: assembleToolCalls(toolOrder, toolCalls)}, err
	}
	return StreamResult{Content: answer.String(), Reasoning: reasoning.String(), ToolCalls: assembleToolCalls(toolOrder, toolCalls), Usage: usage}, nil
}

func (c *Client) ContextWindow(ctx context.Context) (int, error) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	errors := make([]string, 0, 3)
	for _, path := range []string{"/v1/models", "/model_info", "/get_model_info"} {
		value, err := c.contextWindowFrom(ctx, c.endpoint+path)
		if value > 0 {
			return value, nil
		}
		if err != nil {
			errors = append(errors, err.Error())
		}
	}
	return 0, fmt.Errorf("context window discovery failed: %s", strings.Join(errors, "; "))
}

func (c *Client) contextWindowFrom(ctx context.Context, endpoint string) (int, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return 0, err
	}
	c.authorize(req)
	resp, err := c.http.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("model info: HTTP %d", resp.StatusCode)
	}
	var payload any
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return 0, err
	}
	return findContextWindow(payload), nil
}

func findContextWindow(value any) int {
	if values, ok := value.([]any); ok {
		for _, item := range values {
			if found := findContextWindow(item); found > 0 {
				return found
			}
		}
		return 0
	}
	info, ok := value.(map[string]any)
	if !ok {
		return 0
	}
	for _, key := range []string{"context_len", "max_model_len", "max_total_num_tokens", "max_num_tokens"} {
		if value, ok := info[key].(float64); ok && value > 0 {
			return int(value)
		}
	}
	for _, key := range []string{"data", "models", "model_info"} {
		if found := findContextWindow(info[key]); found > 0 {
			return found
		}
	}
	return 0
}

func (c *Client) SummarizeContext(ctx context.Context, model, previous, transcript string) (string, error) {
	if model == "" {
		var err error
		model, err = c.Model(ctx)
		if err != nil {
			return "", err
		}
	}
	prompt := `Update a durable conversation checkpoint from the supplied previous checkpoint and transcript.
Return concise Markdown with exactly these headings: Objective, Decisions, Constraints, Facts, Artifacts, Completed, Unresolved, Next Steps.
Preserve exact file paths, commands, URLs, numbers, user preferences, failures, and message references. Do not invent information. Attachments must be represented by their names, types, and any conclusions stated in the transcript.`
	content := "Previous checkpoint:\n" + strings.TrimSpace(previous) + "\n\nNew transcript:\n" + transcript
	payload := map[string]any{
		"model":    model,
		"messages": []Message{{Role: "system", Content: prompt}, {Role: "user", Content: content}},
		"stream":   false, "temperature": 0.1, "max_completion_tokens": 2048,
		"reasoning_effort": "none",
	}
	body, _ := json.Marshal(payload)
	ctx, cancel := context.WithTimeout(ctx, 3*time.Minute)
	defer cancel()
	resp, err := c.post(ctx, body)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		detail, _ := io.ReadAll(io.LimitReader(resp.Body, 8192))
		return "", fmt.Errorf("context summary: HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(detail)))
	}
	var result struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}
	if len(result.Choices) == 0 || strings.TrimSpace(result.Choices[0].Message.Content) == "" {
		return "", fmt.Errorf("context summary returned no content")
	}
	return strings.TrimSpace(result.Choices[0].Message.Content), nil
}

func assembleToolCalls(order []int, accs map[int]*toolCallAccum) []ToolCall {
	out := make([]ToolCall, 0, len(order))
	for _, index := range order {
		acc := accs[index]
		if acc == nil || acc.Name == "" {
			continue
		}
		id := acc.ID
		if id == "" {
			id = fmt.Sprintf("call_%d", index)
		}
		callType := acc.Type
		if callType == "" {
			callType = "function"
		}
		out = append(out, ToolCall{ID: id, Type: callType, Function: FunctionCall{
			Name: acc.Name, Arguments: sanitizeToolArgs(acc.Arguments.String()),
		}})
	}
	return out
}

func sanitizeToolArgs(args string) string {
	original := args
	if json.Valid([]byte(args)) {
		return args
	}
	for len(args) > 0 && (args[len(args)-1] == '}' || args[len(args)-1] == ']') {
		args = args[:len(args)-1]
		if json.Valid([]byte(args)) {
			return args
		}
	}
	return original
}

func (c *Client) GenerateTitle(ctx context.Context, model, userText string) (string, error) {
	if model == "" {
		var err error
		model, err = c.Model(ctx)
		if err != nil {
			return "", err
		}
	}
	payload := map[string]any{
		"model": model,
		"messages": []Message{
			{Role: "system", Content: "Create a concise topic title for a chat request. Do not answer or solve the request. Describe its subject and intent. Use the user's language. Return only the title without quotes or terminal punctuation. Maximum 24 characters."},
			{Role: "user", Content: "Chat request:\n" + userText + "\n\nReturn a topic title, not the answer."},
		},
		"stream": false, "temperature": 0.2, "max_completion_tokens": 48,
		"reasoning_effort": "none",
	}
	body, _ := json.Marshal(payload)
	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()
	resp, err := c.post(ctx, body)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("title generation: HTTP %d", resp.StatusCode)
	}
	var result struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}
	if len(result.Choices) == 0 {
		return "", fmt.Errorf("title generation returned no choices")
	}
	title := strings.Trim(strings.TrimSpace(result.Choices[0].Message.Content), "\"'`#* ")
	if runes := []rune(title); len(runes) > 40 {
		title = string(runes[:40])
	}
	return title, nil
}

func (c *Client) Health(ctx context.Context) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	models, err := c.Models(ctx)
	if err != nil {
		return "", err
	}
	if c.model == "" {
		return models[0], nil
	}
	for _, model := range models {
		if model == c.model {
			return model, nil
		}
	}
	return c.model, fmt.Errorf("configured model is not available")
}

func (c *Client) post(ctx context.Context, body []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	c.authorize(req)
	return c.http.Do(req)
}

func (c *Client) authorize(req *http.Request) {
	if c.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
}

func reasoningValue(value string) any {
	value = strings.TrimSpace(value)
	if value == "" {
		return nil
	}
	if number, err := strconv.ParseFloat(value, 64); err == nil && number >= 0 && number <= 0.99 {
		return number
	}
	return value
}
