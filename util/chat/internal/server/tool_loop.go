package server

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
)

type completionResult struct {
	Content   string
	Reasoning string
	ToolTrace []db.ToolEvent
}

type eventEmitter func(event string, payload any) error

const toolLimitFinalInstruction = "The tool execution limit has been reached. Do not request or imitate any more tool calls and do not output tool protocol markup. Give a concise final answer using only the tool results already present. If more inspection is necessary, say so plainly."

var toolProtocolBlock = regexp.MustCompile(`(?is)<tool_call\b[^>]*>.*?</tool_call>`)
var danglingToolProtocol = regexp.MustCompile(`(?is)<tool_call\b[^>]*>.*$`)

func runCompletionLoop(
	ctx context.Context,
	client *llm.Client,
	messages []llm.Message,
	model, reasoningEffort string,
	systemPrompt string,
	toolConfig config.ToolsConfig,
	toolsEnabled bool,
	emit eventEmitter,
) (completionResult, error) {
	return runCompletionLoopForServer(nil, ctx, client, messages, model, reasoningEffort, systemPrompt, toolConfig, toolsEnabled, emit)
}

func runCompletionLoopForServer(
	server *Server,
	ctx context.Context,
	client *llm.Client,
	messages []llm.Message,
	model, reasoningEffort string,
	systemPrompt string,
	toolConfig config.ToolsConfig,
	toolsEnabled bool,
	emit eventEmitter,
) (completionResult, error) {
	return runCompletionLoopForSession(server, "", ctx, client, messages, model, reasoningEffort, systemPrompt, toolConfig, toolsEnabled, emit)
}

func runCompletionLoopForSession(
	server *Server,
	sessionID string,
	ctx context.Context,
	client *llm.Client,
	messages []llm.Message,
	model, reasoningEffort string,
	systemPrompt string,
	toolConfig config.ToolsConfig,
	toolsEnabled bool,
	emit eventEmitter,
) (completionResult, error) {
	return runCompletionLoopForSessionWithMedia(server, sessionID, ctx, client, messages, model, reasoningEffort, systemPrompt, toolConfig, toolsEnabled, emit, nil)
}

func runCompletionLoopForSessionWithMedia(
	server *Server,
	sessionID string,
	ctx context.Context,
	client *llm.Client,
	messages []llm.Message,
	model, reasoningEffort string,
	systemPrompt string,
	toolConfig config.ToolsConfig,
	toolsEnabled bool,
	emit eventEmitter,
	mediaSink mediaAttachmentSink,
) (completionResult, error) {
	registry := newCompletionToolRegistry(server, sessionID, toolConfig, toolsEnabled, mediaSink)
	useTools := len(registry.definitions) > 0
	systemParts := make([]string, 0, 3)
	if prompt := strings.TrimSpace(systemPrompt); prompt != "" {
		systemParts = append(systemParts, prompt)
	}
	systemParts = append(systemParts, registry.prompts...)
	// SGLang accepts only one system message and requires it to be the first
	// message. Context checkpoints arrive as a leading system message, so merge
	// them with the global/tool instructions instead of emitting a second one.
	leadingSystems := 0
	for leadingSystems < len(messages) && messages[leadingSystems].Role == "system" {
		if content, ok := messages[leadingSystems].Content.(string); ok && strings.TrimSpace(content) != "" {
			systemParts = append(systemParts, content)
		}
		leadingSystems++
	}
	conversation := make([]llm.Message, 0, len(messages)+1+toolConfig.MaxRounds*3)
	if len(systemParts) > 0 {
		conversation = append(conversation, llm.Message{Role: "system", Content: strings.Join(systemParts, "\n\n")})
	}
	conversation = append(conversation, messages[leadingSystems:]...)

	if !useTools {
		result, err := client.Stream(ctx, conversation, model, reasoningEffort, nil, textEmitter(emit))
		return completionResult{Content: result.Content, Reasoning: result.Reasoning}, err
	}

	var allReasoning strings.Builder
	trace := []db.ToolEvent{}
	toolRounds := 0
	for {
		if toolRounds >= toolConfig.MaxRounds {
			conversation = append(conversation, llm.Message{Role: "user", Content: toolLimitFinalInstruction})
			result, err := client.Stream(ctx, conversation, model, reasoningEffort, nil, func(kind, text string) error {
				if kind == "reasoning" {
					return emit(kind, map[string]string{"delta": text})
				}
				return nil
			})
			if allReasoning.Len() > 0 && result.Reasoning != "" {
				allReasoning.WriteString("\n\n")
			}
			allReasoning.WriteString(result.Reasoning)
			content, leaked := cleanToolProtocol(result.Content)
			if leaked && content == "" {
				content = fmt.Sprintf("추가 도구 호출이 필요하지만 실행 한도(%d회)에 도달했습니다. 최대 호출 라운드를 늘리거나 새 요청으로 계속해 주세요.", toolConfig.MaxRounds)
			}
			if content != "" {
				if emitErr := emit("delta", map[string]string{"delta": content}); emitErr != nil {
					return completionResult{Content: content, Reasoning: allReasoning.String(), ToolTrace: trace}, emitErr
				}
			}
			return completionResult{Content: content, Reasoning: allReasoning.String(), ToolTrace: trace}, err
		}
		result, err := client.Stream(ctx, conversation, model, reasoningEffort, registry.definitions, textEmitter(emit))
		if allReasoning.Len() > 0 && result.Reasoning != "" {
			allReasoning.WriteString("\n\n")
		}
		allReasoning.WriteString(result.Reasoning)
		if err != nil {
			return completionResult{Content: result.Content, Reasoning: allReasoning.String(), ToolTrace: trace}, err
		}
		if len(result.ToolCalls) == 0 {
			content, _ := cleanToolProtocol(result.Content)
			return completionResult{Content: content, Reasoning: allReasoning.String(), ToolTrace: trace}, nil
		}

		conversation = append(conversation, llm.Message{
			Role: "assistant", Content: result.Content, ToolCalls: result.ToolCalls,
		})
		toolFollowups := make([]llm.Message, 0, 1)
		for _, call := range result.ToolCalls {
			if err := emit("tool_start", map[string]any{
				"id": call.ID, "name": call.Function.Name, "arguments": call.Function.Arguments,
			}); err != nil {
				return completionResult{Reasoning: allReasoning.String(), ToolTrace: trace}, err
			}
			execution, toolErr := registry.execute(ctx, call, conversation, emit)
			toolResult := execution.Result
			toolFollowups = append(toolFollowups, execution.Followups...)
			if toolErr == nil && execution.Attachment != nil {
				if emitErr := emit("media_attached", *execution.Attachment); emitErr != nil {
					return completionResult{Reasoning: allReasoning.String(), ToolTrace: trace}, emitErr
				}
			}
			record := db.ToolEvent{Name: call.Function.Name, Arguments: call.Function.Arguments, Result: toolResult}
			if toolErr != nil {
				record.Error = toolErr.Error()
				data, _ := json.Marshal(map[string]string{"error": toolErr.Error()})
				toolResult = string(data)
			}
			trace = append(trace, record)
			if err := emit("tool_result", map[string]any{
				"id": call.ID, "name": call.Function.Name, "arguments": call.Function.Arguments,
				"result": record.Result, "error": record.Error,
			}); err != nil {
				return completionResult{Reasoning: allReasoning.String(), ToolTrace: trace}, err
			}
			conversation = append(conversation, llm.Message{
				Role: "tool", Content: toolResult, ToolCallID: call.ID,
			})
		}
		// Every tool result must immediately follow the assistant tool_calls
		// message. Add model-facing media only after all results, otherwise a
		// multi-tool response would produce an invalid role sequence.
		conversation = append(conversation, toolFollowups...)
		toolRounds++
	}
}

func cleanToolProtocol(content string) (string, bool) {
	leaked := toolProtocolBlock.MatchString(content) || danglingToolProtocol.MatchString(content)
	cleaned := toolProtocolBlock.ReplaceAllString(content, "")
	cleaned = danglingToolProtocol.ReplaceAllString(cleaned, "")
	return strings.TrimSpace(cleaned), leaked
}

func textEmitter(emit eventEmitter) func(kind, text string) error {
	return func(kind, text string) error {
		return emit(kind, map[string]string{"delta": text})
	}
}
