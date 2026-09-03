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
	Content     string
	Reasoning   string
	ToolTrace   []db.ToolEvent
	Attachments []db.Attachment
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
	// SGLang accepts only one system message and requires it at index zero.
	conversation := assembleModelConversation(systemPrompt, messages, registry.prompts, toolConfig.MaxRounds*3)
	conversation = retainLatestVideoInput(conversation)

	if !useTools {
		result, err := client.Stream(ctx, conversation, model, reasoningEffort, nil, textEmitter(emit))
		return completionResult{Content: result.Content, Reasoning: result.Reasoning}, err
	}

	var allReasoning strings.Builder
	trace := []db.ToolEvent{}
	outputAttachments := []db.Attachment{}
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
					return completionResult{Content: content, Reasoning: allReasoning.String(), ToolTrace: trace, Attachments: outputAttachments}, emitErr
				}
			}
			return completionResult{Content: content, Reasoning: allReasoning.String(), ToolTrace: trace, Attachments: outputAttachments}, err
		}
		result, err := client.Stream(ctx, conversation, model, reasoningEffort, registry.definitions, textEmitter(emit))
		if allReasoning.Len() > 0 && result.Reasoning != "" {
			allReasoning.WriteString("\n\n")
		}
		allReasoning.WriteString(result.Reasoning)
		if err != nil {
			return completionResult{Content: result.Content, Reasoning: allReasoning.String(), ToolTrace: trace, Attachments: outputAttachments}, err
		}
		if len(result.ToolCalls) == 0 {
			content, _ := cleanToolProtocol(result.Content)
			return completionResult{Content: content, Reasoning: allReasoning.String(), ToolTrace: trace, Attachments: outputAttachments}, nil
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
			if server != nil && server.db != nil && call.Function.Name != "ssh_exec" && call.Function.Name != "memory_propose" && call.Function.Name != "memory_manage" && call.Function.Name != "knowledge_import" {
				decision, detail := "executed", ""
				if toolErr != nil {
					decision, detail = "execution_error", compactHistoryText(toolErr.Error(), 300)
				}
				_ = server.db.AddToolAudit(sessionID, call.Function.Name, "", "execute", decision, detail)
			}
			toolResult := execution.Result
			toolFollowups = append(toolFollowups, execution.Followups...)
			if toolErr == nil && execution.Attachment != nil {
				if emitErr := emit("media_attached", *execution.Attachment); emitErr != nil {
					return completionResult{Reasoning: allReasoning.String(), ToolTrace: trace}, emitErr
				}
			}
			if toolErr == nil {
				for _, attachment := range execution.Attachments {
					outputAttachments = append(outputAttachments, attachment)
					payload := map[string]any{"id": attachment.ID, "name": attachment.Name, "mime": attachment.MIME, "size": attachment.Size, "url": attachment.URL, "target_role": "assistant"}
					if emitErr := emit("media_attached", payload); emitErr != nil {
						return completionResult{Reasoning: allReasoning.String(), ToolTrace: trace}, emitErr
					}
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
		conversation = retainLatestVideoInput(conversation)
		toolRounds++
	}
}

// retainLatestVideoInput enforces the conservative one-video contract used by
// the local multimodal servers. Visible history remains untouched; this only
// removes older raw video parts from the request assembled for the model.
func retainLatestVideoInput(messages []llm.Message) []llm.Message {
	keptVideo := false
	out := append([]llm.Message(nil), messages...)
	for messageIndex := len(out) - 1; messageIndex >= 0; messageIndex-- {
		parts, ok := out[messageIndex].Content.([]map[string]any)
		if !ok {
			continue
		}
		filtered := make([]map[string]any, 0, len(parts))
		for partIndex := len(parts) - 1; partIndex >= 0; partIndex-- {
			part := parts[partIndex]
			if part["type"] == "video_url" {
				if keptVideo {
					continue
				}
				keptVideo = true
			}
			filtered = append(filtered, part)
		}
		for left, right := 0, len(filtered)-1; left < right; left, right = left+1, right-1 {
			filtered[left], filtered[right] = filtered[right], filtered[left]
		}
		out[messageIndex].Content = filtered
	}
	return out
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
