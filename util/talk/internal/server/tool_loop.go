package server

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"sort"
	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/performance"
	"sparktalk/internal/skills"
	"sparktalk/internal/workflows"
)

type completionResult struct {
	Performance *performance.Summary
	Report      *workflows.Report
	Content     string
	Reasoning   string
	ToolTrace   []db.ToolEvent
	Attachments []db.Attachment
}

type eventEmitter func(event string, payload any) error

const toolLimitFinalInstruction = "The tool execution limit has been reached. Do not request or imitate any more tool calls and do not output tool protocol markup. Give a concise final answer using only the tool results already present. If more inspection is necessary, say so plainly. Separate verified facts, estimates, and unknowns. Never invent dates, schedules, roster changes, source titles, or URLs to fill missing evidence. Do not present predictions as confirmed. Cite actual supporting URLs from the available results; if the evidence is insufficient, provide a partial answer and identify what remains unverified."
const emptyFinalRetryInstruction = "Your previous response ended during reasoning without a final answer. Do not reason further or call tools. Give the final answer now using only the conversation and tool results already present."

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
	stage, inStage := ctx.Value(workflowStageKey{}).(*workflowStage)
	if !inStage {
		marker, goal := workflowMarker(messages)
		if marker != "" {
			return server.runWorkflow(ctx, sessionID, marker, goal, messages, client, model, reasoningEffort, systemPrompt, toolConfig, toolsEnabled, emit, mediaSink)
		}
	}
	registry := newCompletionToolRegistry(server, sessionID, toolConfig, toolsEnabled, mediaSink)
	if inStage {
		for name := range stage.Skills {
			if _, ok := registry.skills[name]; !ok {
				return completionResult{}, fmt.Errorf("단계 스킬 %s에 필요한 도구가 없거나 스킬이 비활성입니다", name)
			}
		}
		for name := range registry.skills {
			delete(registry.skills, name)
		}
		for name, item := range stage.Skills {
			registry.skills[name] = item
		}
		currentSkills := []skills.Skill{}
		for _, item := range stage.Skills {
			currentSkills = append(currentSkills, item)
		}
		sort.Slice(currentSkills, func(i, j int) bool { return currentSkills[i].Name < currentSkills[j].Name })
		for i, definition := range registry.definitions {
			if definition.Function.Name == "skill_view" {
				registry.definitions[i] = skillViewDefinition(currentSkills)
			}
		}
		for i, instruction := range registry.prompts {
			if strings.HasPrefix(instruction, "On-demand SparkTalk skills are available.") {
				registry.prompts[i] = skillIndexPrompt(currentSkills)
			}
		}
		registry.definitions = append(registry.definitions, workflowReportTool())
	} else if server != nil && server.db != nil && toolConfig.SkillsEnabled && len(requestedSkills(messages)) == 0 {
		if items, err := server.allWorkflows(); err != nil {
			return completionResult{}, err
		} else {
			available := []workflows.Definition{}
			for _, item := range items {
				ok := item.Enabled
				for _, step := range item.Steps {
					for _, name := range step.Skills {
						if _, found := registry.skills[name]; !found {
							ok = false
						}
					}
					if step.VerifyTool != "" {
						if _, found := registry.handlers[step.VerifyTool]; !found {
							ok = false
						}
					}
				}
				if ok {
					available = append(available, item)
				}
			}
			if len(available) > 0 {
				registry.definitions = append(registry.definitions, workflowStartTool(available))
			}
		}
	}

	if run, ok := ctx.Value(contextRunKey{}).(*contextRun); ok {
		run.state.Skills = nil
		run.state.SkillTokens = 0
	}
	trace, selectErr := registry.selectSkills(ctx, messages)
	if selectErr != nil {
		return completionResult{}, selectErr
	}
	for i, record := range trace {
		id := fmt.Sprintf("selected-skill-%d", i)
		if err := emit("tool_start", map[string]any{"id": id, "name": record.Name, "arguments": record.Arguments}); err != nil {
			return completionResult{}, err
		}
		if err := emit("tool_result", map[string]any{"id": id, "name": record.Name, "arguments": record.Arguments, "result": record.Result, "error": ""}); err != nil {
			return completionResult{}, err
		}
	}
	useTools := len(registry.definitions) > 0
	// SGLang accepts only one system message and requires it at index zero.
	referencePolicy, _ := ctx.Value(referencePolicyKey{}).(bool)
	conversation := assembleModelConversation(systemPrompt, messages, registry.prompts, toolConfig.MaxRounds*3, referencePolicy)
	conversation = retainLatestVideoInput(conversation)

	refs := make(map[string]int64)
	requestOnce := func(messages []llm.Message, effort string, definitions []llm.Tool, receiver func(string, string) error) (llm.StreamResult, error) {
		prepared, err := updateRequestContext(ctx, client.InputMessages(messages), definitions, refs, emit)
		if err != nil {
			return llm.StreamResult{}, err
		}
		copy(messages, prepared)
		result, err := client.Stream(ctx, prepared, model, effort, definitions, receiver)
		emitContextUsage(ctx, result.Usage, emit)
		return result, err
	}
	request := func(messages []llm.Message, effort string, definitions []llm.Tool, receiver func(string, string) error) (llm.StreamResult, error) {
		return continueLimitedText(ctx, messages, effort, definitions, receiver, requestOnce, !inStage)
	}
	if !useTools {
		result, err := request(conversation, reasoningEffort, nil, textEmitter(emit))
		if err == nil {
			result, err = recoverEmptyFinal(ctx, client, conversation, model, result, textEmitter(emit))
		}
		return completionResult{Content: result.Content, Reasoning: result.Reasoning, ToolTrace: trace}, err
	}

	var allReasoning strings.Builder
	outputAttachments := []db.Attachment{}
	toolRounds := 0
	procedureRounds := 0
	for {
		if toolRounds >= toolConfig.MaxRounds && !inStage {
			conversation = append(conversation, llm.Message{Role: "user", Content: toolLimitFinalInstruction})
			finalEmitter := func(kind, text string) error {
				if kind == "reasoning" {
					return emit(kind, map[string]string{"delta": text})
				}
				return nil
			}
			result, err := request(conversation, reasoningEffort, nil, finalEmitter)
			if err == nil {
				result, err = recoverEmptyFinal(ctx, client, conversation, model, result, finalEmitter)
			}
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
			if err == nil {
				err = fmt.Errorf("도구 실행 한도(%d라운드)에 도달했습니다. 마지막 응답은 확보된 결과로 정리했습니다. 추가 작업은 설정 > 기능에서 최대 호출 라운드를 늘리거나 이어서 요청하세요.", toolConfig.MaxRounds)
			}
			return completionResult{Content: content, Reasoning: allReasoning.String(), ToolTrace: trace, Attachments: outputAttachments}, err
		}
		definitions := registry.definitions
		if !inStage && toolRounds > 0 {
			definitions = nil
			for _, definition := range registry.definitions {
				if definition.Function.Name != "workflow_start" {
					definitions = append(definitions, definition)
				}
			}
		}
		if inStage && toolRounds >= toolConfig.MaxRounds {
			definitions = []llm.Tool{workflowReportTool()}
			conversation = append(conversation, llm.Message{Role: "user", Content: "No more work-tool rounds are available. Submit workflow_report now using existing evidence. If this stage is incomplete, report blocked or failed honestly."})
		}
		result, err := request(conversation, reasoningEffort, definitions, textEmitter(emit))
		if err != nil {
			if allReasoning.Len() > 0 && result.Reasoning != "" {
				allReasoning.WriteString("\n\n")
			}
			allReasoning.WriteString(result.Reasoning)
			return completionResult{Content: result.Content, Reasoning: allReasoning.String(), ToolTrace: trace, Attachments: outputAttachments}, err
		}
		for _, call := range result.ToolCalls {
			if call.Function.Name == "workflow_report" && inStage {
				if len(result.ToolCalls) != 1 {
					return completionResult{}, fmt.Errorf("단계 보고는 다른 도구 호출과 분리해야 합니다")
				}
				var report workflows.Report
				if json.Unmarshal([]byte(call.Function.Arguments), &report) != nil || (report.Status != "completed" && report.Status != "blocked" && report.Status != "failed") {
					return completionResult{}, fmt.Errorf("잘못된 단계 완료 보고입니다")
				}
				return completionResult{Content: report.Summary, Reasoning: mergeReasoning(allReasoning.String(), result.Reasoning), ToolTrace: trace, Attachments: outputAttachments, Report: &report}, nil
			}
			if call.Function.Name == "workflow_start" && !inStage {
				if len(result.ToolCalls) != 1 || toolRounds > 0 {
					return completionResult{}, fmt.Errorf("작업 절차 시작은 다른 도구 호출과 분리해야 합니다")
				}
				var args struct {
					Name string `json:"name"`
				}
				if json.Unmarshal([]byte(call.Function.Arguments), &args) != nil {
					return completionResult{}, fmt.Errorf("작업 절차 선택이 잘못됐습니다")
				}
				_, goal := workflowMarker(messages)
				return server.runWorkflow(ctx, sessionID, "@workflow:"+args.Name, goal, messages, client, model, reasoningEffort, systemPrompt, toolConfig, toolsEnabled, emit, mediaSink)
			}
		}
		if inStage && toolRounds >= toolConfig.MaxRounds && len(result.ToolCalls) > 0 {
			return completionResult{Reasoning: allReasoning.String(), ToolTrace: trace}, fmt.Errorf("단계의 도구 실행 한도에 도달했습니다")
		}
		if len(result.ToolCalls) == 0 {
			result, err = recoverEmptyFinal(ctx, client, conversation, model, result, textEmitter(emit))
			if allReasoning.Len() > 0 && result.Reasoning != "" {
				allReasoning.WriteString("\n\n")
			}
			allReasoning.WriteString(result.Reasoning)
			content, _ := cleanToolProtocol(result.Content)
			return completionResult{Content: content, Reasoning: allReasoning.String(), ToolTrace: trace, Attachments: outputAttachments}, err
		}
		if allReasoning.Len() > 0 && result.Reasoning != "" {
			allReasoning.WriteString("\n\n")
		}
		allReasoning.WriteString(result.Reasoning)

		conversation = append(conversation, llm.Message{
			Role: "assistant", Content: result.Content, ToolCalls: result.ToolCalls,
			ReasoningContent: result.Reasoning,
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
			if server != nil && server.db != nil && sessionID != "" {
				var anchor int64
				if run, ok := ctx.Value(contextRunKey{}).(*contextRun); ok {
					anchor = run.state.ActiveEnd
				}
				id, archiveErr := server.db.ArchiveContextTool(sessionID, call.Function.Name, toolResult, anchor)
				if archiveErr == nil {
					if call.Function.Name != "skill_view" {
						refs[call.ID] = id
					}
					record.ArchiveID = id
				}
			}
			if inStage && call.Function.Name != "skill_view" {
				stage.Evidence = append(stage.Evidence, workflows.Evidence{ID: call.ID, Tool: record.Name, Arguments: record.Arguments, Result: record.Result, Error: record.Error})
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
		onlySkills := len(result.ToolCalls) > 0
		for _, call := range result.ToolCalls {
			if call.Function.Name != "skill_view" {
				onlySkills = false
			}
		}
		if onlySkills {
			procedureRounds++
			if procedureRounds > 16 {
				return completionResult{Reasoning: allReasoning.String(), ToolTrace: trace}, fmt.Errorf("스킬을 반복해서 불러와 작업을 중단했습니다")
			}
		} else {
			toolRounds++
		}
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

func recoverEmptyFinal(
	ctx context.Context,
	client *llm.Client,
	conversation []llm.Message,
	model string,
	result llm.StreamResult,
	emit func(kind, text string) error,
) (llm.StreamResult, error) {
	if strings.TrimSpace(result.Content) != "" || strings.TrimSpace(result.Reasoning) == "" || len(result.ToolCalls) > 0 {
		return result, nil
	}
	retryConversation := append([]llm.Message(nil), conversation...)
	retryConversation = append(retryConversation, llm.Message{Role: "user", Content: emptyFinalRetryInstruction})
	notify := func(string, any) error { return nil }
	if run, ok := ctx.Value(contextRunKey{}).(*contextRun); ok && run.emit != nil {
		notify = run.emit
	}
	prepared, budgetErr := updateRequestContext(ctx, client.InputMessages(retryConversation), nil, nil, notify)
	if budgetErr != nil {
		return result, budgetErr
	}
	retry, err := client.Stream(ctx, prepared, model, "off", nil, emit)
	emitContextUsage(ctx, retry.Usage, notify)
	retry.Reasoning = mergeReasoning(result.Reasoning, retry.Reasoning)
	if err != nil {
		return retry, err
	}
	if strings.TrimSpace(retry.Content) == "" {
		return retry, fmt.Errorf("model returned no final answer after retry")
	}
	return retry, nil
}

func mergeReasoning(first, second string) string {
	if first == "" {
		return second
	}
	if second == "" {
		return first
	}
	return first + "\n\n" + second
}
