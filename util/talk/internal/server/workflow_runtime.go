package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sparktalk/internal/config"
	"sparktalk/internal/llm"
	"sparktalk/internal/skills"
	"sparktalk/internal/workflows"
	"strings"
)

type workflowStageKey struct{}
type workflowStage struct {
	Skills   map[string]skills.Skill
	Evidence []workflows.Evidence
}

func workflowMarker(messages []llm.Message) (string, string) {
	if len(messages) == 0 || messages[len(messages)-1].Role != "user" {
		return "", ""
	}
	value := messages[len(messages)-1].Content
	var text string
	switch v := value.(type) {
	case string:
		text = v
	case []map[string]any:
		if len(v) > 0 {
			text, _ = v[0]["text"].(string)
		}
	}
	fields := strings.Fields(text)
	if len(fields) == 0 {
		return "", text
	}
	if strings.HasPrefix(fields[0], "@workflow:") || strings.HasPrefix(fields[0], "@resume:") {
		return fields[0], strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(text), fields[0]))
	}
	return "", text
}
func workflowStartTool(items []workflows.Definition) llm.Tool {
	names := []string{}
	hints := []string{}
	for _, x := range items {
		if x.Enabled {
			names = append(names, x.Name)
			hints = append(hints, x.Name+": "+x.Description)
		}
	}
	p, _ := json.Marshal(map[string]any{"type": "object", "properties": map[string]any{"name": map[string]any{"type": "string", "enum": names}}, "required": []string{"name"}, "additionalProperties": false})
	return llm.Tool{Type: "function", Function: llm.ToolFunction{Name: "workflow_start", Description: "Start a saved multi-stage task procedure only if the user requests substantial work matching it. Do not use for simple questions. Call alone before performing any work tools. Available: " + strings.Join(hints, "; "), Parameters: p}}
}
func workflowReportTool() llm.Tool {
	return llm.Tool{Type: "function", Function: llm.ToolFunction{Name: "workflow_report", Description: "Finish the current stage. Report blocked/failed honestly. Evidence IDs must be IDs of tools actually called in this stage. Call this alone after completing the stage instead of a normal final answer.", Parameters: json.RawMessage(`{"type":"object","properties":{"status":{"type":"string","enum":["completed","blocked","failed"]},"summary":{"type":"string","description":"Complete deliverable and concise handoff, preserving code and source links. Do not claim unexecuted tests."},"handoff":{"type":"string","description":"Brief handoff: changes, evidence, unresolved issues. Keep below 1500 characters."},"evidence":{"type":"array","items":{"type":"string"}}},"required":["status","summary","evidence"],"additionalProperties":false}`)}}
}
func evidenceMatches(step workflows.Step, e workflows.Evidence) bool {
	if e.Tool != step.VerifyTool || e.Error != "" || e.Result == "" {
		return false
	}
	var obj map[string]any
	if json.Unmarshal([]byte(e.Result), &obj) == nil {
		if x, ok := obj["error"]; ok && x != nil && x != "" {
			return false
		}
		if step.VerifyTool == "ssh_exec" {
			code, ok := obj["exit_code"].(float64)
			if !ok || code != 0 {
				return false
			}
		}
	} else if step.VerifyTool == "ssh_exec" {
		return false
	}
	if step.VerifyCommand != "" {
		var args struct {
			Command string `json:"command"`
		}
		if json.Unmarshal([]byte(e.Arguments), &args) != nil || strings.TrimSpace(args.Command) != strings.TrimSpace(step.VerifyCommand) {
			return false
		}
	}
	return true
}
func (s *Server) runWorkflow(ctx context.Context, session, marker, goal string, messages []llm.Message, client *llm.Client, model, effort, prompt string, cfg config.ToolsConfig, enabled bool, emit eventEmitter, sink mediaAttachmentSink) (answer completionResult, runErr error) {
	if s == nil || s.db == nil || !cfg.SkillsEnabled {
		return answer, fmt.Errorf("작업 절차를 사용할 수 없습니다")
	}
	var run workflows.Run
	if strings.HasPrefix(marker, "@resume:") {
		var e error
		run, e = s.db.ClaimWorkflowRun(strings.TrimPrefix(marker, "@resume:"), session)
		if e != nil {
			return answer, e
		}
	} else {
		name := strings.TrimPrefix(marker, "@workflow:")
		items, e := s.allWorkflows()
		if e != nil {
			return answer, e
		}
		found := false
		for _, x := range items {
			if x.Name == name && x.Enabled {
				run.Definition = x
				found = true
				break
			}
		}
		if !found {
			return answer, fmt.Errorf("사용할 수 없는 작업 절차: %s", name)
		}
		all, e := s.allSkills()
		if e != nil {
			return answer, e
		}
		run.Skills = map[string]skills.Skill{}
		for _, x := range all {
			run.Skills[x.Name] = x
		}
		// Save only referenced instructions, freezing the procedure for later resume.
		used := map[string]skills.Skill{}
		for _, step := range run.Definition.Steps {
			for _, n := range step.Skills {
				v, ok := run.Skills[n]
				if !ok || !v.Enabled {
					return answer, fmt.Errorf("필요한 스킬을 사용할 수 없습니다: %s", n)
				}
				used[n] = v
			}
		}
		run.Skills = used
		var b [16]byte
		if _, e = rand.Read(b[:]); e != nil {
			return answer, e
		}
		run.ID = hex.EncodeToString(b[:])
		run.SessionID = session
		run.Goal = goal
		run.Status = "running"
		for range run.Definition.Steps {
			run.Steps = append(run.Steps, workflows.StepState{Status: "pending", Evidence: []workflows.Evidence{}})
		}
		if e = s.db.SaveWorkflowRun(&run); e != nil {
			return answer, e
		}
	}
	publish := func() error {
		if e := s.db.SaveWorkflowRun(&run); e != nil {
			return e
		}
		public := run
		public.Skills = nil
		return emit("workflow", public)
	}
	defer func() {
		if runErr != nil {
			run.Status = "paused"
			if run.Current < len(run.Steps) {
				step := &run.Steps[run.Current]
				if step.Status == "running" {
					step.Status = "paused"
				}
				step.Error = runErr.Error()
			}
			if e := s.db.SaveWorkflowRun(&run); e != nil {
				runErr = fmt.Errorf("%v; 진행 저장 실패: %w", runErr, e)
			}
			public := run
			public.Skills = nil
			_ = emit("workflow", public)
		}
	}()
	for run.Current < len(run.Steps) {
		if e := ctx.Err(); e != nil {
			return answer, e
		}
		index := run.Current
		step := run.Definition.Steps[index]
		state := &run.Steps[index]
		previousEvidence := append([]workflows.Evidence{}, state.Evidence...)
		previousError := state.Error
		if state.Attempts > 0 {
			state.History = append(state.History, workflows.Attempt{Status: state.Status, Summary: state.Summary, Error: state.Error, Evidence: previousEvidence})
		}
		state.Status = "running"
		state.Error = ""
		state.Attempts++
		state.Evidence = nil
		if e := publish(); e != nil {
			return answer, e
		}
		stage := &workflowStage{Skills: map[string]skills.Skill{}}
		prefixes := []string{}
		for _, name := range step.Skills {
			stage.Skills[name] = run.Skills[name]
			prefixes = append(prefixes, "@skill:"+name)
		}
		handoff := ""
		for i := 0; i < index; i++ {
			content := run.Steps[i].Handoff
			if i == index-1 {
				content = run.Steps[i].Summary
			}
			handoff += fmt.Sprintf("\nCompleted stage %d (%s):\n%s\n", i+1, run.Definition.Steps[i].Name, content)
		}
		guide := strings.Join(prefixes, " ") + "\nOriginal task:\n" + run.Goal + fmt.Sprintf("\nCurrent stage %d/%d: %s\nGoal: %s\nCompletion criteria: %s\n", index+1, len(run.Steps), step.Name, step.Goal, step.DoneWhen) + handoff
		if len(previousEvidence) > 0 {
			prior, _ := json.Marshal(previousEvidence)
			guide += "\nPrevious attempt's actual tool results (do not blindly repeat side effects; inspect current state first):\n" + string(prior)
		}
		if previousError != "" {
			guide += "\nPrevious interruption: " + previousError
		}
		if state.Summary != "" {
			guide += "\nPrevious attempt (incomplete or needing correction):\n" + state.Summary
		}
		if step.VerifyTool != "" {
			guide += "\nCompletion requires successful tool evidence from this stage: " + step.VerifyTool + ". Include the tool call IDs in workflow_report.evidence. Without evidence report blocked, never claim verification."
		}
		if step.VerifyCommand != "" {
			guide += "\nRequired verification command (normal SSH permissions still apply): " + step.VerifyCommand
		}
		guide += "\nWork only on this stage. Do not start other workflows. End with workflow_report, including a complete deliverable for the next stage. User intent and existing tool permissions remain authoritative."
		stageMessages := append(append([]llm.Message{}, messages...), llm.Message{Role: "user", Content: guide})
		stageEmit := func(kind string, value any) error {
			if kind == "tool_result" {
				// Persist observed effects before the next model request, including on process restart.
				state.Evidence = append([]workflows.Evidence{}, stage.Evidence...)
				if e := s.db.SaveWorkflowRun(&run); e != nil {
					return e
				}
			}
			if kind == "delta" {
				return nil
			}
			return emit(kind, value)
		}
		result, e := runCompletionLoopForSessionWithMedia(s, session, context.WithValue(ctx, workflowStageKey{}, stage), client, stageMessages, model, effort, prompt, cfg, enabled, stageEmit, sink)
		answer.Reasoning = mergeReasoning(answer.Reasoning, result.Reasoning)
		answer.ToolTrace = append(answer.ToolTrace, result.ToolTrace...)
		answer.Attachments = append(answer.Attachments, result.Attachments...)
		state.Evidence = stage.Evidence
		state.Summary = result.Content
		if result.Report != nil {
			state.Summary = result.Report.Summary
			state.Handoff = compactHistoryText(result.Report.Handoff, 1500)
			if state.Handoff == "" {
				state.Handoff = compactHistoryText(state.Summary, 1500)
			}
		}
		if e != nil {
			answer.Content = state.Summary
			if answer.Content != "" {
				_ = emit("delta", map[string]string{"delta": answer.Content})
			}
			return answer, e
		}
		if result.Report == nil {
			state.Status = "unverified"
			answer.Content = state.Summary
			if answer.Content != "" {
				_ = emit("delta", map[string]string{"delta": answer.Content})
			}
			return answer, fmt.Errorf("%s: 단계 완료 보고가 없어 검증하지 못했습니다", step.Name)
		}
		report := result.Report
		state.Status = report.Status
		if report.Status == "completed" && strings.TrimSpace(report.Summary) == "" {
			state.Status = "unverified"
		}
		if state.Status == "completed" && step.VerifyTool != "" {
			valid := false
			for _, id := range report.Evidence {
				for _, proof := range state.Evidence {
					if proof.ID == id && evidenceMatches(step, proof) {
						valid = true
					}
				}
			}
			if !valid {
				state.Status = "unverified"
				state.Error = "필요한 도구의 성공 근거가 없습니다"
			}
		}
		if state.Status != "completed" {
			if state.Status == "failed" && step.OnFailure >= 0 && state.Retries < step.MaxRetries {
				state.Retries++
				failure := state.Summary
				for j := step.OnFailure; j <= index; j++ {
					run.Steps[j].Status = "pending"
				}
				run.Steps[step.OnFailure].Summary += "\nVerification failed; correct this before retrying:\n" + failure
				run.Current = step.OnFailure
				if e = publish(); e != nil {
					return answer, e
				}
				continue
			}
			answer.Content = state.Summary
			if answer.Content != "" {
				_ = emit("delta", map[string]string{"delta": answer.Content})
			}
			return answer, fmt.Errorf("%s: %s. 실행 기록에서 근거를 확인하고 이어갈 수 있습니다", step.Name, map[string]string{"failed": "검증 실패", "blocked": "진행에 필요한 정보나 도구가 없습니다", "unverified": "완료 근거를 검증하지 못했습니다"}[state.Status])
		}
		run.Current++
		if e = publish(); e != nil {
			return answer, e
		}
	}
	run.Status = "completed"
	answer.Content = run.Steps[len(run.Steps)-1].Summary
	if e := publish(); e != nil {
		return answer, e
	}
	if e := emit("delta", map[string]string{"delta": answer.Content}); e != nil {
		return answer, e
	}
	return answer, nil
}
