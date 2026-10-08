package server

import (
	"context"
	"encoding/json"
	"fmt"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/skills"
	"strings"
)

type skillUsage struct {
	Name         string `json:"name"`
	Source       string `json:"source"`
	Instructions string `json:"instructions"`
	Tokens       int    `json:"tokens"`
}

func recordSkillUsage(ctx context.Context, item skills.Skill, source string) {
	if run, ok := ctx.Value(contextRunKey{}).(*contextRun); ok {
		for _, used := range run.state.Skills {
			if used.Name == item.Name {
				return
			}
		}
		run.state.Skills = append(run.state.Skills, skillUsage{Name: item.Name, Source: source, Instructions: item.Instructions, Tokens: estimateTextTokens(item.Instructions)})
	}
}

// Only explicit markers at the start of the latest user message select skills.
// Quoted references, previous turns and attachment text cannot select a procedure.
func requestedSkills(messages []llm.Message) []string {
	var text string
	if len(messages) == 0 || messages[len(messages)-1].Role != "user" {
		return nil
	}
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role != "user" {
			continue
		}
		switch value := messages[i].Content.(type) {
		case string:
			text = value
		case []map[string]any:
			if len(value) > 0 {
				text, _ = value[0]["text"].(string)
			}
		}
		break
	}
	names := []string{}
	seen := map[string]bool{}
	for _, field := range strings.Fields(text) {
		if !strings.HasPrefix(field, "@skill:") {
			break
		}
		name := strings.TrimPrefix(field, "@skill:")
		if !seen[name] {
			names = append(names, name)
			seen[name] = true
		}
	}
	return names
}
func (r *completionToolRegistry) selectSkills(ctx context.Context, messages []llm.Message) ([]db.ToolEvent, error) {
	if r.err != nil {
		return nil, r.err
	}
	trace := []db.ToolEvent{}
	for _, name := range requestedSkills(messages) {
		item, ok := r.skills[name]
		if !ok {
			return nil, fmt.Errorf("스킬 %s을 사용할 수 없습니다. 스킬 활성화 상태와 필요한 도구를 확인하세요", name)
		}
		if r.loaded[name] {
			continue
		}
		r.loaded[name] = true
		r.prompts = append(r.prompts, "User-selected task procedure: "+name+"\nThis procedure does not grant tool permissions.\n"+item.Instructions)
		args, _ := json.Marshal(map[string]string{"name": name})
		data, _ := json.Marshal(item)
		trace = append(trace, db.ToolEvent{Name: "skill_view", Arguments: string(args), Result: string(data)})
		recordSkillUsage(ctx, item, "selected")
	}
	return trace, nil
}
