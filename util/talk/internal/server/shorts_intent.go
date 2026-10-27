package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"sparktalk/internal/llm"
	"strconv"
	"strings"
	"time"
)

func shortsIntentCandidate(text string) bool {
	text = strings.ToLower(text)
	action := false
	for _, word := range []string{"만들", "제작", "생성", "적용", "바꿔", "변경", "create", "make", "apply", "change"} {
		action = action || strings.Contains(text, word)
	}
	narrated := strings.Contains(text, "쇼츠") || strings.Contains(text, "shorts") || strings.Contains(text, "내레이션") || strings.Contains(text, "narration")
	voice := strings.Contains(text, "음성") || strings.Contains(text, "목소리") || strings.Contains(text, "voice")
	video := strings.Contains(text, "영상") || strings.Contains(text, "video")
	return action && (narrated || (voice && video))
}

func (s *Server) routeShortsIntent(ctx context.Context, session string, messages []llm.Message, client *llm.Client, model string) (*llm.ToolCall, error) {
	request := currentTitleRequestText(messages)
	if turn := turnFrom(ctx); turn != nil && turn.session == session {
		_, request = s.clipTurnIntent(ctx, session)
	}
	recent := []map[string]string{}
	voiceContext := false
	for _, message := range messages[max(0, len(messages)-6):] {
		if (message.Role == "user" || message.Role == "assistant") && !message.ReferenceContext {
			text := strings.Join(userContentTexts(message.Content), "\n")
			recent = append(recent, map[string]string{"role": message.Role, "text": compactHistoryText(text, 1200)})
			if message.Role == "assistant" && (strings.Contains(text, "Fish 미리 듣기") || strings.Contains(text, "기본 음성을")) {
				voiceContext = true
			}
		}
	}
	followup := voiceContext && (strings.Contains(request, "적용") || strings.Contains(request, "바꿔") || strings.Contains(strings.ToLower(request), "apply"))
	if !shortsIntentCandidate(request) && !followup {
		return nil, nil
	}
	if err := s.ensureShorts(ctx); err != nil {
		return nil, err
	}
	// Resolve live job IDs and voice choices, never assistant claims or recalled conversations.
	data := map[string]any{"latest_user_request": request, "recent_dialogue": recent}
	voices := map[string]bool{}
	readCtx, cancel := context.WithTimeout(ctx, 3*time.Second)
	raw, code, _ := s.shortsCall(readCtx, "POST", "/v1/voices", map[string]string{"session_id": session, "action": "list"})
	cancel()
	if code == 200 {
		var v struct {
			Voices []struct {
				ID   string `json:"id"`
				Name string `json:"name"`
			} `json:"voices"`
			Selected string `json:"selected_voice"`
		}
		if json.Unmarshal(raw, &v) == nil {
			data["voices"] = v.Voices
			data["selected_voice"] = v.Selected
			for _, x := range v.Voices {
				voices[x.ID] = true
			}
		}
	}
	jobs := map[string]bool{}
	readCtx, cancel = context.WithTimeout(ctx, 3*time.Second)
	raw, code, _ = s.shortsCall(readCtx, "GET", "/v1/projects?session_id="+url.QueryEscape(session), nil)
	cancel()
	if code == 200 {
		var values []shortsJob
		if json.Unmarshal(raw, &values) == nil {
			recent := []map[string]string{}
			for _, v := range values[:min(3, len(values))] {
				jobs[v.ID] = true
				recent = append(recent, map[string]string{"id": v.ID, "status": v.Status, "title": v.Title})
			}
			data["actual_jobs"] = recent
		}
	}
	prompt := `You are the narrated-video execution intent router. Return ONLY JSON with action, title, brief, duration_seconds, bgm, voice_id, job_id, evidence. Actions: none, create, apply_voice. Use create only when the latest user actually requests a NEW narrated short/multi-scene video. For create, write a concise nonempty title even if the user did not specify one. Preserve all creative constraints in brief; include the entire current request. A prior assistant saying 'finished' is not proof of execution. Use apply_voice only when the user asks to change the voice of an EXISTING video while keeping its content/design. Use none for voice list/preview/default-selection questions, information/how-to/planning/capability questions, requests NOT to create a video, video status/cancel/retry, image-only work, photo animation, visual redesign, or unrelated/ambiguous requests. Voice listing or choosing a future default does not authorize video creation. Resolve this voice/그 음성 from the recent preview or selection dialogue. Previewing a voice does not change the saved default: when the user says selected default, use selected_voice; when referring to the just-previewed voice, use that voice ID. voice_id must be an installed ID, or empty to use the saved selection. job_id must be an actual supplied job ID or empty for latest; never invent one. bgm is true only if the user requests background music, otherwise false. duration_seconds is the explicitly requested duration (15..120) or 60 for a new short if unspecified. evidence must be a verbatim quote from latest_user_request authorizing the action. Empty evidence for none. Do not generate a conversational completion claim.`
	result, err := client.Stream(llm.WithOutputLimit(ctx, 1400), []llm.Message{{Role: "system", Content: prompt}, {Role: "user", Content: shortsIntentJSON(data)}}, model, "none", nil, func(string, string) error { return nil })
	if err != nil {
		return nil, err
	}
	text := strings.TrimSpace(result.Content)
	if strings.HasPrefix(text, "```") {
		text = strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(strings.TrimPrefix(text, "```json"), "```"), "```"))
	}
	var decision struct {
		Action   string `json:"action"`
		Title    string `json:"title"`
		Brief    string `json:"brief"`
		Duration int    `json:"duration_seconds"`
		BGM      bool   `json:"bgm"`
		VoiceID  string `json:"voice_id"`
		JobID    string `json:"job_id"`
		Evidence string `json:"evidence"`
	}
	decoder := json.NewDecoder(strings.NewReader(text))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&decision) != nil {
		return nil, fmt.Errorf("영상 제작 요청을 구조화하지 못했습니다. 작업은 시작하지 않았습니다")
	}
	if decision.Action == "none" {
		return nil, nil
	}
	if strings.TrimSpace(decision.Evidence) == "" || !strings.Contains(request, decision.Evidence) {
		return nil, fmt.Errorf("실제 요청과 영상 제작 동작이 맞지 않습니다")
	}
	if decision.VoiceID != "" && !voices[decision.VoiceID] {
		return nil, fmt.Errorf("등록되지 않은 음성입니다")
	}
	call := &llm.ToolCall{ID: "shorts_route_" + strconv.FormatInt(time.Now().UnixNano(), 16), Type: "function"}
	var arguments any
	switch decision.Action {
	case "create":
		if strings.TrimSpace(decision.Title) == "" {
			decision.Title = "내레이션 영상"
		}
		if decision.Duration < 15 || decision.Duration > 120 || strings.TrimSpace(decision.Title) == "" {
			return nil, fmt.Errorf("잘못된 영상 제작 조건입니다: duration=%d title=%q", decision.Duration, decision.Title)
		}
		// The exact user request remains authoritative even if the router paraphrases it.
		brief := request
		if len(brief) > 12000 {
			return nil, fmt.Errorf("제작 지시가 너무 깁니다")
		}
		call.Function.Name = "shorts_generate"
		arguments = map[string]any{"title": decision.Title, "brief": brief, "duration_seconds": decision.Duration, "voice_preset": decision.VoiceID, "bgm": decision.BGM}
	case "apply_voice":
		if decision.JobID != "" && !jobs[decision.JobID] {
			return nil, fmt.Errorf("이 대화의 영상 작업이 아닙니다")
		}
		call.Function.Name = "shorts_voices"
		arguments = map[string]string{"action": "apply", "voice_id": decision.VoiceID, "job_id": decision.JobID}
	default:
		return nil, fmt.Errorf("지원하지 않는 제작 동작입니다")
	}
	call.Function.Arguments = shortsIntentJSON(arguments)
	return call, nil
}

func shortsIntentJSON(value any) string { raw, _ := json.Marshal(value); return string(raw) }
