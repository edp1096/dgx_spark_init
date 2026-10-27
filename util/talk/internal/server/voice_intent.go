package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"sparktalk/internal/llm"
	"strings"
	"time"
)

func (s *Server) routeVoiceIntent(ctx context.Context, session string, messages []llm.Message, client *llm.Client, model string) (*llm.ToolCall, error) {
	request := currentTitleRequestText(messages)
	if turn := turnFrom(ctx); turn != nil && turn.session == session {
		_, request = s.clipTurnIntent(ctx, session)
	}
	// A media URL is a source, not an instruction to create a video (e.g. /shorts/).
	lower := strings.ToLower(userURLPattern.ReplaceAllString(request, " "))
	if voiceHelpRequest(request) {
		return &llm.ToolCall{ID: fmt.Sprintf("voice_help_%x", time.Now().UnixNano()), Type: "function", Function: llm.FunctionCall{Name: "voice_tools", Arguments: `{"action":"help"}`}}, nil
	}
	video := strings.Contains(lower, "영상") || strings.Contains(lower, "쇼츠") || strings.Contains(lower, "video") || strings.Contains(lower, "shorts")
	voiceSpecific := false
	for _, word := range []string{"등록", "복제", "참조", "도움말", "어떻게", "음성만", "목소리만", "register", "clone", "help", "audio only", "세팅", "재설정", "교체", "replace", "reset"} {
		voiceSpecific = voiceSpecific || strings.Contains(lower, word)
	}
	if video && !voiceSpecific {
		return nil, nil
	}
	candidate := false
	for _, word := range []string{"음성", "목소리", "voice", "tts", "후보", "참조 구간"} {
		candidate = candidate || strings.Contains(lower, word)
	}
	if userURLPattern.MatchString(request) {
		for _, word := range []string{"세팅", "재설정", "교체", "바꿔", "변경", "replace", "reset"} {
			candidate = candidate || strings.Contains(lower, word)
		}
	}
	recent := []map[string]string{}
	for _, m := range messages[max(0, len(messages)-6):] {
		if (m.Role == "user" || m.Role == "assistant") && !m.ReferenceContext {
			text := strings.Join(userContentTexts(m.Content), "\n")
			recent = append(recent, map[string]string{"role": m.Role, "text": compactHistoryText(text, 1600)})
			if m.Role == "assistant" && (strings.Contains(text, "음성 작업") || strings.Contains(text, "참조 후보") || strings.Contains(text, "voice_") || strings.Contains(text, "저장된 Fish 미리 듣기")) {
				candidate = true
			}
		}
	}
	if !candidate {
		return nil, nil
	}
	caps, err := s.voiceCapabilities(ctx, session)
	if err != nil {
		return nil, err
	}
	sources, err := s.voiceSources(ctx, session)
	if err != nil {
		return nil, err
	}
	sourceSummary := []map[string]any{}
	for _, v := range sources[max(0, len(sources)-12):] {
		sourceSummary = append(sourceSummary, map[string]any{"id": v.Attachment.ID, "name": v.Attachment.Name, "mime": v.Attachment.MIME})
	}
	raw, code, err := s.shortsCall(ctx, "GET", "/v1/voice-jobs?session_id="+url.QueryEscape(session), nil)
	if err != nil {
		return nil, err
	}
	var jobs []voiceJob
	if code == 200 {
		if err = json.Unmarshal(raw, &jobs); err != nil {
			return nil, err
		}
	}
	if len(jobs) > 5 {
		jobs = jobs[:5]
	}
	data := map[string]any{"latest_user_request": request, "recent_dialogue": recent, "capabilities": caps, "sources": sourceSummary, "actual_jobs": jobs}
	prompt := `Return ONLY JSON {"action":"none|help|list|sources|inspect|prepare|generate|rename|delete|promote|status|cancel|retry|choose","evidence":"verbatim latest user quote","arguments":{}}. Route standalone voice generation and reference registration/management. Use help for how-to, capabilities, examples and questions about making/registering voices; these MUST NOT create work. Use none for video creation, applying voices to existing videos, cached preview/default selection, unrelated requests, quoted commands that the user is only discussing, and ambiguous instructions. prepare registers a reference from supplied attachment or URL: default scope persistent; use temporary only if requested. generate creates standalone speech with exact supplied text (do not invent dialogue), voice_id from catalog or provided source, scope temporary by default. Put overall tone direction in instructions, preserve inline tags and spoken wording in text. If the source reference is merely a source for a new video, return none. Resolve named voices and sources ONLY from supplied actual data. Prefer latest user upload when explicitly referenced; do not mistake generated preview/output audio for the original. Follow-up candidate selection/transcript correction uses choose with job_id and changes, zero-based candidate indices; for a completed reference use prepare with source_job_id and target_voice_id. status/cancel/retry require actual job_id (status may omit for list). Never create another job for status/retry. arguments may contain only action, voice_id, target_voice_id, name, scope, text, instructions, source_attachment_id, source_url, source_job_id, job_id, start, end, track, channel, candidate, transcript, changes. evidence must quote authorization in latest_user_request; none needs no evidence. Do not claim completion.`
	prompt += ` When replacing/reconfiguring an existing registered voice from a NEW URL or upload, use prepare with target_voice_id from the actual catalog and that NEW source_url or source_attachment_id. Preserve the existing voice name and scope. Do not use voice_id for prepare. Use source_job_id ONLY when reusing the previous source, never together with a new source. Requests such as "이걸로 다시 세팅해" can authorize replacing the named existing voice; resolve the name using the actual catalog and dialogue. For evidence, prefer the short exact action phrase; never paraphrase, add words or quote an earlier request.`
	routingMessages := []llm.Message{{Role: "system", Content: prompt}, {Role: "user", Content: shortsIntentJSON(data)}}
	var d voiceIntentDecision
	for attempt := 0; attempt < 2; attempt++ {
		result, err := client.Stream(llm.WithOutputLimit(ctx, 1800), routingMessages, model, "none", nil, func(string, string) error { return nil })
		if err != nil {
			return nil, err
		}
		text := strings.TrimSpace(result.Content)
		if strings.HasPrefix(text, "```") {
			text = strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(strings.TrimPrefix(text, "```json"), "```"), "```"))
		}
		d = voiceIntentDecision{}
		if err = json.Unmarshal([]byte(text), &d); err != nil {
			return nil, fmt.Errorf("음성 요청을 구조화하지 못했습니다: %w", err)
		}
		if d.Action == "none" {
			return nil, nil
		}
		if voiceIntentEvidenceMatches(request, d.Evidence) {
			break
		}
		if attempt == 1 {
			return nil, fmt.Errorf("음성 동작의 근거 문구를 원문과 대조하지 못했습니다. 요청을 실행하지 않았습니다")
		}
		routingMessages = append(routingMessages, llm.Message{Role: "assistant", Content: text}, llm.Message{Role: "user", Content: "The evidence is not a quote from latest_user_request. Return the complete decision again with a short exact contiguous quote from that request, without adding or changing words. Whitespace differences are allowed. If the latest request does not authorize the action, choose none. Do not use earlier dialogue as authorization."})
	}
	allowed := map[string]bool{"help": true, "list": true, "sources": true, "inspect": true, "prepare": true, "generate": true, "rename": true, "delete": true, "promote": true, "status": true, "cancel": true, "retry": true, "choose": true}
	if !allowed[d.Action] {
		return nil, fmt.Errorf("잘못된 음성 동작")
	}
	if d.Arguments == nil {
		d.Arguments = map[string]any{}
	}
	d.Arguments["action"] = d.Action
	return &llm.ToolCall{ID: fmt.Sprintf("voice_route_%x", time.Now().UnixNano()), Type: "function", Function: llm.FunctionCall{Name: "voice_tools", Arguments: shortsIntentJSON(d.Arguments)}}, nil
}

type voiceIntentDecision struct {
	Action    string         `json:"action"`
	Evidence  string         `json:"evidence"`
	Arguments map[string]any `json:"arguments"`
}

func voiceIntentEvidenceMatches(request, evidence string) bool {
	evidence = strings.Join(strings.Fields(evidence), " ")
	return evidence != "" && strings.Contains(strings.Join(strings.Fields(request), " "), evidence)
}

func voiceHelpRequest(text string) bool {
	lower := strings.ToLower(strings.TrimSpace(text))
	subject := strings.Contains(lower, "음성") || strings.Contains(lower, "목소리") || strings.Contains(lower, "tts") || strings.Contains(lower, "voice")
	if !subject {
		return false
	}
	for _, phrase := range []string{"도움말", "사용법", "생성 방법", "등록 방법", "만드는 방법", "어떻게 하", "어떻게 만", "어떻게 생", "어떻게 등", "how to", "how do i", "help"} {
		if strings.Contains(lower, phrase) {
			return true
		}
	}
	return false
}
