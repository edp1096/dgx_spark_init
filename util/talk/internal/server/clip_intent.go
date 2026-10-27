package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"time"

	"sparktalk/internal/llm"
)

// This is a cheap eligibility check, not an intent decision. The local model
// resolves output type, negation, references and context in a separate small
// request, before a long chat history can be mistaken for a fresh execution.
func clipIntentCandidate(text string) bool {
	text = strings.ToLower(text)
	if strings.Contains(text, "쇼츠") || strings.Contains(text, "shorts") {
		return false
	}
	media, action := false, false
	for _, word := range []string{"영상", "동영상", "video", "animate", "움직", "흔들", "애니메이션"} {
		media = media || strings.Contains(text, word)
	}
	for _, word := range []string{"만들", "바꿔", "바꾸", "움직", "흔들", "다시", "이어", "재개", "중지", "취소", "진행", "상태", "어디까지", "make", "create", "turn", "animate", "retry", "resume", "stop", "cancel", "status"} {
		action = action || strings.Contains(text, word)
	}
	if media && action {
		return true
	}
	for _, word := range []string{"새 버전", "새버전", "이어서", "재개", "중지", "취소", "다시", "계속", "new version", "resume", "cancel", "retry", "again", "continue"} {
		if strings.Contains(text, word) {
			return true
		}
	}
	return false
}

func clipContextCandidate(messages []llm.Message) bool {
	for _, message := range messages[max(0, len(messages)-5):] {
		if message.Role != "assistant" || message.ReferenceContext {
			continue
		}
		for _, call := range message.ToolCalls {
			if call.Function.Name == "clip_produce" || call.Function.Name == "clip_manage" {
				return true
			}
		}
		text := strings.Join(userContentTexts(message.Content), "\n")
		if strings.Contains(text, "검수한 영상을") || strings.Contains(text, "완성본을 첨부하지") ||
			(strings.Contains(text, "사진") && strings.Contains(text, "영상")) {
			return true
		}
	}
	return false
}

func (s *Server) routeClipIntent(ctx context.Context, session string, messages []llm.Message, client *llm.Client, model string) (*llm.ToolCall, error) {
	request := currentTitleRequestText(messages)
	if turn := turnFrom(ctx); turn != nil && turn.session == session {
		_, request = s.clipTurnIntent(ctx, session)
	}
	focused := clipContextCandidate(messages)
	if !clipIntentCandidate(request) && !focused {
		return nil, nil
	}
	images, err := s.sessionImageAttachmentsForContext(ctx, session)
	if err != nil {
		return nil, err
	}
	ids := make([]string, 0, len(images))
	for id := range images {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	available := []map[string]string{}
	for _, id := range ids {
		available = append(available, map[string]string{"id": id, "name": images[id].Name})
	}
	recent := []map[string]string{}
	for _, message := range messages[max(0, len(messages)-8):] {
		if (message.Role == "user" || message.Role == "assistant") && !message.ReferenceContext {
			text := strings.Join(userContentTexts(message.Content), "\n")
			if text != "" {
				recent = append(recent, map[string]string{"role": message.Role, "text": compactHistoryText(text, 1800)})
			}
		}
	}
	jobs := []map[string]any{}
	readCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	raw, code, readErr := s.shortsCall(readCtx, "GET", "/v1/clips?session_id="+url.QueryEscape(session), nil)
	cancel()
	if readErr == nil && code == 200 {
		var values []struct {
			ID      string `json:"id"`
			Status  string `json:"status"`
			Project struct {
				SourceImageID string `json:"source_image_id"`
			} `json:"project"`
		}
		if json.Unmarshal(raw, &values) == nil {
			for _, job := range values[:min(3, len(values))] {
				jobs = append(jobs, map[string]any{"id": job.ID, "status": job.Status, "source_image_id": job.Project.SourceImageID})
			}
		}
	}
	if len(images) == 0 && len(jobs) == 0 {
		return nil, nil
	}
	explicitVideo := false
	for _, word := range []string{"영상", "동영상", "video", "animate", "움직", "흔들", "애니메이션"} {
		explicitVideo = explicitVideo || strings.Contains(strings.ToLower(request), word)
	}
	if !explicitVideo && len(jobs) == 0 && !focused {
		return nil, nil
	}
	data, _ := json.Marshal(map[string]any{"latest_user_request": request, "recent_dialogue": recent, "available_images": available, "actual_photo_video_jobs": jobs})
	prompt := `You are the photo-to-video intent router, not a conversational assistant. Return ONLY JSON with keys action, source_image_id, job_id, brief, resolution, evidence. action is one of none, produce, status, cancel, retry. Read the latest user's actual request, including negation, and use recent dialogue only to resolve references. Choose produce for a request to animate an attached photo or to edit/restyle it then animate it. A request for a NEW VERSION must choose produce, even when an earlier job failed; old assistant claims are not a new execution. Use the original source image ID from the actual job when the user says same original photo. Choose retry only to continue the same stopped/interrupted brief; status for a progress question; cancel for stopping this photo-video job. Choose none for image-only work, explanations/how-to/capability questions, existing-video restyling, face/person swaps, unrelated work, narrated/multi-scene/long-duration shorts, or ambiguity about whether execution is wanted. An image-only request mentioning "do not make a video" is none. Do not confuse a mention of an old failure with a request to repeat it. evidence must quote the exact words in latest_user_request that authorize the chosen action; empty for none. source_image_id must be an available ID for produce. job_id must be a supplied job ID or empty for latest. brief preserves the actual request and relevant earlier creative constraints; never invent a request to mute audio, remove an action or change the subject. resolution is 480p by default or 720p if requested. Do not invent IDs, outcomes, permissions or questions. Output JSON, no markdown.`
	result, err := client.Stream(llm.WithOutputLimit(ctx, 1800), []llm.Message{{Role: "system", Content: prompt}, {Role: "user", Content: string(data)}}, model, "none", nil, func(string, string) error { return nil })
	if err != nil {
		return nil, err
	}
	text := strings.TrimSpace(result.Content)
	if strings.HasPrefix(text, "```") {
		text = strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(strings.TrimPrefix(text, "```json"), "```"), "```"))
	}
	var decision struct {
		Action        string `json:"action"`
		SourceImageID string `json:"source_image_id"`
		JobID         string `json:"job_id"`
		Brief         string `json:"brief"`
		Resolution    string `json:"resolution"`
		Evidence      string `json:"evidence"`
	}
	decoder := json.NewDecoder(strings.NewReader(text))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&decision) != nil {
		return nil, fmt.Errorf("사진 영상 요청을 구조화하지 못했습니다. 작업은 시작하지 않았습니다")
	}
	if decision.Action == "none" {
		return nil, nil
	}
	if strings.TrimSpace(decision.Evidence) == "" || !strings.Contains(request, decision.Evidence) {
		return nil, fmt.Errorf("실제 사용자 요청과 제작 동작이 맞지 않습니다")
	}
	call := &llm.ToolCall{ID: "clip_route_" + strconv.FormatInt(time.Now().UnixNano(), 16), Type: "function"}
	var arguments any
	switch decision.Action {
	case "produce":
		if decision.SourceImageID == "" && len(ids) == 1 {
			decision.SourceImageID = ids[0]
		}
		if _, ok := images[decision.SourceImageID]; !ok {
			return nil, fmt.Errorf("제작할 원본 사진이 이 대화에 없습니다")
		}
		if decision.Resolution == "" {
			decision.Resolution = "480p"
		}
		if strings.TrimSpace(decision.Brief) == "" {
			decision.Brief = request
		}
		if decision.Resolution != "480p" && decision.Resolution != "720p" || len(decision.Brief) > 12000 {
			return nil, fmt.Errorf("잘못된 사진 영상 제작 조건입니다")
		}
		call.Function.Name = "clip_produce"
		arguments = clipCreate{SourceImageID: decision.SourceImageID, Brief: decision.Brief, Resolution: decision.Resolution}
	case "status", "cancel", "retry":
		if decision.JobID != "" {
			found := false
			for _, job := range jobs {
				found = found || job["id"] == decision.JobID
			}
			if !found {
				return nil, fmt.Errorf("확인되지 않은 제작 작업 ID입니다")
			}
		}
		call.Function.Name = "clip_manage"
		arguments = map[string]string{"action": decision.Action, "job_id": decision.JobID}
	default:
		return nil, fmt.Errorf("지원하지 않는 제작 의도입니다")
	}
	encoded, _ := json.Marshal(arguments)
	call.Function.Arguments = string(encoded)
	return call, nil
}
