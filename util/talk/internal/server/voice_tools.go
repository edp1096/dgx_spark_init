package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"sparktalk/internal/llm"
	"strings"
	"time"
)

func voiceReferenceIntent(text string) bool {
	text = strings.ToLower(text)
	voice := strings.Contains(text, "음성") || strings.Contains(text, "목소리") || strings.Contains(text, "voice")
	action := false
	for _, word := range []string{"등록", "참조", "복제", "클론", "생성", "만들", "register", "clone", "reference", "generate"} {
		action = action || strings.Contains(text, word)
	}
	return voice && action
}

func (s *Server) registerVoiceTools(reg *completionToolRegistry) {
	cfg, _ := s.snapshot()
	if cfg.SupportDisabled("fish") {
		return
	}
	reg.prompts = append(reg.prompts, `Use voice_tools help for voice/audio creation how-to or capability questions: never create jobs for help, examples, quoted commands, or questions. voice_tools supports standalone WAV generation and reusable reference voices, without weight training. Register/prepare defaults to persistent across this local Talk library; use temporary only when requested or for one-off source-conditioned generation. Use exact source_attachment_id from visible media references, source_url for a user-provided media URL, or source_job_id to re-analyze this conversation's previous source. Never invent text to speak: preserve supplied dialogue verbatim including supported inline tags, put overall performance direction in instructions. Long reference sources are analyzed automatically for up to three contiguous candidates; do not transcribe entire reference uploads with attachment_read. A needs_input result asks for candidate, transcript, track/channel or range; use choose with changes. To replace an existing voice reference, use prepare with target_voice_id (never voice_id). Prefer the new source_url or source_attachment_id supplied by the user; use source_job_id only to reuse the previous source. Preserve the name and scope. Never send more than one source selector. status/cancel/retry use the actual job ID; never duplicate an existing job on a status request. Use shorts_voices for cached previews, default selection and applying a voice to an existing video. Never claim registration or generation succeeded before actual completed results.`)
	schema := json.RawMessage(`{"type":"object","properties":{"action":{"type":"string","enum":["help","list","sources","inspect","prepare","generate","rename","delete","promote","status","cancel","retry","choose"]},"voice_id":{"type":"string"},"target_voice_id":{"type":"string"},"name":{"type":"string"},"scope":{"type":"string","enum":["persistent","temporary"]},"text":{"type":"string","maxLength":500},"instructions":{"type":"string","maxLength":200},"source_attachment_id":{"type":"string"},"source_url":{"type":"string"},"source_job_id":{"type":"string"},"job_id":{"type":"string"},"start":{"type":"number"},"end":{"type":"number"},"track":{"type":"integer"},"channel":{"type":"integer"},"candidate":{"type":"integer"},"transcript":{"type":"string"},"changes":{"type":"object","description":"For choose/retry: candidate (zero-based), transcript, start/end seconds, track/channel indices, name or scope."}},"required":["action"],"additionalProperties":false}`)
	reg.register(llm.Tool{Type: "function", Function: llm.ToolFunction{Name: "voice_tools", Description: "Help, register/manage custom reference voices, or generate standalone speech WAV; durable progress and playable results in this conversation.", Parameters: schema}}, func(ctx context.Context, call llm.ToolCall, _ []llm.Message, emit eventEmitter) (registeredToolResult, error) {
		var a voiceRequest
		if err := json.Unmarshal([]byte(call.Function.Arguments), &a); err != nil {
			return registeredToolResult{}, err
		}
		a.SessionID = reg.sessionID
		if a.Action == "prepare" && a.Scope == "" {
			a.Scope = "persistent"
		}
		if a.Action == "prepare" || a.Action == "generate" {
			job, err := s.createVoiceJob(ctx, a)
			if err != nil {
				return registeredToolResult{}, err
			}
			return s.waitForVoice(ctx, call.ID, job, emit)
		}
		if err := s.ensureShorts(ctx); err != nil {
			return registeredToolResult{}, err
		}
		if a.Action == "sources" {
			items, err := s.voiceSources(ctx, a.SessionID)
			return registeredToolResult{Result: shortsIntentJSON(items)}, err
		}
		if a.Action == "status" || a.Action == "cancel" || a.Action == "retry" || a.Action == "choose" {
			if a.JobID == "" && a.Action == "status" {
				raw, code, err := s.shortsCall(ctx, "GET", "/v1/voice-jobs?session_id="+url.QueryEscape(a.SessionID), nil)
				if err == nil && code != 200 {
					err = fmt.Errorf("음성 작업 조회 HTTP %d", code)
				}
				return registeredToolResult{Result: string(raw)}, err
			}
			job, err := s.conversationVoiceJob(ctx, a.SessionID, a.JobID)
			if err != nil {
				return registeredToolResult{}, err
			}
			if a.Action != "status" {
				raw, code, err := s.shortsCall(ctx, "POST", "/v1/voice-jobs/"+job.ID+"/action", map[string]any{"session_id": a.SessionID, "action": a.Action, "changes": a.Changes})
				if err != nil {
					return registeredToolResult{}, err
				}
				if code != 200 {
					return registeredToolResult{}, fmt.Errorf("음성 작업 HTTP %d: %s", code, raw)
				}
				if err = json.Unmarshal(raw, &job); err != nil {
					return registeredToolResult{}, err
				}
			}
			if a.Action == "status" && (job.Status == "queued" || job.Status == "running") {
				return registeredToolResult{Result: shortsIntentJSON(job), FinalContent: fmt.Sprintf("음성 작업 %s: %s · %.0f%%", job.ID, job.Stage, job.Progress)}, nil
			}
			return s.waitForVoice(ctx, call.ID, job, emit)
		}
		switch a.Action {
		case "help", "list":
			caps, err := s.voiceCapabilities(ctx, a.SessionID)
			out := registeredToolResult{Result: shortsIntentJSON(caps)}
			if err == nil && a.Action == "help" {
				out.FinalContent = voiceHelpText(caps)
			}
			return out, err
		case "inspect", "rename", "delete", "promote":
			raw, code, err := s.shortsCall(ctx, "POST", "/v1/voice-tools", map[string]any{"session_id": a.SessionID, "action": a.Action, "voice_id": a.VoiceID, "name": a.Name})
			if err == nil && code != 200 {
				err = fmt.Errorf("음성 관리 HTTP %d: %s", code, raw)
			}
			return registeredToolResult{Result: string(raw)}, err
		default:
			return registeredToolResult{}, fmt.Errorf("지원하지 않는 음성 동작")
		}
	})
}

func (s *Server) waitForVoice(ctx context.Context, callID string, job voiceJob, emit eventEmitter) (registeredToolResult, error) {
	defer func() {
		if errors.Is(context.Cause(ctx), errChatStopped) && (job.Status == "queued" || job.Status == "running") {
			stopCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			_, _, _ = s.shortsCall(stopCtx, "POST", "/v1/voice-jobs/"+job.ID+"/action", map[string]any{"session_id": job.SessionID, "action": "cancel", "changes": map[string]any{}})
		}
	}()
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		if emit != nil {
			step, total := int(job.Progress), 100
			if err := emit("tool_output", map[string]any{"id": callID, "stream": "stdout", "progress": generationProgress{Kind: "voice", Stage: job.Status, Detail: job.Stage, Step: &step, Total: &total, Elapsed: max(0, float64(time.Now().UnixMilli())/1000-job.Created)}}); err != nil {
				return registeredToolResult{}, err
			}
		}
		if job.Status != "queued" && job.Status != "running" {
			message, err := s.publishVoice(ctx, job.SessionID, job.ID)
			if err != nil {
				return registeredToolResult{Result: shortsIntentJSON(job)}, err
			}
			return registeredToolResult{Result: shortsIntentJSON(map[string]any{"job": job, "published_message_id": message.ID}), FinalContent: "음성 작업 결과를 이 대화에 게시했습니다."}, nil
		}
		if turn := turnFrom(ctx); turn != nil && turn.hasPending() {
			return registeredToolResult{Result: shortsIntentJSON(job)}, nil
		}
		select {
		case <-ctx.Done():
			return registeredToolResult{}, ctx.Err()
		case <-ticker.C:
		}
		next, err := s.conversationVoiceJob(ctx, job.SessionID, job.ID)
		if err != nil {
			return registeredToolResult{Result: shortsIntentJSON(job)}, err
		}
		job = next
	}
}

func voiceHelpText(caps map[string]any) string {
	formats := []string{}
	if items, ok := caps["input_formats"].([]any); ok {
		for _, item := range items {
			if name, ok := item.(string); ok {
				formats = append(formats, strings.ToUpper(name))
			}
		}
	}
	return "음성 파일 또는 음성이 포함된 영상·미디어 URL을 보내고 다음처럼 요청하세요.\n\n" +
		"- 영구 등록: ‘이 파일의 목소리를 내 목소리라는 이름으로 등록해.’ 등록한 음성은 다른 대화에서도 쓸 수 있습니다.\n" +
		"- 일회용 참조: ‘이 목소리를 이번 대화에서만 임시로 써.’ 임시 참조는 완료 후 24시간 보관하며, 원본 첨부와 생성 결과는 대화에 남습니다.\n" +
		"- 음성 생성: ‘내 목소리로 다음 대사를 차분하게 읽어줘: 오늘도 좋은 하루 보내세요.’ 결과는 WAV 파일입니다. 대사는 500자, 전체 말투 지시는 200자까지입니다.\n" +
		"- 참조 수정: 후보 번호, 시작·끝 시각, 채널 또는 정확한 전사문을 알려 주세요. 이름 변경·삭제·영구 등록 전환도 요청할 수 있습니다.\n" +
		"- 기존 영상 변경: ‘기존 영상에 내 목소리를 적용해.’ 기존 영상을 보존하고 새 버전을 만듭니다.\n\n" +
		"긴 음원에서는 최대 15초의 연속 발화 구간을 분석합니다. 발화가 1초 미만이면 추가 음원이 필요하고, 5초 미만이면 품질 경고가 표시됩니다. 화자 구분 정보를 얻지 못한 경우에는 그 사실을 표시합니다.\n\n" +
		"입력 형식: " + strings.Join(formats, ", ") + ". 실제 디코딩 가능 여부를 확인합니다. 태그가 들어간 합성 입력은 발화 160자 이내여야 하며, 감정·구간 지시 준수나 목소리 유사도를 보장하지 않습니다. 모델 가중치를 학습하지 않고 참조 음성을 사용합니다.\n\n도움말 조회로 생성 작업을 시작하지 않았습니다."
}

func (s *Server) cancelDeletedSessionVoices(session string) {
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		// No service startup for deletion. A missed callback is retried by the worker
		// when publication finds the conversation gone.
		_, _, _ = s.shortsCall(ctx, "POST", "/v1/voice-sessions/"+url.PathEscape(session)+"/cancel", nil)
	}()
}
