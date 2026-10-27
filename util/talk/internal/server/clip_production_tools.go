package server

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"sparktalk/internal/llm"
)

const clipProductionPrompt = `For a request to animate a photo or to restyle/edit a photo and then animate it, use clip_produce. Pass the original user brief with all constraints and an actual available source_image_id. This durable tool interprets the scene, edits with the real source image, inspects full images and hand/face crops, repairs defects, generates a 5.17-second H3 clip, and inspects timestamped and adjacent frames before delivery. Prefer it for photo-to-video requests over manually chaining image_generate/video_generate, which do not provide this durable review/repair workflow. Do not ask the user for technical coordinates, prompts, seeds or node settings. Infer reasonable artistic choices from the image and request; ask only when the subject/source is genuinely ambiguous. Default 480p; use 720p when requested. It does not transform existing videos, implement face/person swaps, or produce arbitrary durations. Face/person swaps are on hold; do not route them to this tool. Cartoon stylization of the same subject is supported. Image-only edits still use image_generate; text-only single clips use video_generate; narrated multi-scene shorts use shorts_generate. Use clip_manage for status, wait, cancel and retry, not duplicate creation. A failed inspection is not success. Never claim perfect hands, identity, exhaustive temporal inspection or audio review absent from the actual report. Completed output is already attached. Keep the interaction in chat without asking the user to operate ComfyUI.`

func (s *Server) registerClipProduction(ctx context.Context, reg *completionToolRegistry) {
	if s == nil || s.db == nil || s.media == nil || reg.sessionID == "" {
		return
	}
	cfg, _ := s.snapshot()
	if _, ok := videoRuntime(cfg); !ok || cfg.Runtime.Mode != "managed" {
		return
	}
	reg.prompts = append(reg.prompts, clipProductionPrompt+"\n"+imageAttachmentCatalogForContext(ctx, s, reg.sessionID))
	reg.prompts = append(reg.prompts, `Route the user's intended output and action precisely. "사진을 만화로 바꾸고 움직이게" means clip_produce. "만화 사진만, 영상은 만들지 마" means image_generate with identity_edit, not a video tool or shorts_project capabilities. For an existing photo-to-video job, "중지해/취소해/그만해" means clip_manage action=cancel, not status; "어디까지 됐어/진행 상황" means status; "실패한 것 다시 시도/이어서 해" means retry. A direct stop request must actually stop the job, not merely inspect it.`)
	reg.prompts = append(reg.prompts, `Use retry only to resume the existing brief. If the user changes style, motion or preserved elements, create a new version with clip_produce using the original source image and the revised brief. Read the previous job's report for concrete defects before revising. Never turn a cancellation or exhausted repair budget into another production without a new user instruction.`)
	reg.register(llm.Tool{Type: "function", Function: llm.ToolFunction{Name: "clip_produce", Description: "Create a reviewed 5.17-second video from a photo in this conversation. Interprets the brief, optionally edits/restyles the photo, checks and repairs hands/faces/composition, animates the accepted image, checks video frames and retries defects. Durable across reconnects; attaches only the final accepted video. Use for 'make this photo cartoon and animate it' and ordinary photo animation. No existing-video restyling or face swaps.", Parameters: json.RawMessage(`{"type":"object","properties":{"source_image_id":{"type":"string","minLength":1},"brief":{"type":"string","minLength":3,"maxLength":12000,"description":"Complete original user request, retaining desired style, motion, preserved elements, audio and constraints."},"resolution":{"type":"string","enum":["480p","720p"]},"seed":{"type":"integer","minimum":0,"maximum":9223372036854775791}},"required":["source_image_id","brief"],"additionalProperties":false}`)}}, func(ctx context.Context, call llm.ToolCall, _ []llm.Message, emit eventEmitter) (registeredToolResult, error) {
		var args clipCreate
		decoder := json.NewDecoder(strings.NewReader(call.Function.Arguments))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&args); err != nil {
			return registeredToolResult{}, fmt.Errorf("잘못된 사진 영상 제작 요청")
		}
		job, err := s.createClip(ctx, reg.sessionID, args)
		if err != nil {
			return registeredToolResult{}, err
		}
		return s.waitForClip(ctx, reg.sessionID, call.ID, job, emit)
	})
	reg.register(llm.Tool{Type: "function", Function: llm.ToolFunction{Name: "clip_manage", Description: "Inspect, wait for, cancel, or retry this conversation's photo-to-video production. Omit job_id for the latest production. Retry resumes saved steps without resetting exhausted quality-repair budgets; it is useful after interruption or temporary service failure.", Parameters: json.RawMessage(`{"type":"object","properties":{"action":{"type":"string","enum":["status","wait","cancel","retry"],"description":"status reads progress; wait monitors completion; cancel actually stops an active production (중지/취소); retry resumes failed/interrupted work."},"job_id":{"type":"string","pattern":"^[a-f0-9]{32}$"}},"required":["action"],"additionalProperties":false}`)}}, func(ctx context.Context, call llm.ToolCall, _ []llm.Message, emit eventEmitter) (registeredToolResult, error) {
		var args struct {
			Action string `json:"action"`
			JobID  string `json:"job_id"`
		}
		decoder := json.NewDecoder(strings.NewReader(call.Function.Arguments))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&args); err != nil || (args.Action != "status" && args.Action != "wait" && args.Action != "cancel" && args.Action != "retry") {
			return registeredToolResult{}, fmt.Errorf("잘못된 제작 관리 요청")
		}
		if err := s.ensureShorts(ctx); err != nil {
			return registeredToolResult{}, err
		}
		job, err := s.conversationClip(ctx, reg.sessionID, args.JobID)
		if err != nil {
			return registeredToolResult{}, err
		}
		if args.Action == "cancel" || args.Action == "retry" {
			if args.Action == "retry" && job.Status == "canceled" {
				requested, err := s.clipResumeRequested(ctx, job)
				if err != nil {
					return registeredToolResult{}, err
				}
				if !requested {
					return registeredToolResult{}, fmt.Errorf("사용자가 중지한 제작입니다. 새로운 재개 요청 없이 다시 실행하지 마세요")
				}
			}
			var payload any
			if args.Action == "cancel" {
				key, _ := s.clipTurnIntent(ctx, reg.sessionID)
				payload = map[string]string{"cancel_turn_key": key}
			}
			raw, code, err := s.shortsCall(ctx, "POST", "/v1/clips/"+job.ID+"/"+args.Action, payload)
			if err != nil {
				return registeredToolResult{}, err
			}
			if code != 200 || json.Unmarshal(raw, &job) != nil || job.SessionID != reg.sessionID {
				return registeredToolResult{}, fmt.Errorf("제작 상태 변경 실패: HTTP %d", code)
			}
		}
		if args.Action == "wait" || args.Action == "retry" || args.Action == "cancel" {
			return s.waitForClip(ctx, reg.sessionID, call.ID, job, emit)
		}
		return s.clipResult(ctx, job)
	})
}
