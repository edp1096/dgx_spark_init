package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"sparktalk/internal/db"
	"sparktalk/internal/llm"
)

const shortsStudioPrompt = `Short-video studio tools are available. Use shorts_generate for a new finished video. For redesigns of an existing video, use shorts_project revise to reuse its media and create a new version; preview_only=true lets you inspect actual frames before shorts_project render. The local model authors the template's CSS, HTML/SVG and GSAP; there is no forced card theme. shorts_project capabilities documents the schemas and available features. shorts_assets searches/imports verified free LottieFiles animations, lists assets, and creates original Lottie JSON/SVG. Use it when animation assets help, not just as decoration on every project. shorts_templates creates real template code or saves a project's authored design for reuse; descriptions alone are not template implementations. All tools operate in this conversation. Preserve prior versions. Use source_job_id/template_id/asset_ids when appropriate. Media reuse and design reuse are separate. design_mode defaults to redesign: it clears inherited template selection and runs the designer. Set preserve only for an explicit request to keep the current visual design or apply a saved template unchanged; never preserve a design the user criticized. Request the actual duration via duration_seconds. Use shorts_manage for status/wait/cancel/retry, never create duplicates to check progress. Keep interaction in this chat, with no dedicated panel or modal. A completed result includes applied, verification and limitations: describe ONLY those measured/applied facts, not unverified wishes from the original brief. Do not claim piano recordings, voice identity, exact word alignment, colors, layout or scene counts absent from the applied report. Preview frames show visuals only, not audio quality. A failed requirement must be reported and fixed, never silently replaced by the old template. No universal '2026 standard' exists in these tools; treat web guides as design references. Native video_generate produces a single short H3 clip, while shorts_generate assembles a multi-scene short.`

func (s *Server) registerShortsStudio(reg *completionToolRegistry) {
	reg.prompts = append(reg.prompts, shortsStudioPrompt)
	for _, tool := range []struct{ name, path, description, schema string }{
		{"shorts_assets", "/v1/assets", "Search free commercial-use Lottie animations, import a verified source page, create original Lottie JSON/SVG, or list/inspect this conversation's assets. Preserve source and license metadata. Import expects a public lottiefiles.com/free-animation page; do not invent URLs. create expects actual vector JSON or SVG markup, not a description.", `{"type":"object","properties":{"action":{"type":"string","enum":["search","import","create","list","inspect"]},"query":{"type":"string"},"source_url":{"type":"string"},"asset_id":{"type":"string"},"kind":{"type":"string","enum":["lottie","svg"]},"name":{"type":"string"},"content":{"type":["object","string"]}},"required":["action"],"additionalProperties":false}`},
		{"shorts_templates", "/v1/templates", "Create model-written template CSS/GSAP/HTML, save an existing project's actual template, or list/inspect saved templates in this conversation. Use shorts_project capabilities for the authoring contract. Templates are code, not fixed visual presets. Apply an ID with shorts_generate or shorts_project revise.", `{"type":"object","properties":{"action":{"type":"string","enum":["create","save","list","inspect"]},"name":{"type":"string"},"template_id":{"type":"string"},"job_id":{"type":"string"},"artwork":{"type":"object","properties":{"name":{"type":"string"},"css":{"type":"string"},"javascript":{"type":"string"},"overlays_html":{"type":"string"},"highlights":{"type":"object"},"notes":{"type":"string"},"limitations":{"type":"array","items":{"type":"string"}}},"required":["css"],"additionalProperties":false}},"required":["action"],"additionalProperties":false}`},
		{"shorts_project", "/v1/project-tools", "Inspect production capabilities, list/inspect jobs and real applied settings, draft a preview, revise an existing video using its assets, or render a checked preview. Revision creates a new version and preserves the original. preview/revise can return preview frames in chat. Use brief for change instructions, optional plan for a complete structured production plan, and template_id/asset_ids for reusable resources. Omit job_id for the latest job.", `{"type":"object","properties":{"action":{"type":"string","enum":["capabilities","list","inspect","draft","revise","preview","render"]},"job_id":{"type":"string"},"brief":{"type":"string","maxLength":12000},"title":{"type":"string"},"duration_seconds":{"type":"integer","minimum":15,"maximum":120},"preview_only":{"type":"boolean"},"template_id":{"type":"string"},"design_mode":{"type":"string","enum":["redesign","preserve"]},"asset_ids":{"type":"array","items":{"type":"string"}},"plan":{"type":"object"}},"required":["action"],"additionalProperties":false}`},
	} {
		reg.register(llm.Tool{Type: "function", Function: llm.ToolFunction{Name: tool.name, Description: tool.description, Parameters: json.RawMessage(tool.schema)}}, func(ctx context.Context, call llm.ToolCall, _ []llm.Message, emit eventEmitter) (registeredToolResult, error) {
			var args map[string]any
			if len(call.Function.Arguments) > 200000 || json.Unmarshal([]byte(call.Function.Arguments), &args) != nil || args == nil {
				return registeredToolResult{}, fmt.Errorf("잘못된 제작 도구 입력")
			}
			args["session_id"] = reg.sessionID
			if err := s.ensureShorts(ctx); err != nil {
				return registeredToolResult{}, err
			}
			action, _ := args["action"].(string)
			if (tool.name == "shorts_project" && action != "capabilities" && action != "list" && action != "draft") || (tool.name == "shorts_templates" && action == "save") {
				id, _ := args["job_id"].(string)
				job, err := s.conversationShortsJob(ctx, reg.sessionID, id)
				if err != nil {
					return registeredToolResult{}, err
				}
				args["job_id"] = job.ID
			}
			raw, code, err := s.shortsCall(ctx, "POST", tool.path, args)
			if err != nil {
				return registeredToolResult{}, err
			}
			if code != 200 {
				return registeredToolResult{}, fmt.Errorf("제작 도구 HTTP %d: %s", code, raw)
			}
			if tool.name == "shorts_project" && (action == "draft" || action == "revise" || action == "preview" || action == "render") {
				var job shortsJob
				if json.Unmarshal(raw, &job) != nil || !validShortsJobID(job.ID) || job.SessionID != reg.sessionID {
					return registeredToolResult{}, fmt.Errorf("작업 응답이 잘못됐습니다")
				}
				return s.waitForShorts(ctx, reg.sessionID, call.ID, job, emit)
			}
			return registeredToolResult{Result: string(raw)}, nil
		})
	}
}

func (s *Server) shortsPreviewAttachment(ctx context.Context, job shortsJob) (*db.Attachment, error) {
	marker := "/api/shorts/" + job.ID + "/files/preview-sheet.jpg?session_id=" + url.QueryEscape(job.SessionID)
	messages, err := s.db.Messages(job.SessionID)
	if err != nil {
		return nil, err
	}
	for _, message := range messages {
		for _, attachment := range message.Attachments {
			if attachment.SourceURL == marker {
				return &attachment, nil
			}
		}
	}
	req, err := http.NewRequestWithContext(ctx, "GET", s.shortsEndpoint()+"/v1/projects/"+job.ID+"/files/preview-sheet.jpg", nil)
	if err != nil {
		return nil, err
	}
	response, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != 200 || !strings.HasPrefix(response.Header.Get("Content-Type"), "image/jpeg") {
		return nil, fmt.Errorf("미리보기 이미지를 읽지 못했습니다")
	}
	attachment, err := s.media.SaveReader(io.LimitReader(response.Body, 8<<20), "Shorts-preview.jpg", "image/jpeg", 8<<20)
	if err != nil {
		return nil, err
	}
	attachment.SourceURL = marker
	return &attachment, nil
}
