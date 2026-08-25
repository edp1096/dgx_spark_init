package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

const assistantSystemPrompt = `You are the embedded Korean AI operator for Spark Media, a local media creation panel.
Answer naturally and completely in Korean. Be concise when appropriate, but never omit the requested explanation, list, options, or creative guidance. You can prepare the visible controls, navigate the app, and offer a confirmation button for an operation. Never claim that generation started or completed: execution happens only after the user presses the confirmation button.

Return one JSON object only, with this schema:
{
  "reply": "Korean response",
  "actions": [
    {"type":"navigate","tab":"image|video|speech|recognition|lora|history|settings"},
    {"type":"set_image","prompt":"...","width":1024,"height":1024,"seed":-1,"enhance_enabled":true},
    {"type":"set_video","prompt":"...","width":768,"height":512,"fps":24,"duration":5,"seed":-1,"enhance_enabled":true},
    {"type":"set_speech","text":"...","instructions":"...","language":"Korean","speaker":"Sohee","seed":-1},
    {"type":"set_recognition","context":"...","language":"Auto","translation_mode":"none|translated|bilingual","target_language":"Korean"},
    {"type":"set_module","module":"identity|depth|style|userLora|vision|styleReference|nk2e|anypaint","enabled":true,"preset":"restage|sheet|faceSwap|headSwap|personSwap|tryon|replace|"},
    {"type":"set_recent_image","image_index":1,"target":"identity|identityReference|depth|nk2e|anypaint|vision|styleReference"},
    {"type":"set_outpaint","image_index":1,"outpaint_left":64,"outpaint_top":0,"outpaint_right":64,"outpaint_bottom":0},
    {"type":"open_modules"},
    {"type":"show_results","tab":"image|video|speech|recognition"}
  ],
  "confirmation": "image|video|speech|recognition|"
}

Rules:
- Use only the listed action types and fields.
- Treat brainstorming, scene-description help, prompt critique, explanations, recommendations, and questions as conversation. Answer them fully in reply and return an empty actions array. Do not silently convert advice or a requested list into form values.
- Use actions only when the latest user message explicitly asks to apply, set, change, open, navigate, select, prepare, create, generate, speak, transcribe, or otherwise operate Spark Media. An earlier operation does not authorize actions for a later advice-only message.
- When actions are requested, set the controls and explain what changed.
- "Help me describe it", "what should I describe?", and similar requests mean the user wants creative guidance from you. Give concrete, useful categories, options, or follow-up questions in Korean; do not return set_image or set_video.
- Never end after merely announcing a list, such as "주요 항목은 다음과 같습니다:". Put every requested item in the reply itself. For example, a scene-description checklist should return a reply like "- 주체: 인물·사물과 행동\n- 장소: 공간과 시대\n- 구도: 시점과 배치\n- 조명: 방향과 시간대\n- 분위기: 감정과 색감", with actions set to an empty array.
- Use confirmation only when the user explicitly asks to create, generate, speak, or start recognition.
- Image and video prompts should be useful English generation prompts unless the user requests otherwise.
- When labeled video conditioning frames are attached and the user asks for a prompt, inspect every frame and provide a concrete LTX motion/transition prompt now. Also return set_video with that English prompt so it is applied to the video form. Never answer with generic advice such as "enter a prompt" or ask the user to describe images that are attached.
- The latest user request is authoritative. A new creation request replaces the previous visual concept by default.
- Never carry colors, weather, time of day, mood, subjects, or style from an older prompt or the current UI prompt unless the latest user explicitly refers to it with wording such as "keep it", "same as before", "continue", or "change only".
- For a short or broad request, stay visually neutral and do not invent a dominant palette or extreme atmosphere. In particular, a night view is after dark; never add sunset, red sky, crimson lighting, fire, dystopia, or apocalyptic styling unless explicitly requested.
- Do not invent an uploaded file or reference image. If one is required, navigate and tell the user to select it.
- Numbered recent images are listed in Current UI state. Use set_recent_image only for an index present there.
- Recent image indices start at 1. Never invent index 0, an index absent from recent_images, or a prompt not copied exactly from recent_images.
- If no recent_images entry matches the user's description by its saved prompt, say that you cannot identify it from the available prompt metadata and ask the user for an index.
- For "replace the face in image A with the face from image B", set A as target identity, B as identityReference, and enable module identity with preset faceSwap.
- For extending image canvas edges, use set_outpaint only. Never use nk2e, depth, identity, or set_image. Pure outpaint needs no prompt.
- Recognition cannot execute without an uploaded file or URL; prepare its settings and ask the user to provide the source.
- Keep numeric values practical: image 256-2048, video 256-2048, fps 1-60, duration 1-30.
- The current UI state is appended below for control awareness, not as creative prompt material. Preserve existing input only for explicit adjustment requests; replace it for a new creation request.`

const assistantGroundingReminder = `

CRITICAL RECENT-IMAGE GROUNDING:
- You receive text metadata only, never image pixels.
- If asked whether an image is visible or what it depicts, begin exactly with: "이미지 자체는 볼 수 없지만, 저장 프롬프트 기준으로"
- Do not say "확인했습니다", "보입니다", "담고 있습니다", or otherwise imply visual inspection.
- Quote only indices and prompt facts present in recent_images. Indices start at 1. If metadata is insufficient, say so.
- Answer the metadata question completely in the same reply. Never say that you will check later. List every matching index and its saved prompt facts now.
- When the user only asks a question about image metadata, return an empty actions array and do not navigate anywhere.`

const assistantVisionReminder = `

CRITICAL RECENT-IMAGE VISION GROUNDING:
- The latest user message includes one contact sheet made from actual recent image pixels. Each tile has a visible #index badge.
- Inspect the contact sheet itself. The badge numbers correspond exactly to recent_images; indices start at 1.
- Examine every numbered tile one by one before answering and list every matching index; do not stop after the first similar matches.
- Cross-check visual findings against saved prompt metadata, but let actual pixels decide when they conflict.
- Begin visual findings exactly with: "연락처 시트를 직접 확인한 결과"
- Distinguish actual visual findings from saved prompt metadata. Never invent an index absent from recent_images.
- Answer completely now. For a visual question only, return an empty actions array and do not navigate.`

const assistantVideoVisionReminder = `

CRITICAL VIDEO-CONDITIONING VISION:
- The latest user message includes one contact sheet made from the actual images currently selected in the video tab.
- Every tile is labeled START, KEYFRAME n, or END with its timeline position. Inspect every labeled tile and respect their chronological order.
- START and END appearance are already fixed by conditioning. Write an LTX prompt describing coherent subject motion, camera motion, environmental motion, continuity, and the transition needed to connect them; do not merely restate the two still images.
- If the user asks what prompt would be good, give one immediately usable English prompt and return a set_video action containing exactly that prompt. Preserve the existing duration, resolution, FPS, seed, and selected images by omitting those fields.
- Do not reply with generic guidance, tell the user to enter a prompt, or ask what the selected frames contain. You can see them in the attached contact sheet.
- Briefly explain in Korean what motion and transition you chose.`

type assistantChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type assistantChatRequest struct {
	Messages      []assistantChatMessage  `json:"messages"`
	State         map[string]any          `json:"state"`
	VisualContext *assistantVisualContext `json:"visual_context,omitempty"`
}

type assistantVisualContext struct {
	Kind     string   `json:"kind"`
	ImageURL string   `json:"image_url"`
	Labels   []string `json:"labels,omitempty"`
}

type assistantAction struct {
	Type            string  `json:"type"`
	Tab             string  `json:"tab,omitempty"`
	Prompt          string  `json:"prompt,omitempty"`
	Text            string  `json:"text,omitempty"`
	Instructions    string  `json:"instructions,omitempty"`
	Context         string  `json:"context,omitempty"`
	Language        string  `json:"language,omitempty"`
	Speaker         string  `json:"speaker,omitempty"`
	TargetLanguage  string  `json:"target_language,omitempty"`
	TranslationMode string  `json:"translation_mode,omitempty"`
	Width           int     `json:"width,omitempty"`
	Height          int     `json:"height,omitempty"`
	Seed            *int64  `json:"seed,omitempty"`
	FPS             float64 `json:"fps,omitempty"`
	Duration        float64 `json:"duration,omitempty"`
	EnhanceEnabled  *bool   `json:"enhance_enabled,omitempty"`
	Module          string  `json:"module,omitempty"`
	Preset          string  `json:"preset,omitempty"`
	Enabled         *bool   `json:"enabled,omitempty"`
	ImageIndex      int     `json:"image_index,omitempty"`
	Target          string  `json:"target,omitempty"`
	OutpaintLeft    int     `json:"outpaint_left,omitempty"`
	OutpaintTop     int     `json:"outpaint_top,omitempty"`
	OutpaintRight   int     `json:"outpaint_right,omitempty"`
	OutpaintBottom  int     `json:"outpaint_bottom,omitempty"`
}

type assistantChatResponse struct {
	Reply        string            `json:"reply"`
	Actions      []assistantAction `json:"actions"`
	Confirmation string            `json:"confirmation,omitempty"`
	VisionUsed   bool              `json:"vision_used,omitempty"`
}

func (s *Server) assistantChat(w http.ResponseWriter, r *http.Request) {
	var request assistantChatRequest
	decoder := json.NewDecoder(io.LimitReader(r.Body, 4<<20))
	if err := decoder.Decode(&request); err != nil {
		http.Error(w, "invalid assistant request", http.StatusBadRequest)
		return
	}
	if len(request.Messages) == 0 {
		http.Error(w, "at least one message is required", http.StatusBadRequest)
		return
	}
	if len(request.Messages) > 24 {
		request.Messages = request.Messages[len(request.Messages)-24:]
	}
	stateJSON, _ := json.Marshal(request.State)
	visionDataURL := ""
	visionUsed := false
	grounding := assistantGroundingReminder
	if context := request.VisualContext; context != nil && context.Kind == "video_conditioning" && strings.HasPrefix(context.ImageURL, "data:image/jpeg;base64,") && len(context.ImageURL) <= 3<<20 {
		visionDataURL = context.ImageURL
		visionUsed = true
		grounding = assistantVideoVisionReminder + fmt.Sprintf("\n- Attached timeline labels: %v", context.Labels)
	} else {
		recentVisionURL, visionIndices, visionErr := s.assistantContactSheet(request)
		if visionErr == nil && recentVisionURL != "" {
			visionDataURL = recentVisionURL
			visionUsed = true
			grounding = assistantVisionReminder + fmt.Sprintf("\n- Contact sheet indices: %v", visionIndices)
		}
	}
	messages := []map[string]any{{"role": "system", "content": assistantSystemPrompt + "\n\nCurrent UI state:\n" + string(stateJSON) + grounding}}
	lastUsableUser := -1
	for index, message := range request.Messages {
		if strings.EqualFold(strings.TrimSpace(message.Role), "user") && strings.TrimSpace(message.Content) != "" {
			lastUsableUser = index
		}
	}
	for messageIndex, message := range request.Messages {
		role := strings.ToLower(strings.TrimSpace(message.Role))
		content := strings.TrimSpace(message.Content)
		if (role != "user" && role != "assistant") || content == "" {
			continue
		}
		if len([]rune(content)) > 4000 {
			content = string([]rune(content)[:4000])
		}
		messageContent := any(content)
		if visionUsed && role == "user" && messageIndex == lastUsableUser {
			visionInstruction := "번호표가 붙은 최근 이미지 연락처 시트를 실제 픽셀 기준으로 확인하세요."
			if request.VisualContext != nil && request.VisualContext.Kind == "video_conditioning" {
				visionInstruction = "START·KEYFRAME·END 라벨이 붙은 현재 영상 조건 이미지를 실제 픽셀 기준으로 모두 확인하고 시간 순서대로 연결하세요."
			}
			messageContent = []map[string]any{
				{"type": "image_url", "image_url": map[string]string{"url": visionDataURL}},
				{"type": "text", "text": content + "\n\n" + visionInstruction},
			}
		}
		messages = append(messages, map[string]any{"role": role, "content": messageContent})
	}
	if len(messages) == 1 {
		http.Error(w, "no usable assistant messages", http.StatusBadRequest)
		return
	}
	cfg := s.config()
	payload := map[string]any{
		"model":                 cfg.PromptEnhancement.Model,
		"messages":              messages,
		"max_completion_tokens": 900,
		"temperature":           0,
		"top_k":                 1,
		"seed":                  42,
		"reasoning_effort":      "none",
		"response_format":       map[string]any{"type": "json_object"},
	}
	data, _, err := s.callJSON(strings.TrimRight(cfg.Engines["prompt"].Endpoint, "/")+"/v1/chat/completions", payload)
	if err != nil {
		http.Error(w, "assistant: "+err.Error(), http.StatusBadGateway)
		return
	}
	var completion struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if json.Unmarshal(data, &completion) != nil || len(completion.Choices) == 0 {
		http.Error(w, "assistant returned an invalid response", http.StatusBadGateway)
		return
	}
	result, err := decodeAssistantResponse(completion.Choices[0].Message.Content)
	if err != nil {
		http.Error(w, "assistant returned invalid controls: "+err.Error(), http.StatusBadGateway)
		return
	}
	result = sanitizeAssistantResponse(result)
	result = normalizeAssistantOutpaint(result, request)
	result = normalizeAssistantVideoPromptAdvice(result, request)
	result.VisionUsed = visionUsed
	writeJSON(w, http.StatusOK, result)
}

func normalizeAssistantVideoPromptAdvice(result assistantChatResponse, request assistantChatRequest) assistantChatResponse {
	if request.VisualContext == nil || request.VisualContext.Kind != "video_conditioning" {
		return result
	}
	latest := strings.ToLower(latestAssistantUserMessage(request.Messages))
	if !strings.Contains(latest, "프롬프트") && !strings.Contains(latest, "prompt") {
		return result
	}
	for index, action := range result.Actions {
		if action.Type != "set_video" || strings.TrimSpace(action.Prompt) == "" {
			continue
		}
		prompt := strings.TrimSpace(action.Prompt)
		// Prompt advice must not silently reset unrelated controls that the user
		// already configured in the video panel.
		result.Actions[index] = assistantAction{Type: "set_video", Prompt: prompt}
		if !strings.Contains(result.Reply, prompt) {
			result.Reply = strings.TrimSpace(result.Reply) + "\n\n추천 프롬프트\n" + prompt
		}
		return result
	}
	return result
}

func decodeAssistantResponse(content string) (assistantChatResponse, error) {
	content = strings.TrimSpace(content)
	content = strings.TrimPrefix(content, "```json")
	content = strings.TrimPrefix(content, "```")
	content = strings.TrimSuffix(strings.TrimSpace(content), "```")
	start, end := strings.Index(content, "{"), strings.LastIndex(content, "}")
	if start < 0 || end < start {
		return assistantChatResponse{}, fmt.Errorf("JSON object not found")
	}
	var result assistantChatResponse
	decoder := json.NewDecoder(bytes.NewBufferString(content[start : end+1]))
	if err := decoder.Decode(&result); err != nil {
		return assistantChatResponse{}, err
	}
	if strings.TrimSpace(result.Reply) == "" {
		return assistantChatResponse{}, fmt.Errorf("reply is empty")
	}
	return result, nil
}

func sanitizeAssistantResponse(result assistantChatResponse) assistantChatResponse {
	result.Reply = strings.TrimSpace(result.Reply)
	allowedTabs := map[string]bool{"image": true, "video": true, "speech": true, "recognition": true, "lora": true, "history": true, "settings": true}
	allowedTypes := map[string]bool{"navigate": true, "set_image": true, "set_video": true, "set_speech": true, "set_recognition": true, "set_module": true, "set_recent_image": true, "set_outpaint": true, "open_modules": true, "show_results": true}
	allowedModules := map[string]bool{"identity": true, "depth": true, "style": true, "userLora": true, "vision": true, "styleReference": true, "nk2e": true, "anypaint": true}
	allowedPresets := map[string]bool{"": true, "restage": true, "sheet": true, "faceSwap": true, "headSwap": true, "personSwap": true, "tryon": true, "replace": true}
	allowedImageTargets := map[string]bool{"identity": true, "identityReference": true, "depth": true, "nk2e": true, "anypaint": true, "vision": true, "styleReference": true}
	allowedTranslationModes := map[string]bool{"": true, "none": true, "translated": true, "bilingual": true}
	clean := make([]assistantAction, 0, len(result.Actions))
	for _, action := range result.Actions {
		if !allowedTypes[action.Type] || (action.Tab != "" && !allowedTabs[action.Tab]) {
			continue
		}
		if action.Type == "set_module" && (!allowedModules[action.Module] || !allowedPresets[action.Preset]) {
			continue
		}
		if action.Type == "set_recent_image" && (action.ImageIndex < 1 || !allowedImageTargets[action.Target]) {
			continue
		}
		if action.Type == "set_outpaint" && action.ImageIndex < 1 {
			continue
		}
		if !allowedTranslationModes[action.TranslationMode] {
			action.TranslationMode = ""
		}
		action.Width = clampInt(action.Width, 0, 2048)
		action.Height = clampInt(action.Height, 0, 2048)
		action.OutpaintLeft = clampInt(action.OutpaintLeft, 0, 1024)
		action.OutpaintTop = clampInt(action.OutpaintTop, 0, 1024)
		action.OutpaintRight = clampInt(action.OutpaintRight, 0, 1024)
		action.OutpaintBottom = clampInt(action.OutpaintBottom, 0, 1024)
		if action.FPS < 0 || action.FPS > 60 {
			action.FPS = 0
		}
		if action.Duration < 0 || action.Duration > 30 {
			action.Duration = 0
		}
		clean = append(clean, action)
		if len(clean) == 8 {
			break
		}
	}
	result.Actions = clean
	if result.Confirmation != "image" && result.Confirmation != "video" && result.Confirmation != "speech" && result.Confirmation != "recognition" {
		result.Confirmation = ""
	}
	return result
}

func clampInt(value, minimum, maximum int) int {
	if value < minimum {
		return minimum
	}
	if value > maximum {
		return maximum
	}
	return value
}
