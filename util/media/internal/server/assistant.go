package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

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
	data, err := s.chatWithPromptEngine(payload)
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
