package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

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
