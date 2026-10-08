package server

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

var (
	assistantNamedImagePattern = regexp.MustCompile(`(?i)이미지\s*#?\s*(\d+)`)
	assistantOutpaintPatterns  = map[string]*regexp.Regexp{
		"horizontal": regexp.MustCompile(`(?i)좌우(?:를|로)?\s*(\d+)\s*(?:px|픽셀)`),
		"vertical":   regexp.MustCompile(`(?i)상하(?:를|로)?\s*(\d+)\s*(?:px|픽셀)`),
		"left":       regexp.MustCompile(`(?i)왼쪽(?:을|으로)?\s*(\d+)\s*(?:px|픽셀)`),
		"top":        regexp.MustCompile(`(?i)위쪽(?:을|으로)?\s*(\d+)\s*(?:px|픽셀)`),
		"right":      regexp.MustCompile(`(?i)오른쪽(?:을|으로)?\s*(\d+)\s*(?:px|픽셀)`),
		"bottom":     regexp.MustCompile(`(?i)아래쪽(?:을|으로)?\s*(\d+)\s*(?:px|픽셀)`),
	}
)

func normalizeAssistantOutpaint(result assistantChatResponse, request assistantChatRequest) assistantChatResponse {
	content := latestAssistantUserMessage(request.Messages)
	lower := strings.ToLower(content)
	if !strings.Contains(lower, "늘리") && !strings.Contains(lower, "확장") && !strings.Contains(lower, "outpaint") && !strings.Contains(lower, "아웃페인트") {
		return result
	}
	index := assistantOutpaintImageIndex(content)
	if index < 1 || !assistantRecentImageExists(request.State, index) {
		return result
	}
	values := map[string]int{}
	for direction, pattern := range assistantOutpaintPatterns {
		match := pattern.FindStringSubmatch(content)
		if len(match) != 2 {
			continue
		}
		value, err := strconv.Atoi(match[1])
		if err == nil && value > 0 {
			values[direction] = clampInt(value, 0, 1024)
		}
	}
	left, top, right, bottom := values["left"], values["top"], values["right"], values["bottom"]
	if values["horizontal"] > 0 {
		left, right = values["horizontal"], values["horizontal"]
	}
	if values["vertical"] > 0 {
		top, bottom = values["vertical"], values["vertical"]
	}
	if left+top+right+bottom == 0 {
		return result
	}
	action := assistantAction{Type: "set_outpaint", ImageIndex: index, OutpaintLeft: left, OutpaintTop: top, OutpaintRight: right, OutpaintBottom: bottom}
	result.Actions = []assistantAction{action}
	result.Confirmation = "image"
	result.Reply = fmt.Sprintf("%d번 이미지를 %s 확장하도록 준비했습니다. 확장만 하는 작업이라 프롬프트 없이 원본을 자연스럽게 이어갑니다.", index, assistantOutpaintDirectionLabel(action))
	return result
}

func assistantOutpaintImageIndex(content string) int {
	match := assistantNamedImagePattern.FindStringSubmatch(content)
	if len(match) != 2 {
		match = assistantImageIndexPattern.FindStringSubmatch(content)
	}
	if len(match) != 2 {
		return 0
	}
	value, _ := strconv.Atoi(match[1])
	return value
}

func assistantRecentImageExists(state map[string]any, index int) bool {
	images, err := decodeAssistantRecentImages(state["recent_images"])
	if err != nil {
		return false
	}
	for _, image := range images {
		if image.Index == index && image.JobID != "" && image.Status == "completed" {
			return true
		}
	}
	return false
}

func assistantOutpaintDirectionLabel(action assistantAction) string {
	parts := make([]string, 0, 4)
	if action.OutpaintLeft == action.OutpaintRight && action.OutpaintLeft > 0 && action.OutpaintTop == 0 && action.OutpaintBottom == 0 {
		return fmt.Sprintf("좌우 각각 %dpx", action.OutpaintLeft)
	}
	if action.OutpaintTop == action.OutpaintBottom && action.OutpaintTop > 0 && action.OutpaintLeft == 0 && action.OutpaintRight == 0 {
		return fmt.Sprintf("상하 각각 %dpx", action.OutpaintTop)
	}
	for _, item := range []struct {
		name  string
		value int
	}{{"왼쪽", action.OutpaintLeft}, {"위쪽", action.OutpaintTop}, {"오른쪽", action.OutpaintRight}, {"아래쪽", action.OutpaintBottom}} {
		if item.value > 0 {
			parts = append(parts, fmt.Sprintf("%s %dpx", item.name, item.value))
		}
	}
	return strings.Join(parts, " · ")
}
