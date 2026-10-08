package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
)

var errInvalidSubtitleTranslation = errors.New("invalid subtitle translation")

type subtitleTranslationWarning struct {
	Segment int    `json:"segment"`
	Source  string `json:"source"`
	Reason  string `json:"reason"`
}

func (s *Server) translateSubtitleSegments(segments []subtitleCue, targetLanguage string, progress func(done, total int), cancelled func() bool) ([]subtitleTranslationWarning, error) {
	cfg := s.config()
	warnings := make([]subtitleTranslationWarning, 0)
	total := (len(segments) + 7) / 8
	done := 0
	for start := 0; start < len(segments); start += 8 {
		if cancelled != nil && cancelled() {
			return warnings, errJobCancelled
		}
		end := start + 8
		if end > len(segments) {
			end = len(segments)
		}
		var input strings.Builder
		for index := start; index < end; index++ {
			fmt.Fprintf(&input, "[[%04d]] %s\n", index, segments[index].Text)
		}
		systemPrompt := "You translate subtitle segments. Translate only the text into " + targetLanguage + ". Preserve every [[NNNN]] marker exactly once and in order. Do not add explanations."
		if strings.EqualFold(targetLanguage, "Korean") {
			systemPrompt = "당신은 전문 영상 자막 번역가입니다. 각 [[NNNN]] 표식을 그대로 유지하면서 뒤의 자막을 자연스러운 한국어로 번역하세요. 일본어·영어 원문을 복사하지 말고 설명 없이 번역문만 출력하세요."
		}
		payload := map[string]any{
			"model": cfg.PromptEnhancement.Model,
			"messages": []map[string]string{
				{"role": "system", "content": systemPrompt},
				{"role": "user", "content": input.String()},
			},
			"max_completion_tokens": 2048, "temperature": 0, "top_k": 1, "reasoning_effort": "none",
		}
		data, err := s.chatWithPromptEngine(payload)
		if cancelled != nil && cancelled() {
			return warnings, errJobCancelled
		}
		if err != nil {
			return warnings, err
		}
		var response struct {
			Choices []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(data, &response); err != nil || len(response.Choices) == 0 {
			return warnings, fmt.Errorf("translation engine returned an invalid response")
		}
		translated := parseMarkedTranslations(response.Choices[0].Message.Content)
		for index := start; index < end; index++ {
			value, ok := translated[index]
			if !ok || !validSubtitleTranslation(segments[index].Text, value, targetLanguage) {
				value, err = s.retrySubtitleTranslation(segments[index].Text, targetLanguage)
				if err != nil {
					if !errors.Is(err, errInvalidSubtitleTranslation) {
						return warnings, fmt.Errorf("segment %d: %w", index+1, err)
					}
					warnings = append(warnings, subtitleTranslationWarning{
						Segment: index + 1,
						Source:  strings.TrimSpace(segments[index].Text),
						Reason:  "한국어 번역 결과가 없어 원문을 유지했습니다.",
					})
					value = segments[index].Text
				}
			}
			segments[index].Translated = strings.TrimSpace(value)
		}
		done++
		if progress != nil {
			progress(done, total)
		}
	}
	return warnings, nil
}

func validSubtitleTranslation(source, translated, targetLanguage string) bool {
	source = strings.TrimSpace(source)
	translated = strings.TrimSpace(translated)
	if translated == "" {
		return false
	}
	if strings.EqualFold(source, translated) && !strings.EqualFold(targetLanguage, "Korean") {
		return false
	}
	if strings.EqualFold(targetLanguage, "Korean") && !containsHangul(source) {
		return containsHangul(translated)
	}
	return true
}

func containsHangul(value string) bool {
	for _, char := range value {
		if (char >= 0x1100 && char <= 0x11ff) || (char >= 0x3130 && char <= 0x318f) || (char >= 0xac00 && char <= 0xd7af) {
			return true
		}
	}
	return false
}

func (s *Server) retrySubtitleTranslation(source, targetLanguage string) (string, error) {
	cfg := s.config()
	systemPrompt := "You are a professional audiovisual subtitle translator. Translate the input into natural " + targetLanguage + ". Return exactly one translated subtitle and nothing else. Never copy untranslated source text."
	if strings.EqualFold(targetLanguage, "Korean") {
		systemPrompt = "당신은 전문 영상 자막 번역가입니다. 입력 자막을 자연스러운 한국어 자막 한 줄로 번역하세요. 원문을 복사하지 말고 번역문만 출력하세요."
	}
	payload := map[string]any{
		"model": cfg.PromptEnhancement.Model,
		"messages": []map[string]string{
			{"role": "system", "content": systemPrompt},
			{"role": "user", "content": source},
		},
		"max_completion_tokens": 512, "temperature": 0, "top_k": 1, "reasoning_effort": "none",
	}
	data, err := s.chatWithPromptEngine(payload)
	if err != nil {
		return "", err
	}
	var response struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(data, &response); err != nil || len(response.Choices) == 0 {
		return "", fmt.Errorf("translation engine returned an invalid retry response")
	}
	translated := strings.TrimSpace(response.Choices[0].Message.Content)
	if !validSubtitleTranslation(source, translated, targetLanguage) {
		return "", fmt.Errorf("%w: translation engine did not produce %s text", errInvalidSubtitleTranslation, targetLanguage)
	}
	return translated, nil
}

func parseMarkedTranslations(value string) map[int]string {
	result := map[int]string{}
	current := -1
	for _, line := range strings.Split(strings.TrimSpace(value), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "[[") {
			if closeIndex := strings.Index(line, "]]"); closeIndex >= 2 {
				if index, err := strconv.Atoi(line[2:closeIndex]); err == nil {
					current = index
					result[current] = strings.TrimSpace(line[closeIndex+2:])
					continue
				}
			}
		}
		if current >= 0 && line != "" {
			result[current] = strings.TrimSpace(result[current] + " " + line)
		}
	}
	return result
}
