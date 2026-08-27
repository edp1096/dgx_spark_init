package server

import (
	"errors"
	"strconv"
	"strings"
	"unicode"
)

func parseRenderedSubtitle(content, format, mode, targetLanguage string) ([]subtitleCue, error) {
	content = strings.ReplaceAll(content, "\r\n", "\n")
	content = strings.TrimSpace(content)
	if format == "vtt" {
		content = strings.TrimSpace(strings.TrimPrefix(content, "WEBVTT"))
	}
	blocks := strings.Split(content, "\n\n")
	cues := make([]subtitleCue, 0, len(blocks))
	for _, block := range blocks {
		lines := strings.Split(strings.TrimSpace(block), "\n")
		if len(lines) < 2 {
			continue
		}
		timingIndex := 0
		if !strings.Contains(lines[timingIndex], "-->") {
			timingIndex++
		}
		if timingIndex >= len(lines) || !strings.Contains(lines[timingIndex], "-->") {
			continue
		}
		parts := strings.SplitN(lines[timingIndex], "-->", 2)
		endFields := strings.Fields(parts[1])
		if len(endFields) == 0 {
			continue
		}
		start, startErr := parseSubtitleClock(strings.TrimSpace(parts[0]))
		end, endErr := parseSubtitleClock(endFields[0])
		if startErr != nil || endErr != nil || timingIndex+1 >= len(lines) {
			continue
		}
		bodyLines := trimSubtitleLines(lines[timingIndex+1:])
		cue := subtitleCue{Start: start, End: end}
		switch mode {
		case "translated":
			cue.Translated = strings.Join(bodyLines, "\n")
		case "bilingual":
			cue.Text, cue.Translated = splitBilingualSubtitle(bodyLines, targetLanguage)
		default:
			cue.Text = strings.Join(bodyLines, "\n")
		}
		cues = append(cues, cue)
	}
	if len(cues) == 0 {
		return nil, errors.New("no subtitle cues")
	}
	return cues, nil
}

func trimSubtitleLines(lines []string) []string {
	result := make([]string, 0, len(lines))
	for _, line := range lines {
		if line = strings.TrimSpace(line); line != "" {
			result = append(result, line)
		}
	}
	return result
}

func splitBilingualSubtitle(lines []string, targetLanguage string) (string, string) {
	if len(lines) < 2 {
		return strings.Join(lines, "\n"), ""
	}
	for split := 1; split < len(lines); split++ {
		left := strings.Join(lines[:split], "\n")
		right := strings.Join(lines[split:], "\n")
		if left == right {
			return left, right
		}
	}
	if strings.EqualFold(strings.TrimSpace(targetLanguage), "Korean") {
		for index := 1; index < len(lines); index++ {
			if containsHangulLine(lines[index]) {
				return strings.Join(lines[:index], "\n"), strings.Join(lines[index:], "\n")
			}
		}
	}
	middle := len(lines) / 2
	if middle < 1 {
		middle = 1
	}
	return strings.Join(lines[:middle], "\n"), strings.Join(lines[middle:], "\n")
}

func containsHangulLine(value string) bool {
	for _, character := range value {
		if unicode.Is(unicode.Hangul, character) {
			return true
		}
	}
	return false
}

func parseSubtitleClock(value string) (float64, error) {
	value = strings.ReplaceAll(strings.TrimSpace(value), ",", ".")
	parts := strings.Split(value, ":")
	if len(parts) != 3 {
		return 0, errors.New("invalid subtitle clock")
	}
	hours, err := strconv.Atoi(parts[0])
	if err != nil {
		return 0, err
	}
	minutes, err := strconv.Atoi(parts[1])
	if err != nil {
		return 0, err
	}
	seconds, err := strconv.ParseFloat(parts[2], 64)
	if err != nil {
		return 0, err
	}
	return float64(hours*3600+minutes*60) + seconds, nil
}
