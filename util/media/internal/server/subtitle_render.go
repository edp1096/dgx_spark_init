package server

import (
	"fmt"
	"os"
	"strings"
)

func (s *Server) writeSubtitleOutputs(id string, segments []subtitleCue, formats []string, translationMode string) (map[string]string, error) {
	outputs := map[string]string{}
	for _, format := range formats {
		var content string
		switch format {
		case "srt":
			content = renderSRT(segments, translationMode)
		case "vtt":
			content = renderVTT(segments, translationMode)
		case "timestamped_txt":
			content = renderTimestampedText(segments, translationMode)
		case "txt":
			content = renderPlainText(segments, translationMode)
		}
		name := id + "." + map[string]string{"timestamped_txt": "timestamps.txt"}[format]
		if format != "timestamped_txt" {
			name = id + "." + format
		}
		if err := os.WriteFile(s.jobs.OutputPath(name), []byte(content+"\n"), 0o644); err != nil {
			return nil, err
		}
		outputs[format] = "/api/outputs/" + name
	}
	return outputs, nil
}

func segmentText(segment subtitleCue, mode string) string {
	switch mode {
	case "translated":
		return segment.Translated
	case "bilingual":
		return segment.Text + "\n" + segment.Translated
	default:
		return segment.Text
	}
}

func renderPlainText(segments []subtitleCue, mode string) string {
	values := make([]string, 0, len(segments))
	for _, segment := range segments {
		if value := strings.TrimSpace(segmentText(segment, mode)); value != "" {
			values = append(values, value)
		}
	}
	return strings.Join(values, "\n")
}

func renderTimestampedText(segments []subtitleCue, mode string) string {
	var output strings.Builder
	for _, segment := range segments {
		fmt.Fprintf(&output, "[%s --> %s] %s\n", formatClock(segment.Start, '.'), formatClock(segment.End, '.'), segmentText(segment, mode))
	}
	return strings.TrimSpace(output.String())
}

func renderSRT(segments []subtitleCue, mode string) string {
	var output strings.Builder
	for index, segment := range segments {
		fmt.Fprintf(&output, "%d\n%s --> %s\n%s\n\n", index+1, formatClock(segment.Start, ','), formatClock(segment.End, ','), segmentText(segment, mode))
	}
	return strings.TrimSpace(output.String())
}

func renderVTT(segments []subtitleCue, mode string) string {
	var output strings.Builder
	output.WriteString("WEBVTT\n\n")
	for _, segment := range segments {
		fmt.Fprintf(&output, "%s --> %s\n%s\n\n", formatClock(segment.Start, '.'), formatClock(segment.End, '.'), segmentText(segment, mode))
	}
	return strings.TrimSpace(output.String())
}

func formatClock(seconds float64, separator rune) string {
	if seconds < 0 {
		seconds = 0
	}
	milliseconds := int64(seconds*1000 + 0.5)
	hours := milliseconds / 3600000
	milliseconds %= 3600000
	minutes := milliseconds / 60000
	milliseconds %= 60000
	secs := milliseconds / 1000
	millis := milliseconds % 1000
	return fmt.Sprintf("%02d:%02d:%02d%c%03d", hours, minutes, secs, separator, millis)
}
