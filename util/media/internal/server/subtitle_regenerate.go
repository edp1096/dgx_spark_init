package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"unicode"

	"mediaapp/internal/jobs"
)

type subtitleRegenerateRequest struct {
	TranslationMode string   `json:"translation_mode"`
	OutputFormats   []string `json:"output_formats"`
}

// regenerateSubtitle only rebuilds display files from already recognized cues.
// It deliberately does not download media, run ASR, or translate again.
func (s *Server) regenerateSubtitle(w http.ResponseWriter, r *http.Request) {
	job, ok := s.jobs.Get(r.PathValue("id"))
	if !ok {
		http.NotFound(w, r)
		return
	}
	if job.Kind != "recognition" || job.Status != "completed" {
		http.Error(w, "완료된 받아쓰기 결과만 자막을 재생성할 수 있습니다", http.StatusConflict)
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, 64<<10)
	var request subtitleRegenerateRequest
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, "invalid request", http.StatusBadRequest)
		return
	}
	if request.TranslationMode != "none" && request.TranslationMode != "translated" && request.TranslationMode != "bilingual" {
		http.Error(w, "invalid translation_mode", http.StatusBadRequest)
		return
	}
	formats, err := validateSubtitleOutputFormats(request.OutputFormats)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	cues, migrated, err := s.loadSubtitleCueArchive(job)
	if err != nil {
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}
	if err := validateSubtitleCueMode(cues, request.TranslationMode); err != nil {
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}
	if migrated {
		if err := s.writeSubtitleCueArchive(job.ID, cues); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}

	outputs, err := s.writeSubtitleOutputs(job.ID, cues, formats, request.TranslationMode)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if job.MediaAssetID != "" || job.CaptionURL != "" {
		captionName := job.ID + ".player.vtt"
		if err := os.WriteFile(s.jobs.OutputPath(captionName), []byte(renderVTT(cues, request.TranslationMode)+"\n"), 0o644); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		job.CaptionURL = "/api/outputs/" + captionName
	}

	for kind, outputURL := range job.Outputs {
		if _, keep := outputs[kind]; keep {
			continue
		}
		name := filepath.Base(outputURL)
		if strings.HasPrefix(name, job.ID+".") {
			_ = os.Remove(s.jobs.OutputPath(name))
		}
	}
	preview := renderPlainText(cues, request.TranslationMode)
	if len([]rune(preview)) > 4000 {
		preview = string([]rune(preview)[:4000]) + "…"
	}
	if job.Params == nil {
		job.Params = map[string]any{}
	}
	job.Params["translation_mode"] = request.TranslationMode
	job.Params["output_formats"] = formats
	job.Params["text"] = preview
	job.Params["cues"] = len(cues)
	job.Outputs = outputs
	job.OutputURL = ""
	if output, exists := outputs["txt"]; exists {
		job.OutputURL = output
	} else {
		for _, format := range formats {
			if output, exists := outputs[format]; exists {
				job.OutputURL = output
				break
			}
		}
	}
	if err := s.jobs.Save(job); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusOK, job)
}

func validateSubtitleOutputFormats(values []string) ([]string, error) {
	allowed := map[string]bool{"srt": true, "vtt": true, "timestamped_txt": true, "txt": true}
	seen := map[string]bool{}
	formats := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if !allowed[value] {
			return nil, fmt.Errorf("unsupported output format: %s", value)
		}
		if !seen[value] {
			seen[value] = true
			formats = append(formats, value)
		}
	}
	if len(formats) == 0 {
		return nil, errors.New("결과 형식을 하나 이상 선택하세요")
	}
	return formats, nil
}

func validateSubtitleCueMode(cues []subtitleCue, mode string) error {
	if len(cues) == 0 {
		return errors.New("재생성할 자막 큐가 없습니다")
	}
	for _, cue := range cues {
		if (mode == "none" || mode == "bilingual") && strings.TrimSpace(cue.Text) == "" {
			return errors.New("이전 결과에 원문 데이터가 없어 원문 자막을 재생성할 수 없습니다")
		}
		if (mode == "translated" || mode == "bilingual") && strings.TrimSpace(cue.Translated) == "" {
			return errors.New("이전 결과에 번역문 데이터가 없어 번역 자막을 재생성할 수 없습니다")
		}
	}
	return nil
}

func (s *Server) subtitleCueArchivePath(id string) string {
	return s.jobs.OutputPath(id + ".cues.json")
}

func (s *Server) writeSubtitleCueArchive(id string, cues []subtitleCue) error {
	data, err := json.Marshal(cues)
	if err != nil {
		return fmt.Errorf("subtitle cue archive: %w", err)
	}
	path := s.subtitleCueArchivePath(id)
	temporary := path + ".tmp"
	if err := os.WriteFile(temporary, data, 0o600); err != nil {
		return fmt.Errorf("subtitle cue archive: %w", err)
	}
	if err := os.Rename(temporary, path); err != nil {
		_ = os.Remove(temporary)
		return fmt.Errorf("subtitle cue archive: %w", err)
	}
	return nil
}

func (s *Server) loadSubtitleCueArchive(job jobs.Job) ([]subtitleCue, bool, error) {
	data, err := os.ReadFile(s.subtitleCueArchivePath(job.ID))
	if err == nil {
		var cues []subtitleCue
		if json.Unmarshal(data, &cues) == nil && len(cues) > 0 {
			return cues, false, nil
		}
	}
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, false, fmt.Errorf("자막 큐를 읽지 못했습니다: %w", err)
	}

	mode := jobStringParam(job.Params, "translation_mode", "none")
	targetLanguage := jobStringParam(job.Params, "target_language", "")
	if outputURL := job.Outputs["srt"]; outputURL != "" {
		data, readErr := os.ReadFile(s.jobs.OutputPath(filepath.Base(outputURL)))
		if readErr == nil {
			cues, parseErr := parseRenderedSubtitle(string(data), "srt", mode, targetLanguage)
			if parseErr == nil && len(cues) > 0 {
				return cues, true, nil
			}
		}
	}
	if job.CaptionURL != "" {
		data, readErr := os.ReadFile(s.jobs.OutputPath(filepath.Base(job.CaptionURL)))
		if readErr == nil {
			cues, parseErr := parseRenderedSubtitle(string(data), "vtt", mode, targetLanguage)
			if parseErr == nil && len(cues) > 0 {
				return cues, true, nil
			}
		}
	}
	return nil, false, errors.New("이전 자막에서 원문·번역문 큐를 복구할 수 없습니다. 이 작업은 받아쓰기를 다시 실행해야 합니다")
}

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
