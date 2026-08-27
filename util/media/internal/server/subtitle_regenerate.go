package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
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
