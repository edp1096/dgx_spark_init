package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"mediaapp/internal/jobs"
	"os"
	"path/filepath"
)

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

	params := decodeSubtitleJobParams(job.Params, s.config().Recognition)
	mode := params.TranslationMode
	targetLanguage := params.TargetLanguage
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
