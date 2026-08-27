package server

import (
	"encoding/json"
	"mediaapp/internal/config"
)

type subtitleJobParams struct {
	Language        string   `json:"language"`
	Context         string   `json:"context,omitempty"`
	Source          string   `json:"source"`
	SourceJobID     string   `json:"source_job_id,omitempty"`
	OutputFormats   []string `json:"output_formats"`
	TranslationMode string   `json:"translation_mode"`
	TargetLanguage  string   `json:"target_language"`
	MediaPart       string   `json:"media_part,omitempty"`
	MediaSource     string   `json:"media_source,omitempty"`
	Stage           string   `json:"stage,omitempty"`
	QueuedAt        string   `json:"queued_at,omitempty"`
}

func newSubtitleJobParams(cfg config.Recognition) subtitleJobParams {
	return subtitleJobParams{
		Language: cfg.DefaultLanguage, Source: "file",
		OutputFormats:   append([]string(nil), cfg.DefaultOutputFormats...),
		TranslationMode: cfg.DefaultTranslationMode,
		TargetLanguage:  cfg.DefaultTranslationLanguage,
	}
}

func decodeSubtitleJobParams(values map[string]any, cfg config.Recognition) subtitleJobParams {
	result := newSubtitleJobParams(cfg)
	data, err := json.Marshal(values)
	if err == nil {
		_ = json.Unmarshal(data, &result)
	}
	return result
}

func (p subtitleJobParams) toMap() map[string]any {
	result := map[string]any{
		"language": p.Language, "context": p.Context, "source": p.Source,
		"output_formats": p.OutputFormats, "translation_mode": p.TranslationMode,
		"target_language": p.TargetLanguage, "stage": p.Stage, "queued_at": p.QueuedAt,
	}
	if p.SourceJobID != "" {
		result["source_job_id"] = p.SourceJobID
	}
	if p.MediaPart != "" {
		result["media_part"] = p.MediaPart
	}
	if p.MediaSource != "" {
		result["media_source"] = p.MediaSource
	}
	return result
}
