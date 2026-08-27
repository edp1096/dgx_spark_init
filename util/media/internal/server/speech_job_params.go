package server

import (
	"encoding/json"
	"mediaapp/internal/config"
)

type speechJobParams struct {
	Language     string `json:"language"`
	Speaker      string `json:"speaker"`
	Instructions string `json:"instructions,omitempty"`
	Seed         int64  `json:"seed"`
	Stage        string `json:"stage,omitempty"`
	QueuedAt     string `json:"queued_at,omitempty"`
}

func newSpeechJobParams(cfg config.Speech) speechJobParams {
	return speechJobParams{Language: cfg.DefaultLanguage, Speaker: cfg.DefaultSpeaker, Seed: -1}
}

func decodeSpeechJobParams(values map[string]any, cfg config.Speech) speechJobParams {
	result := newSpeechJobParams(cfg)
	data, err := json.Marshal(values)
	if err == nil {
		_ = json.Unmarshal(data, &result)
	}
	return result
}

func (p speechJobParams) toMap() map[string]any {
	return map[string]any{
		"language": p.Language, "speaker": p.Speaker, "instructions": p.Instructions,
		"seed": p.Seed, "stage": p.Stage, "queued_at": p.QueuedAt,
	}
}
