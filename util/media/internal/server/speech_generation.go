package server

import (
	"context"
	"mediaapp/internal/jobs"
	"net/http"
	"os"
	"strings"
	"time"
)

func (s *Server) createSpeech(w http.ResponseWriter, r *http.Request) {
	cfg := s.config()
	if err := r.ParseMultipartForm(40 << 20); err != nil {
		http.Error(w, "invalid form", 400)
		return
	}
	text := strings.TrimSpace(r.FormValue("text"))
	if text == "" {
		http.Error(w, "text is required", 400)
		return
	}
	id := newID()
	language := valueOr(r.FormValue("language"), cfg.Speech.DefaultLanguage)
	speaker := valueOr(r.FormValue("speaker"), cfg.Speech.DefaultSpeaker)
	instructions := strings.TrimSpace(r.FormValue("instructions"))
	seed := formInt64(r, "seed", -1)
	params := speechJobParams{
		Language: language, Speaker: speaker, Instructions: instructions, Seed: seed,
		Stage: "queued",
	}
	now := time.Now()
	params.QueuedAt = now.Format(time.RFC3339Nano)
	j := jobs.Job{ID: id, Kind: "speech", Status: "queued", Prompt: text, Params: params.toMap(), CreatedAt: now}
	if err := s.jobs.Save(j); err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	s.wakeGenerationQueue()
	writeJSON(w, 202, j)
}

func (s *Server) runSpeech(ctx context.Context, j jobs.Job, params speechJobParams) {
	cfg := s.config()
	request := map[string]any{
		"model": cfg.Speech.CustomVoiceModel, "input": j.Prompt,
		"language": params.Language, "voice": strings.ToLower(params.Speaker),
		"instructions": params.Instructions,
		"task_type":    "CustomVoice", "response_format": "wav", "stream": false,
	}
	if params.Seed >= 0 {
		request["seed"] = params.Seed
	}
	s.publishLocalRuntimePhase(
		j.ID, "sampling", "Qwen3-TTS 1.7B", "상주 모델로 음성 조건 인코딩·합성", .35, "retain", runtimeBoolPointer(true),
	)
	data, err := s.generateSpeechWithEngine(ctx, request)
	if err != nil {
		s.fail(j, err)
		return
	}
	s.publishLocalRuntimePhase(j.ID, "finalizing", "WAV 출력", "음성 파일 저장", .96, "", runtimeBoolPointer(true))
	name := j.ID + ".wav"
	if err = os.WriteFile(s.jobs.OutputPath(name), data, 0o644); err != nil {
		s.fail(j, err)
		return
	}
	s.publishLocalRuntimePhase(
		j.ID, "completed", "Qwen3-TTS 1.7B", "음성 생성 완료·모델 상주 유지", 1, "retain", runtimeBoolPointer(true),
	)
	_, _ = s.completeGenerationJob(&j, "/api/outputs/"+name, func() {
		_ = os.Remove(s.jobs.OutputPath(name))
	})
}
