package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"sparktalk/internal/db"
	"sparktalk/internal/llm"
)

type musicRequest struct {
	Model    string `json:"model"`
	Prompt   string `json:"prompt"`
	Duration int    `json:"duration_seconds"`
	Seed     int    `json:"seed"`
	BPM      *int   `json:"bpm,omitempty"`
}

type musicResponse struct {
	Audio    string          `json:"audio"`
	MIME     string          `json:"mime_type"`
	Metadata json.RawMessage `json:"metadata"`
}

func (s *Server) generateMusic(ctx context.Context, a musicRequest) (result musicResponse, resultErr error) {
	cfg, _ := s.snapshot()
	if cfg.SupportDisabled("music-ace") {
		return result, fmt.Errorf("현재 모델 세트에서는 ACE-Step 음악 생성을 제외했습니다")
	}
	if a.Model == "" {
		a.Model = "ace-turbo"
	}
	if a.Duration == 0 {
		a.Duration = 30
	}
	key := "music-ace"
	switch a.Model {
	case "ace-turbo", "ace-xl":
	default:
		return result, fmt.Errorf("지원하지 않는 음악 모델")
	}
	if len([]rune(strings.TrimSpace(a.Prompt))) < 5 || len([]rune(a.Prompt)) > 2000 || a.Duration < 10 || a.Duration > 120 || a.Seed < 0 || a.Seed > 2147483647 || (a.BPM != nil && (*a.BPM < 30 || *a.BPM > 300)) {
		return result, fmt.Errorf("음악 입력·길이·시드·BPM을 확인하세요")
	}
	peak := 24.0
	if a.Model == "ace-xl" {
		peak = 32
	}
	release, err := s.acquireWorkload(ctx, "extra-"+key, peak)
	if err != nil {
		return result, err
	}
	defer finishWorkload(release, &resultErr)
	cfg, _ = s.snapshot()
	raw, _ := json.Marshal(a)
	req, err := http.NewRequestWithContext(ctx, "POST", strings.TrimRight(cfg.SupportEndpoint(key), "/")+"/v1/music/generate", bytes.NewReader(raw))
	if err != nil {
		return result, err
	}
	req.Header.Set("Content-Type", "application/json")
	response, err := (&http.Client{Timeout: 31 * time.Minute}).Do(req)
	if err != nil {
		return result, err
	}
	defer response.Body.Close()
	if response.StatusCode != 200 {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, 8000))
		return result, fmt.Errorf("음악 생성 HTTP %d: %s", response.StatusCode, detail)
	}
	err = json.NewDecoder(io.LimitReader(response.Body, 48<<20)).Decode(&result)
	if err == nil && (result.MIME != "audio/wav" || result.Audio == "" || !json.Valid(result.Metadata)) {
		err = fmt.Errorf("음악 생성 응답이 잘못됐습니다")
	}
	return result, err
}

func (s *Server) registerMusicTool(reg *completionToolRegistry) {
	cfg, _ := s.snapshot()
	if cfg.SupportDisabled("music-ace") {
		return
	}
	reg.prompts = append(reg.prompts, "Use music_generate for requested instrumental music or standalone BGM. Available models are ace-turbo (default) and ace-xl only. If the user explicitly requests an unavailable model, explain that it is unavailable; do not silently substitute another model. It attaches actual model-generated audio. Report the actual model and measured duration; audio metrics do not establish subjective quality or guarantee absence of vocals. For video BGM use shorts tools with music.kind=generated and model/prompt. Never substitute synthetic keyboard notes when music generation fails.")
	reg.register(llm.Tool{Type: "function", Function: llm.ToolFunction{Name: "music_generate", Description: "Generate and attach instrumental music using ACE-Step 1.5 Turbo or ACE-Step XL Base. Takes a music description, duration, optional BPM, and seed. GPU generation may take several minutes. No vocals/lyrics interface. Do not call repeatedly to check progress.", Parameters: json.RawMessage(`{"type":"object","properties":{"model":{"type":"string","enum":["ace-turbo","ace-xl"]},"prompt":{"type":"string","minLength":5,"maxLength":2000},"duration_seconds":{"type":"integer","minimum":10,"maximum":120},"seed":{"type":"integer","minimum":0,"maximum":2147483647},"bpm":{"type":"integer","minimum":30,"maximum":300}},"required":["prompt"],"additionalProperties":false}`)}}, func(ctx context.Context, call llm.ToolCall, _ []llm.Message, emit eventEmitter) (registeredToolResult, error) {
		a := musicRequest{Seed: 42}
		decoder := json.NewDecoder(strings.NewReader(call.Function.Arguments))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&a); err != nil {
			return registeredToolResult{}, err
		}
		result, err := s.generateMusic(ctx, a)
		if err != nil {
			return registeredToolResult{}, err
		}
		data, err := base64.StdEncoding.DecodeString(result.Audio)
		if err != nil || len(data) > 32<<20 {
			return registeredToolResult{}, fmt.Errorf("음악 WAV 응답 오류")
		}
		attachment, err := s.media.SaveReader(bytes.NewReader(data), "Generated-music.wav", "audio/wav", 32<<20)
		if err != nil {
			return registeredToolResult{}, err
		}
		return registeredToolResult{Result: string(result.Metadata), Attachments: []db.Attachment{attachment}}, nil
	})
}
