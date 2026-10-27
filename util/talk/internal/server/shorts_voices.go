package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"sparktalk/internal/db"
	"sparktalk/internal/llm"
)

func (s *Server) registerShortsVoices(reg *completionToolRegistry) {
	cfg, _ := s.snapshot()
	if cfg.SupportDisabled("fish") {
		return
	}
	reg.prompts = append(reg.prompts, `For narration voice questions always call shorts_voices list for the current actual catalog. Never invent voices or use the cloud Fish library as the installed catalog. list shows all installed voices with saved Fish audio previews; preview plays one voice. select sets a conversation default for future new videos; apply changes an existing video's narration voice in a new version while preserving its words, images, music and visual design. Use apply only when the user asks to change an existing video. A list/preview request must never generate or revise a video. Use the exact voice_id from the catalog; preview/apply can omit voice_id to use the saved conversation selection. New shorts_generate accepts voice_preset to enforce an explicit choice. Builtins use synthetic Qwen references; custom voices use user-provided references. No per-user weight training.`)
	reg.register(llm.Tool{Type: "function", Function: llm.ToolFunction{Name: "shorts_voices", Description: "List all installed video narration voices with names and playable Fish samples; preview one, select a conversation default, or apply a voice to an existing video as a new version. Does not train models. Omit job_id for the latest video when applying.", Parameters: json.RawMessage(`{"type":"object","properties":{"action":{"type":"string","enum":["list","preview","select","apply"]},"voice_id":{"type":"string","description":"Catalog ID; omit for preview/apply to use the selected conversation voice."},"job_id":{"type":"string"}},"required":["action"],"additionalProperties":false}`)}}, func(ctx context.Context, call llm.ToolCall, _ []llm.Message, emit eventEmitter) (registeredToolResult, error) {
		var a struct {
			Action  string `json:"action"`
			VoiceID string `json:"voice_id"`
			JobID   string `json:"job_id"`
		}
		if json.Unmarshal([]byte(call.Function.Arguments), &a) != nil {
			return registeredToolResult{}, fmt.Errorf("잘못된 음성 도구 입력")
		}
		if err := s.ensureShorts(ctx); err != nil {
			return registeredToolResult{}, err
		}
		if a.Action == "apply" {
			job, err := s.conversationShortsJob(ctx, reg.sessionID, a.JobID)
			if err != nil {
				return registeredToolResult{}, err
			}
			a.JobID = job.ID
		}
		raw, code, err := s.shortsCall(ctx, "POST", "/v1/voices", map[string]any{"session_id": reg.sessionID, "action": a.Action, "voice_id": a.VoiceID, "job_id": a.JobID})
		if err != nil {
			return registeredToolResult{}, err
		}
		if code != 200 {
			return registeredToolResult{}, fmt.Errorf("음성 도구 HTTP %d: %s", code, raw)
		}
		if a.Action == "apply" {
			var job shortsJob
			if json.Unmarshal(raw, &job) != nil || !validShortsJobID(job.ID) || job.SessionID != reg.sessionID {
				return registeredToolResult{}, fmt.Errorf("잘못된 음성 변경 작업 응답")
			}
			return s.waitForShorts(ctx, reg.sessionID, call.ID, job, emit)
		}
		out := registeredToolResult{Result: string(raw)}
		if a.Action == "select" {
			out.FinalContent = fmt.Sprintf("이 대화에서 앞으로 새로 만드는 영상의 기본 음성을 %s로 선택했습니다. 기존 영상은 그대로입니다.", a.VoiceID)
			return out, nil
		}
		if a.Action == "preview" {
			var selected struct {
				Voice      string `json:"voice"`
				PreviewSHA string `json:"preview_sha256"`
			}
			if json.Unmarshal(raw, &selected) != nil || selected.Voice == "" {
				return registeredToolResult{}, fmt.Errorf("잘못된 미리 듣기 음성 응답")
			}
			a.VoiceID = selected.Voice
			attachment, err := s.shortsVoicePreview(ctx, reg.sessionID, a.VoiceID, selected.PreviewSHA)
			if err != nil {
				return registeredToolResult{}, err
			}
			out.Attachments = []db.Attachment{attachment}
			out.FinalContent = a.VoiceID + "의 저장된 Fish 미리 듣기입니다."
			return out, nil
		}
		var catalog struct {
			Voices []struct {
				ID             string `json:"id"`
				Name           string `json:"name"`
				Description    string `json:"description"`
				NativeLanguage string `json:"native_language"`
				PreviewSHA     string `json:"preview_sha256"`
			} `json:"voices"`
			Selected string `json:"selected_voice"`
		}
		if a.Action != "list" || json.Unmarshal(raw, &catalog) != nil || len(catalog.Voices) == 0 {
			return registeredToolResult{}, fmt.Errorf("잘못된 음성 목록 응답")
		}
		var body strings.Builder
		fmt.Fprintf(&body, "현재 영상 제작에 사용할 수 있는 Fish 음성은 %d개입니다. 기본 음성과 등록된 사용자 음성의 실제 Fish 미리 듣기를 첨부했습니다.\n\n| 음성 ID | 이름·특징 | 기본 언어 |\n|---|---|---|\n", len(catalog.Voices))
		for _, v := range catalog.Voices {
			attachment, err := s.shortsVoicePreview(ctx, reg.sessionID, v.ID, v.PreviewSHA)
			if err != nil {
				return registeredToolResult{}, err
			}
			out.Attachments = append(out.Attachments, attachment)
			fmt.Fprintf(&body, "| `%s` | %s — %s | %s |\n", v.ID, v.Name, v.Description, v.NativeLanguage)
		}
		if catalog.Selected != "" {
			fmt.Fprintf(&body, "\n이 대화의 선택 음성: `%s`.\n", catalog.Selected)
		}
		body.WriteString("\n음성별 억양과 참조 언어는 다릅니다. ‘Ryan을 기본 음성으로 선택해’ 또는 ‘기존 영상에 Ryan을 적용해’처럼 요청하세요.")
		out.FinalContent = body.String()
		return out, nil
	})
}

func (s *Server) shortsVoicePreview(ctx context.Context, sessionID, voiceID, expectedSHA string) (db.Attachment, error) {
	if !validVoiceID(voiceID) {
		return db.Attachment{}, fmt.Errorf("잘못된 음성 ID")
	}
	marker := "/api/shorts/voices/" + voiceID + "/preview.wav"
	if expectedSHA != "" {
		if len(expectedSHA) != 64 || strings.Trim(expectedSHA, "0123456789abcdef") != "" {
			return db.Attachment{}, fmt.Errorf("잘못된 음성 미리 듣기 해시")
		}
		marker += "?sha256=" + expectedSHA
	}
	messages, err := s.db.Messages(sessionID)
	if err != nil {
		return db.Attachment{}, err
	}
	for _, m := range messages {
		for _, a := range m.Attachments {
			if a.SourceURL == marker {
				return a, nil
			}
		}
	}
	request, err := http.NewRequestWithContext(ctx, "GET", s.shortsEndpoint()+"/v1/voices/"+voiceID+"/preview.wav?session_id="+url.QueryEscape(sessionID), nil)
	if err != nil {
		return db.Attachment{}, err
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return db.Attachment{}, err
	}
	defer response.Body.Close()
	if response.StatusCode != 200 || !strings.HasPrefix(response.Header.Get("Content-Type"), "audio/") {
		return db.Attachment{}, fmt.Errorf("음성 미리 듣기를 읽지 못했습니다")
	}
	data, err := io.ReadAll(io.LimitReader(response.Body, 8<<20))
	if err != nil {
		return db.Attachment{}, err
	}
	if expectedSHA != "" && fmt.Sprintf("%x", sha256.Sum256(data)) != expectedSHA {
		return db.Attachment{}, fmt.Errorf("음성 미리 듣기 파일이 목록과 다릅니다")
	}
	attachment, err := s.media.SaveReader(bytes.NewReader(data), voiceID+"-Fish-preview.wav", "audio/wav", 8<<20)
	attachment.SourceURL = marker
	return attachment, err
}
