package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
	"unicode"

	"sparktalk/internal/db"
	"sparktalk/internal/llm"
)

type shortsScene struct {
	Title           string          `json:"title"`
	Narration       string          `json:"narration"`
	Visual          string          `json:"visual"`
	ImagePrompt     string          `json:"image_prompt"`
	AttachmentID    string          `json:"attachment_id"`
	Points          []string        `json:"points"`
	Emphasis        []string        `json:"emphasis,omitempty"`
	ReuseScene      *int            `json:"reuse_scene,omitempty"`
	AssetID         string          `json:"asset_id,omitempty"`
	DurationSeconds float64         `json:"duration_seconds,omitempty"`
	Manim           json.RawMessage `json:"manim,omitempty"`
}
type shortsRequest struct {
	SessionID       string          `json:"session_id"`
	Title           string          `json:"title"`
	Brief           string          `json:"brief"`
	Scenes          []shortsScene   `json:"scenes,omitempty"`
	BGM             bool            `json:"bgm"`
	DurationSeconds int             `json:"duration_seconds,omitempty"`
	Texts           []string        `json:"texts,omitempty"`
	Audio           string          `json:"audio,omitempty"`
	Expected        string          `json:"expected,omitempty"`
	Scene           *shortsScene    `json:"scene,omitempty"`
	JobID           string          `json:"job_id,omitempty"`
	SourceJobID     string          `json:"source_job_id,omitempty"`
	TemplateID      string          `json:"template_id,omitempty"`
	DesignMode      string          `json:"design_mode,omitempty"`
	AssetIDs        []string        `json:"asset_ids,omitempty"`
	PreviewOnly     bool            `json:"preview_only,omitempty"`
	Plan            json.RawMessage `json:"plan,omitempty"`
	VoicePreset     string          `json:"voice_preset,omitempty"`
	Voice           json.RawMessage `json:"voice,omitempty"`
	Phase           string          `json:"phase,omitempty"`
	Data            json.RawMessage `json:"data,omitempty"`
	Images          []string        `json:"images,omitempty"`
	Music           *musicRequest   `json:"music,omitempty"`
}

func (s *Server) shortsEndpoint() string {
	cfg, _ := s.snapshot()
	return strings.TrimRight(cfg.SupportEndpoint("shorts"), "/")
}

func (s *Server) shortsCall(ctx context.Context, method, path string, data any) ([]byte, int, error) {
	var body io.Reader
	if data != nil {
		raw, err := json.Marshal(data)
		if err != nil {
			return nil, 0, err
		}
		body = bytes.NewReader(raw)
	}
	r, err := http.NewRequestWithContext(ctx, method, s.shortsEndpoint()+path, body)
	if err != nil {
		return nil, 0, err
	}
	r.Header.Set("Content-Type", "application/json")
	response, err := (&http.Client{Timeout: 20 * time.Minute}).Do(r)
	if err != nil {
		return nil, 0, err
	}
	defer response.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(response.Body, 64<<20))
	return raw, response.StatusCode, err
}

func (s *Server) ensureShorts(ctx context.Context) error {
	cfg, _ := s.snapshot()
	if s.runtime == nil || cfg.Runtime.Mode != "managed" {
		return fmt.Errorf("쇼츠는 로컬 관리 모드가 필요합니다")
	}
	return s.runtime.EnsureShortsWorker(ctx, cfg.Runtime.ActiveBundle, cfg.Runtime.MemoryReserveGiB)
}

func (s *Server) shortsAPI(w http.ResponseWriter, r *http.Request) {
	path := strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/shorts"), "/")
	cfg, _ := s.snapshot()
	if r.Method == http.MethodPost && ((path == "music" && cfg.SupportDisabled("music-ace")) || (path == "speech" && cfg.SupportDisabled("fish"))) {
		http.Error(w, "현재 모델 세트에서는 Fish·ACE-Step 생성을 제외했습니다", http.StatusConflict)
		return
	}

	if r.Method == "POST" && (path == "plan" || path == "author" || path == "speech" || path == "music" || path == "transcribe" || path == "asset" || path == "publish") {
		var a shortsRequest
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 24<<20)).Decode(&a); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		if _, err := s.db.Session(a.SessionID); err != nil {
			http.Error(w, "대화가 없습니다", 404)
			return
		}
		var result any
		var err error
		switch path {
		case "plan":
			result, err = s.shortsPlan(r.Context(), a)
		case "author":
			result, err = s.shortsAuthor(r.Context(), a)
		case "speech":
			result, err = s.shortsSpeech(r.Context(), a)
		case "music":
			if a.Music == nil {
				err = fmt.Errorf("음악 생성 입력이 필요합니다")
			} else {
				result, err = s.generateMusic(r.Context(), *a.Music)
			}
		case "transcribe":
			result, err = s.shortsTranscribe(r.Context(), a)
		case "asset":
			result, err = s.shortsAsset(r.Context(), a)
		case "publish":
			result, err = s.shortsPublish(r.Context(), a)
		}
		if err != nil {
			http.Error(w, err.Error(), 502)
			return
		}
		writeJSON(w, 200, result)
		return
	}
	session := r.URL.Query().Get("session_id")
	if _, err := s.db.Session(session); err != nil {
		http.Error(w, "대화가 없습니다", 404)
		return
	}
	if err := s.ensureShorts(r.Context()); err != nil {
		http.Error(w, err.Error(), 503)
		return
	}
	if path == "" {
		if r.Method == "GET" {
			raw, code, err := s.shortsCall(r.Context(), "GET", "/v1/projects?session_id="+url.QueryEscape(session), nil)
			shortsReply(w, raw, code, err)
			return
		}
		if r.Method != "POST" {
			methodNotAllowed(w)
			return
		}
		var a shortsRequest
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 200000)).Decode(&a); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		a.SessionID = session
		raw, code, err := s.shortsCall(r.Context(), "POST", "/v1/projects", a)
		shortsReply(w, raw, code, err)
		return
	}
	parts := strings.Split(path, "/")
	if len(parts[0]) != 32 || strings.Trim(parts[0], "0123456789abcdef") != "" {
		http.NotFound(w, r)
		return
	}
	raw, code, err := s.shortsCall(r.Context(), "GET", "/v1/projects/"+parts[0], nil)
	if err != nil || code != 200 {
		shortsReply(w, raw, code, err)
		return
	}
	var job struct {
		SessionID string `json:"session_id"`
	}
	if json.Unmarshal(raw, &job) != nil || job.SessionID != session {
		http.NotFound(w, r)
		return
	}
	if len(parts) == 1 && r.Method == "GET" {
		shortsReply(w, raw, code, nil)
		return
	}
	if len(parts) == 2 && r.Method == "POST" && (parts[1] == "cancel" || parts[1] == "retry" || parts[1] == "render") {
		raw, code, err = s.shortsCall(r.Context(), "POST", "/v1/projects/"+path, nil)
		shortsReply(w, raw, code, err)
		return
	}
	if len(parts) == 3 && parts[1] == "files" && r.Method == "GET" {
		allowed := map[string]bool{"final.mp4": true, "poster.png": true, "storyboard.json": true, "timing.json": true, "quality.json": true, "sources.json": true, "preview.png": true, "preview-sheet.jpg": true, "artwork.json": true, "audit.json": true, "composition-spec.json": true}
		if !allowed[parts[2]] {
			http.NotFound(w, r)
			return
		}
		req, _ := http.NewRequestWithContext(r.Context(), "GET", s.shortsEndpoint()+"/v1/projects/"+path, nil)
		if value := r.Header.Get("Range"); value != "" {
			req.Header.Set("Range", value)
		}
		response, e := http.DefaultClient.Do(req)
		if e != nil {
			http.Error(w, e.Error(), 502)
			return
		}
		defer response.Body.Close()
		for _, key := range []string{"Content-Type", "Content-Length", "Content-Range", "Accept-Ranges", "Last-Modified"} {
			if value := response.Header.Get(key); value != "" {
				w.Header().Set(key, value)
			}
		}
		w.WriteHeader(response.StatusCode)
		_, _ = io.Copy(w, response.Body)
		return
	}
	http.NotFound(w, r)
}

func shortsReply(w http.ResponseWriter, raw []byte, code int, err error) {
	if err != nil {
		http.Error(w, err.Error(), 502)
		return
	}
	if code < 100 {
		code = 502
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_, _ = w.Write(raw)
}

func (s *Server) shortsPublish(ctx context.Context, a shortsRequest) (any, error) {
	if len(a.JobID) != 32 || strings.Trim(a.JobID, "0123456789abcdef") != "" {
		return nil, fmt.Errorf("잘못된 작업 ID")
	}
	s.shortsPublishMu.Lock()
	defer s.shortsPublishMu.Unlock()
	raw, code, err := s.shortsCall(ctx, "GET", "/v1/projects/"+a.JobID, nil)
	if err != nil {
		return nil, err
	}
	if code != 200 {
		return nil, fmt.Errorf("작업을 찾을 수 없습니다")
	}
	var job struct {
		SessionID   string `json:"session_id"`
		Title       string `json:"title"`
		Status      string `json:"status"`
		OutputReady bool   `json:"output_ready"`
	}
	if json.Unmarshal(raw, &job) != nil || job.SessionID != a.SessionID || job.Status != "completed" || !job.OutputReady {
		return nil, fmt.Errorf("이 대화에서 완료된 영상이 필요합니다")
	}
	marker := "/api/shorts/" + a.JobID + "/files/final.mp4?session_id=" + url.QueryEscape(a.SessionID)
	messages, err := s.db.Messages(a.SessionID)
	if err != nil {
		return nil, err
	}
	for _, message := range messages {
		for _, attachment := range message.Attachments {
			if attachment.SourceURL == marker {
				return map[string]any{"message_id": message.ID, "attachment": attachment}, nil
			}
		}
	}
	req, _ := http.NewRequestWithContext(ctx, "GET", s.shortsEndpoint()+"/v1/projects/"+a.JobID+"/files/final.mp4", nil)
	response, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != 200 || !strings.HasPrefix(response.Header.Get("Content-Type"), "video/mp4") {
		return nil, fmt.Errorf("완성된 MP4를 읽지 못했습니다")
	}
	attachment, err := s.media.SaveReader(response.Body, "Shorts.mp4", "video/mp4", s.media.Limits().MaxBytes())
	if err != nil {
		return nil, err
	}
	attachment.SourceURL = marker
	message, err := s.db.AddMessage(a.SessionID, "assistant", job.Title+" · 720×1280 세로 영상", "", nil, []db.Attachment{attachment})
	if err != nil {
		return nil, err
	}
	return map[string]any{"message_id": message.ID, "attachment": attachment}, nil
}

func (s *Server) shortsPlan(ctx context.Context, a shortsRequest) (any, error) {
	// Compatibility entrypoint; the worker supplies the full editable schema to author.
	data, _ := json.Marshal(map[string]any{"brief": a.Brief, "duration_seconds": a.DurationSeconds, "target_narration_characters": a.DurationSeconds * 6, "schema_hint": "title,design_brief,scenes[{title,narration,visual,image_prompt,points,emphasis}],captions,voice,music,asset_ids,lottie_queries,sources,limitations"})
	a.Phase, a.Data = "plan", data
	return s.shortsAuthor(ctx, a)
}

func (s *Server) shortsSpeech(ctx context.Context, a shortsRequest) (result any, resultErr error) {
	cfg, _ := s.snapshot()
	if cfg.SupportDisabled("fish") {
		return nil, fmt.Errorf("현재 모델 세트에서는 Fish 음성 생성을 제외했습니다")
	}
	texts := a.Texts
	if len(texts) < 1 || len(texts) > 16 {
		return nil, fmt.Errorf("음성 장면 수는 1–16개여야 합니다")
	}
	total := 0
	for _, text := range texts {
		total += len([]rune(text))
		if len([]rune(text)) < 1 || len([]rune(text)) > 500 {
			return nil, fmt.Errorf("잘못된 음성 대본")
		}
	}
	if total > 1800 {
		return nil, fmt.Errorf("전체 내레이션은 1800자 이하여야 합니다")
	}
	var voice map[string]any
	if len(a.Voice) > 0 {
		if err := json.Unmarshal(a.Voice, &voice); err != nil {
			return nil, fmt.Errorf("잘못된 음성 설정")
		}
		voice["session_id"] = a.SessionID
		delete(voice, "pinned_reference")
		preparing, _ := voice["preparing_reference"].(bool)
		delete(voice, "preparing_reference")
		preset, _ := voice["preset"].(string)
		if customVoiceID.MatchString(preset) {
			raw, code, err := s.shortsCall(ctx, "POST", "/v1/voice-resolve", map[string]any{"session_id": a.SessionID, "voice_id": preset, "revision": voice["reference_revision"], "job_id": a.JobID, "preparing": preparing})
			if err != nil {
				return nil, err
			}
			if code != 200 {
				return nil, fmt.Errorf("참조 음성 확인 HTTP %d: %s", code, raw)
			}
			var resolved struct {
				Revision  int  `json:"revision"`
				Preparing bool `json:"preparing_reference"`
				Pinned    bool `json:"pinned_reference"`
			}
			if err := json.Unmarshal(raw, &resolved); err != nil {
				return nil, err
			}
			voice["reference_revision"] = resolved.Revision
			voice["preparing_reference"] = resolved.Preparing
			voice["pinned_reference"] = resolved.Pinned
		}
	}
	release, err := s.acquireWorkload(ctx, "extra-fish")
	if err != nil {
		return nil, err
	}
	defer finishWorkload(release, &resultErr)
	cfg, _ = s.snapshot()
	payload := map[string]any{"texts": texts}
	if len(a.Voice) > 0 {
		payload["voice"] = voice
	}
	raw, _ := json.Marshal(payload)
	req, _ := http.NewRequestWithContext(ctx, "POST", strings.TrimRight(cfg.SupportEndpoint("fish"), "/")+"/v1/audio/batch", bytes.NewReader(raw))
	req.Header.Set("Content-Type", "application/json")
	response, err := (&http.Client{Timeout: 15 * time.Minute}).Do(req)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode != 200 {
		detail, _ := io.ReadAll(io.LimitReader(response.Body, 4096))
		return nil, fmt.Errorf("Fish HTTP %d: %s", response.StatusCode, detail)
	}
	var data json.RawMessage
	err = json.NewDecoder(io.LimitReader(response.Body, 40<<20)).Decode(&data)
	return data, err
}

func speechCoverage(expected, actual string) float64 {
	normalize := func(text string) []rune {
		var out []rune
		for _, r := range strings.ToLower(text) {
			if unicode.IsLetter(r) || unicode.IsDigit(r) {
				out = append(out, r)
			}
		}
		return out
	}
	a, b := normalize(expected), normalize(actual)
	if len(a) == 0 {
		return 0
	}
	previous := make([]int, len(b)+1)
	for j := range previous {
		previous[j] = j
	}
	for i, x := range a {
		current := make([]int, len(b)+1)
		current[0] = i + 1
		for j, y := range b {
			cost := 0
			if x != y {
				cost = 1
			}
			current[j+1] = min(previous[j+1]+1, current[j]+1, previous[j]+cost)
		}
		previous = current
	}
	return float64(previous[len(b)]) / float64(len(a))
}

func (s *Server) shortsTranscribe(ctx context.Context, a shortsRequest) (any, error) {
	data, err := base64.StdEncoding.DecodeString(a.Audio)
	if err != nil || len(data) > 12<<20 || len(a.Expected) > 4000 {
		return nil, fmt.Errorf("잘못된 음성 검사 입력")
	}
	client, release, err := s.prepareASRWorkload(ctx, s.asrSnapshot())
	if err != nil {
		return nil, err
	}
	result, err := client.Transcribe(ctx, bytes.NewReader(data), "shorts.wav", "audio/wav")
	cleanup := release(err)
	if err != nil {
		return nil, err
	}
	if cleanup != nil {
		return nil, cleanup
	}
	cer := speechCoverage(a.Expected, result.Text)
	return map[string]any{"expected": a.Expected, "transcript": result.Text, "character_error_rate": cer, "passed": cer <= .25}, nil
}

func (s *Server) shortsAsset(ctx context.Context, a shortsRequest) (any, error) {
	if a.Scene == nil {
		return nil, fmt.Errorf("장면이 없습니다")
	}
	scene := a.Scene
	cfg, _ := s.snapshot()
	var attachment db.Attachment
	kind := "image"
	if scene.AttachmentID != "" {
		images, err := s.sessionImageAttachments(a.SessionID)
		if err != nil {
			return nil, err
		}
		var ok bool
		attachment, ok = images[scene.AttachmentID]
		if !ok {
			return nil, fmt.Errorf("이 대화에 이미지가 없습니다")
		}
	} else if scene.Visual == "image" {
		args, _ := json.Marshal(map[string]any{"operation": "generate", "prompt": scene.ImagePrompt})
		result, err := s.executeImageGenerateTool(ctx, a.SessionID, cfg.Image, llm.ToolCall{ID: "shorts-image", Function: llm.FunctionCall{Arguments: string(args)}}, nil)
		if err != nil {
			return nil, err
		}
		if result.Attachment != nil {
			attachment = *result.Attachment
		} else if len(result.Attachments) > 0 {
			attachment = result.Attachments[0]
		} else {
			return nil, fmt.Errorf("이미지 출력이 없습니다")
		}
	} else if scene.Visual == "video" {
		kind = "video"
		reg := completionToolRegistry{sessionID: a.SessionID, handlers: map[string]registeredToolHandler{}}
		s.registerVideoTool(ctx, &reg, cfg, func(item db.Attachment) error { attachment = item; return nil })
		handler, ok := reg.handlers["video_generate"]
		if !ok {
			return nil, fmt.Errorf("영상 생성 서비스가 없습니다")
		}
		args, _ := json.Marshal(map[string]any{"prompt": scene.ImagePrompt, "resolution": "720p"})
		if _, err := handler(ctx, llm.ToolCall{ID: "shorts-video", Function: llm.FunctionCall{Arguments: string(args)}}, nil, nil); err != nil {
			return nil, err
		}
	} else {
		return nil, fmt.Errorf("지원하지 않는 장면 종류")
	}
	file, err := s.media.Open(attachment)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, 60<<20))
	if err != nil {
		return nil, err
	}
	return map[string]any{"data": base64.StdEncoding.EncodeToString(data), "kind": kind, "attachment": attachment}, nil
}

func (s *Server) registerShortsTool(reg *completionToolRegistry) {
	if s.db == nil || reg.sessionID == "" {
		return
	}
	cfg, _ := s.snapshot()
	if cfg.Runtime.Mode != "managed" {
		return
	}
	if cfg.SupportDisabled("fish") {
		reg.prompts = append(reg.prompts, "This model set excludes Fish speech generation and narrated shorts. Do not claim these are available or use another voice model as a substitute. Explain the current set limitation if requested.")
		if !cfg.SupportDisabled("music-ace") {
			s.registerMusicTool(reg)
		} else {
			reg.prompts = append(reg.prompts, "ACE-Step music generation is also excluded in this model set. Do not substitute synthesized notes or another model.")
		}
		return
	}
	s.registerShortsStudio(reg)
	s.registerShortsVoices(reg)
	s.registerVoiceTools(reg)
	if !cfg.SupportDisabled("music-ace") {
		s.registerMusicTool(reg)
	}
	reg.register(llm.Tool{Type: "function", Function: llm.ToolFunction{Name: "shorts_generate", Description: "Create a 720x1280 Korean narrated short (usually 30–60 seconds) directly in this conversation. Fish voice, captions and animated scenes; optional generated images/video and BGM. Streams progress while waiting, then attaches the MP4. Use only for a requested short video. No separate interface or user button is required.", Parameters: json.RawMessage(`{"type":"object","properties":{"title":{"type":"string","maxLength":100},"brief":{"type":"string","minLength":3,"maxLength":12000},"bgm":{"type":"boolean"},"duration_seconds":{"type":"integer","minimum":15,"maximum":120,"description":"Requested finished duration in seconds; use 60 for one minute."},"source_job_id":{"type":"string"},"template_id":{"type":"string"},"design_mode":{"type":"string","enum":["redesign","preserve"],"description":"Default redesign. Use preserve only when the user explicitly wants the existing visual design unchanged."},"asset_ids":{"type":"array","items":{"type":"string"}},"preview_only":{"type":"boolean"},"voice_preset":{"type":"string","description":"Exact installed voice_id from shorts_voices list; overrides the conversation default."}},"required":["title","brief"],"additionalProperties":false}`)}}, func(ctx context.Context, call llm.ToolCall, _ []llm.Message, emit eventEmitter) (registeredToolResult, error) {
		var args struct {
			Title           string   `json:"title"`
			Brief           string   `json:"brief"`
			BGM             bool     `json:"bgm"`
			DurationSeconds int      `json:"duration_seconds"`
			SourceJobID     string   `json:"source_job_id"`
			TemplateID      string   `json:"template_id"`
			DesignMode      string   `json:"design_mode"`
			AssetIDs        []string `json:"asset_ids"`
			PreviewOnly     bool     `json:"preview_only"`
			VoicePreset     string   `json:"voice_preset"`
		}
		decoder := json.NewDecoder(strings.NewReader(call.Function.Arguments))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&args); err != nil {
			return registeredToolResult{}, err
		}
		a := shortsRequest{SessionID: reg.sessionID, Title: args.Title, Brief: args.Brief, BGM: args.BGM, DurationSeconds: args.DurationSeconds, SourceJobID: args.SourceJobID, TemplateID: args.TemplateID, DesignMode: args.DesignMode, AssetIDs: args.AssetIDs, PreviewOnly: args.PreviewOnly, VoicePreset: args.VoicePreset}
		if a.DurationSeconds < 0 || a.DurationSeconds > 120 || (a.DurationSeconds > 0 && a.DurationSeconds < 15) {
			return registeredToolResult{}, fmt.Errorf("영상 길이는 15–120초여야 합니다")
		}
		if strings.TrimSpace(a.Title) == "" || len([]rune(a.Title)) > 100 || len([]rune(strings.TrimSpace(a.Brief))) < 3 || len([]rune(a.Brief)) > 12000 {
			return registeredToolResult{}, fmt.Errorf("제목은 1–100자, 제작 요청은 3–12000자여야 합니다")
		}
		if err := s.ensureShorts(ctx); err != nil {
			return registeredToolResult{}, err
		}
		raw, code, err := s.shortsCall(ctx, "POST", "/v1/projects", a)
		if err != nil {
			return registeredToolResult{}, err
		}
		if code != 200 {
			return registeredToolResult{}, fmt.Errorf("Shorts HTTP %d: %s", code, raw)
		}
		var job shortsJob
		if err := json.Unmarshal(raw, &job); err != nil || !validShortsJobID(job.ID) || job.SessionID != reg.sessionID {
			return registeredToolResult{}, fmt.Errorf("쇼츠 작업 생성 응답이 잘못됐습니다")
		}
		return s.waitForShorts(ctx, reg.sessionID, call.ID, job, emit)
	})
	reg.register(llm.Tool{Type: "function", Function: llm.ToolFunction{Name: "shorts_manage", Description: "Check, wait for, cancel or retry an existing short video job in this conversation. Omit job_id for its most recent job. Retry resumes a failed/canceled job and waits with progress; wait resumes monitoring without creating a new video. Report the returned status, stage and error accurately.", Parameters: json.RawMessage(`{"type":"object","properties":{"action":{"type":"string","enum":["status","wait","cancel","retry"]},"job_id":{"type":"string","pattern":"^[a-f0-9]{32}$"}},"required":["action"],"additionalProperties":false}`)}}, func(ctx context.Context, call llm.ToolCall, _ []llm.Message, emit eventEmitter) (registeredToolResult, error) {
		var args struct {
			Action string `json:"action"`
			JobID  string `json:"job_id"`
		}
		decoder := json.NewDecoder(strings.NewReader(call.Function.Arguments))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&args); err != nil {
			return registeredToolResult{}, err
		}
		if args.Action != "status" && args.Action != "wait" && args.Action != "cancel" && args.Action != "retry" {
			return registeredToolResult{}, fmt.Errorf("알 수 없는 쇼츠 작업 명령")
		}
		if args.JobID != "" && !validShortsJobID(args.JobID) {
			return registeredToolResult{}, fmt.Errorf("잘못된 작업 ID")
		}
		if err := s.ensureShorts(ctx); err != nil {
			return registeredToolResult{}, err
		}
		job, err := s.conversationShortsJob(ctx, reg.sessionID, args.JobID)
		if err != nil {
			return registeredToolResult{}, err
		}
		if args.Action == "cancel" || args.Action == "retry" {
			raw, code, err := s.shortsCall(ctx, "POST", "/v1/projects/"+job.ID+"/"+args.Action, nil)
			if err != nil {
				return registeredToolResult{}, err
			}
			if code != 200 {
				return registeredToolResult{}, fmt.Errorf("Shorts HTTP %d: %s", code, raw)
			}
			if err := json.Unmarshal(raw, &job); err != nil || job.SessionID != reg.sessionID {
				return registeredToolResult{}, fmt.Errorf("작업 응답이 잘못됐습니다")
			}
		}
		if args.Action == "wait" || args.Action == "retry" {
			return s.waitForShorts(ctx, reg.sessionID, call.ID, job, emit)
		}
		return s.shortsJobResult(ctx, job)
	})
}

type shortsJob struct {
	ID           string          `json:"id"`
	SessionID    string          `json:"session_id"`
	Title        string          `json:"title"`
	Status       string          `json:"status"`
	Stage        string          `json:"stage"`
	Progress     float64         `json:"progress"`
	Created      float64         `json:"created"`
	Error        string          `json:"error,omitempty"`
	Canceled     int             `json:"canceled"`
	OutputReady  bool            `json:"output_ready"`
	PreviewReady bool            `json:"preview_ready"`
	Applied      json.RawMessage `json:"applied,omitempty"`
	Verification json.RawMessage `json:"verification,omitempty"`
	Limitations  []string        `json:"limitations,omitempty"`
}

// Retry/edit commits replace the later branch, including asynchronous delivery
// messages created during this turn. Restore only jobs this turn completed;
// never resurrect videos from the discarded history.
func (s *Server) restoreShortsDeliveries(ctx context.Context, sessionID string, trace []db.ToolEvent) error {
	seen := map[string]bool{}
	for _, event := range trace {
		if event.Error != "" || (event.Name != "shorts_generate" && event.Name != "shorts_manage" && event.Name != "shorts_project") {
			continue
		}
		var result struct {
			Job shortsJob `json:"job"`
		}
		if json.Unmarshal([]byte(event.Result), &result) != nil {
			continue
		}
		job := result.Job
		if job.SessionID != sessionID || job.Status != "completed" || !job.OutputReady || !validShortsJobID(job.ID) || seen[job.ID] {
			continue
		}
		seen[job.ID] = true
		if _, err := s.shortsPublish(ctx, shortsRequest{SessionID: sessionID, JobID: job.ID}); err != nil {
			return err
		}
	}
	return nil
}

func validShortsJobID(id string) bool {
	return len(id) == 32 && strings.Trim(id, "0123456789abcdef") == ""
}

func (s *Server) conversationShortsJob(ctx context.Context, sessionID, id string) (shortsJob, error) {
	path := "/v1/projects?session_id=" + url.QueryEscape(sessionID)
	if id != "" {
		if !validShortsJobID(id) {
			return shortsJob{}, fmt.Errorf("잘못된 작업 ID")
		}
		path = "/v1/projects/" + id
	}
	readCtx, cancel := context.WithTimeout(ctx, 8*time.Second)
	defer cancel()
	raw, code, err := s.shortsCall(readCtx, "GET", path, nil)
	if err != nil {
		return shortsJob{}, err
	}
	if code != 200 {
		return shortsJob{}, fmt.Errorf("작업 조회 HTTP %d", code)
	}
	var job shortsJob
	if id == "" {
		var jobs []shortsJob
		if err = json.Unmarshal(raw, &jobs); err == nil && len(jobs) > 0 {
			job = jobs[0]
		}
	} else {
		err = json.Unmarshal(raw, &job)
	}
	if err != nil || !validShortsJobID(job.ID) || job.SessionID != sessionID || (id != "" && job.ID != id) {
		return shortsJob{}, fmt.Errorf("이 대화의 쇼츠 작업을 찾을 수 없습니다")
	}
	return job, nil
}

func (s *Server) shortsJobResult(ctx context.Context, job shortsJob) (registeredToolResult, error) {
	result := map[string]any{"job": job}
	var attachments []db.Attachment
	if job.Status == "completed" && job.OutputReady {
		delivery, err := s.shortsPublish(ctx, shortsRequest{SessionID: job.SessionID, JobID: job.ID})
		if err != nil {
			return registeredToolResult{}, err
		}
		result["delivery"] = delivery
	}
	if job.Status == "preview_ready" && job.PreviewReady {
		attachment, err := s.shortsPreviewAttachment(ctx, job)
		if err != nil {
			return registeredToolResult{}, err
		}
		attachments = append(attachments, *attachment)
		result["preview"] = attachment
	}
	raw, err := json.Marshal(result)
	final := ""
	switch job.Status {
	case "completed":
		if job.OutputReady {
			final = "검수한 영상을 첨부했습니다."
			var applied struct {
				Voice    string  `json:"voice_preset"`
				Duration float64 `json:"duration_seconds"`
			}
			if json.Unmarshal(job.Applied, &applied) == nil {
				if applied.Voice != "" {
					final += " 음성: " + applied.Voice + "."
				}
				if applied.Duration > 0 {
					final += fmt.Sprintf(" 길이: %.1f초.", applied.Duration)
				}
			}
		}
	case "preview_ready":
		final = "미리보기 검사가 끝났습니다. 확인용 이미지를 첨부했습니다."
	case "failed":
		final = "영상 제작에 실패했습니다. " + job.Error
	case "canceled":
		final = "영상 제작을 중지했습니다."
	default:
		final = "영상 제작 작업이 진행 중입니다. " + job.Stage
	}
	return registeredToolResult{Result: string(raw), Attachments: attachments, FinalContent: final}, err
}

// Waiting is tied to the chat connection; the durable job is not. Yield promptly
// when new chat input arrives so natural-language status/cancel can run.
func (s *Server) waitForShorts(ctx context.Context, sessionID, callID string, job shortsJob, emit eventEmitter) (registeredToolResult, error) {
	defer func() {
		if errors.Is(context.Cause(ctx), errChatStopped) && (job.Status == "queued" || job.Status == "running") {
			stopCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			_, _, _ = s.shortsCall(stopCtx, "POST", "/v1/projects/"+job.ID+"/cancel", nil)
		}
	}()
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	lastStage := ""
	for {
		if emit != nil {
			step, total := int(max(0, min(100, job.Progress))), 100
			elapsed := 0.0
			if job.Created > 0 {
				elapsed = max(0, float64(time.Now().UnixMilli())/1000-job.Created)
			}
			payload := map[string]any{"id": callID, "stream": "stdout", "progress": generationProgress{Kind: "shorts", Stage: job.Status, Detail: job.Stage, Step: &step, Total: &total, Elapsed: elapsed}}
			if job.Stage != lastStage {
				payload["delta"] = fmt.Sprintf("[%s] %s · %d%%\n", progressDuration(elapsed), job.Stage, step)
				lastStage = job.Stage
			}
			if err := emit("tool_output", payload); err != nil {
				return registeredToolResult{}, err
			}
		}
		if job.Status != "queued" && job.Status != "running" {
			return s.shortsJobResult(ctx, job)
		}
		if turnFrom(ctx).hasPending() {
			return s.shortsJobResult(ctx, job)
		}
		select {
		case <-ctx.Done():
			return registeredToolResult{}, ctx.Err()
		case <-ticker.C:
		}
		next, err := s.conversationShortsJob(ctx, sessionID, job.ID)
		if err != nil {
			// Keep the job ID available for a later status query, not a duplicate create.
			raw, _ := json.Marshal(map[string]any{"job": job, "monitor_error": err.Error(), "guidance": "The durable job may still be running. Use shorts_manage to check it; do not create another job."})
			return registeredToolResult{Result: string(raw)}, err
		}
		job = next
	}
}
