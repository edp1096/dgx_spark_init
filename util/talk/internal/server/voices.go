package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"path/filepath"
	"regexp"
	"strings"
	"time"
	"unicode/utf8"

	"sparktalk/internal/asr"
	"sparktalk/internal/db"
)

var customVoiceID = regexp.MustCompile(`^voice_[a-f0-9]{32}$`)
var builtinVoiceID = regexp.MustCompile(`^(sohee|uncle_fu|vivian|serena|dylan|eric|ryan|aiden|ono_anna)$`)

func validVoiceID(id string) bool {
	return customVoiceID.MatchString(id) || builtinVoiceID.MatchString(id)
}

type voiceRequest struct {
	SessionID          string         `json:"session_id"`
	Action             string         `json:"action"`
	VoiceID            string         `json:"voice_id,omitempty"`
	TargetVoiceID      string         `json:"target_voice_id,omitempty"`
	Revision           *int           `json:"revision,omitempty"`
	JobID              string         `json:"job_id,omitempty"`
	Name               string         `json:"name,omitempty"`
	Scope              string         `json:"scope,omitempty"`
	Text               string         `json:"text,omitempty"`
	Instructions       string         `json:"instructions,omitempty"`
	Continuous         bool           `json:"continuous,omitempty"`
	SourceAttachmentID string         `json:"source_attachment_id,omitempty"`
	SourceURL          string         `json:"source_url,omitempty"`
	SourceJobID        string         `json:"source_job_id,omitempty"`
	Start              *float64       `json:"start,omitempty"`
	End                *float64       `json:"end,omitempty"`
	Track              *int           `json:"track,omitempty"`
	Channel            *int           `json:"channel,omitempty"`
	Candidate          *int           `json:"candidate,omitempty"`
	Transcript         string         `json:"transcript,omitempty"`
	RequestID          string         `json:"request_id,omitempty"`
	Audio              string         `json:"audio,omitempty"`
	Changes            map[string]any `json:"changes,omitempty"`
}

type voiceFile struct {
	File  string `json:"file"`
	Name  string `json:"name"`
	SHA   string `json:"sha256"`
	Bytes int64  `json:"bytes"`
}
type voiceJob struct {
	ID              string          `json:"id"`
	SessionID       string          `json:"session_id"`
	Status          string          `json:"status"`
	Stage           string          `json:"stage"`
	Progress        float64         `json:"progress"`
	Version         int             `json:"version"`
	Created         float64         `json:"created"`
	Error           string          `json:"error"`
	DeliveryPending bool            `json:"delivery_pending"`
	Request         json.RawMessage `json:"request"`
	Result          struct {
		Question    string      `json:"question"`
		Registered  bool        `json:"registered"`
		Temporary   bool        `json:"temporary"`
		Warnings    []string    `json:"warnings"`
		Limitations []string    `json:"limitations"`
		Files       []voiceFile `json:"files"`
		Voice       struct {
			ID       string `json:"id"`
			Name     string `json:"name"`
			Scope    string `json:"scope"`
			Status   string `json:"status"`
			Revision int    `json:"revision"`
		} `json:"voice"`
		Reference struct {
			Start            float64 `json:"start"`
			End              float64 `json:"end"`
			Transcript       string  `json:"transcript"`
			TranscriptSource string  `json:"transcript_source"`
		} `json:"reference"`
		Candidates []struct {
			Index             int     `json:"index"`
			Start             float64 `json:"start"`
			End               float64 `json:"end"`
			Transcript        string  `json:"transcript"`
			SpeakerCount      int     `json:"speaker_count"`
			DiarizationStatus string  `json:"diarization_status"`
		} `json:"candidates"`
		Source struct {
			Streams []struct {
				Index    int    `json:"index"`
				Channels int    `json:"channels"`
				Codec    string `json:"codec"`
			} `json:"streams"`
		} `json:"source"`
		Generation struct {
			Text         string `json:"text"`
			Input        string `json:"model_input"`
			Instructions string `json:"requested_instructions"`
			Validation   struct {
				Passed   bool            `json:"passed"`
				Exact    bool            `json:"exact"`
				CER      float64         `json:"character_error_rate"`
				Recovery json.RawMessage `json:"diagnostic_recovery"`
			} `json:"validation"`
		} `json:"generation"`
	} `json:"result"`
}

func (s *Server) voiceSources(ctx context.Context, session string) ([]conversationAttachment, error) {
	all, err := s.conversationAttachments(ctx, session)
	if err != nil {
		return nil, err
	}
	items := []conversationAttachment{}
	for _, x := range all {
		if strings.HasPrefix(x.Attachment.MIME, "audio/") || strings.HasPrefix(x.Attachment.MIME, "video/") {
			items = append(items, x)
		}
	}
	return items, nil
}

func (s *Server) voiceCapabilities(ctx context.Context, session string) (map[string]any, error) {
	if err := s.ensureShorts(ctx); err != nil {
		return nil, err
	}
	raw, code, err := s.shortsCall(ctx, "POST", "/v1/voice-tools", map[string]any{"session_id": session, "action": "capabilities"})
	if err != nil {
		return nil, err
	}
	if code != 200 {
		return nil, fmt.Errorf("음성 기능 조회 HTTP %d: %s", code, raw)
	}
	var result map[string]any
	if err = json.Unmarshal(raw, &result); err != nil {
		return nil, err
	}
	cfg, _ := s.snapshot()
	result["generation_enabled"] = !cfg.SupportDisabled("fish")
	if cfg.SupportDisabled("fish") {
		result["generation_disabled_reason"] = "현재 모델 세트에서는 Fish 음성 생성을 제외했습니다"
	}
	result["attachment_limits"] = s.media.Limits()
	return result, nil
}

func (s *Server) createVoiceJob(ctx context.Context, a voiceRequest) (voiceJob, error) {
	var empty voiceJob
	cfg, _ := s.snapshot()
	if cfg.SupportDisabled("fish") {
		return empty, fmt.Errorf("현재 모델 세트에서는 Fish 음성 생성을 제외했습니다")
	}
	if a.Action != "prepare" && a.Action != "generate" {
		return empty, fmt.Errorf("잘못된 음성 작업")
	}
	if a.Action == "prepare" && a.VoiceID != "" {
		return empty, fmt.Errorf("참조 재설정에는 target_voice_id와 원본을 지정해 주세요. voice_id는 저장된 음성으로 대사를 생성할 때 사용합니다")
	}
	sourceCount := 0
	for _, source := range []string{a.SourceAttachmentID, a.SourceURL, a.SourceJobID} {
		if source != "" {
			sourceCount++
		}
	}
	if sourceCount > 1 {
		return empty, fmt.Errorf("새 파일, 새 URL, 이전 작업 원본 중 하나만 지정해 주세요")
	}
	if a.Scope == "" {
		a.Scope = "temporary"
		if a.Action == "prepare" {
			a.Scope = "persistent"
		}
	}
	if a.Scope != "persistent" && a.Scope != "temporary" {
		return empty, fmt.Errorf("등록 범위는 persistent 또는 temporary입니다")
	}
	if a.VoiceID != "" && !validVoiceID(a.VoiceID) {
		return empty, fmt.Errorf("잘못된 음성 ID")
	}
	if utf8.RuneCountInString(a.Text) > 500 || utf8.RuneCountInString(a.Instructions) > 200 {
		return empty, fmt.Errorf("대사는 500자, 전체 지시는 200자 이하여야 합니다")
	}
	if a.Action == "generate" && strings.TrimSpace(a.Text) == "" {
		return empty, fmt.Errorf("읽을 대사를 입력해 주세요")
	}
	if a.SourceAttachmentID != "" && a.SourceURL != "" {
		return empty, fmt.Errorf("참조 파일과 URL 중 하나를 지정해 주세요")
	}
	if err := s.ensureShorts(ctx); err != nil {
		return empty, err
	}
	if a.Action == "prepare" && a.TargetVoiceID != "" && strings.TrimSpace(a.Name) == "" {
		raw, code, err := s.shortsCall(ctx, "POST", "/v1/voice-tools", map[string]any{"session_id": a.SessionID, "action": "inspect", "voice_id": a.TargetVoiceID})
		if err != nil {
			return empty, err
		}
		var current struct {
			Name string `json:"name"`
		}
		if code != http.StatusOK || json.Unmarshal(raw, &current) != nil || current.Name == "" {
			return empty, fmt.Errorf("참조를 변경할 등록 음성을 찾을 수 없습니다")
		}
		a.Name = current.Name
	}
	var source *db.Attachment
	if a.SourceURL != "" {
		item, err := s.importMediaSource(ctx, a.SourceURL)
		if err != nil {
			return empty, err
		}
		source = &item
	} else if a.SourceAttachmentID != "" {
		sources, err := s.voiceSources(ctx, a.SessionID)
		if err != nil {
			return empty, err
		}
		for _, item := range sources {
			if item.Attachment.ID == a.SourceAttachmentID {
				v := item.Attachment
				source = &v
				break
			}
		}
		if source == nil {
			return empty, fmt.Errorf("이 대화에 첨부된 음성·영상 파일을 지정해 주세요")
		}
	}
	if source != nil && !strings.HasPrefix(source.MIME, "audio/") && !strings.HasPrefix(source.MIME, "video/") {
		return empty, fmt.Errorf("오디오 트랙이 있는 파일을 지정해 주세요")
	}
	if a.Action == "prepare" && a.Scope == "persistent" && strings.TrimSpace(a.Name) == "" {
		if source != nil {
			a.Name = strings.TrimSuffix(source.Name, filepath.Ext(source.Name))
		} else {
			a.Name = "내 음성"
		}
		a.Name = string([]rune(a.Name)[:min(80, len([]rune(a.Name)))])
	}
	if a.Action == "generate" && a.VoiceID == "" && source == nil && a.SourceJobID == "" {
		caps, err := s.voiceCapabilities(ctx, a.SessionID)
		if err != nil {
			return empty, err
		}
		a.VoiceID, _ = caps["selected_voice"].(string)
		if a.VoiceID == "" {
			a.VoiceID = "sohee"
		}
	}
	var messageID int64
	if history, err := s.db.Messages(a.SessionID); err == nil {
		for _, m := range history {
			if m.Role == "user" && m.ID > messageID {
				messageID = m.ID
			}
		}
	}
	identity := a
	identity.RequestID = ""
	identity.Audio = ""
	identity.Changes = nil
	keyJSON, _ := json.Marshal(identity)
	if a.RequestID == "" {
		sum := sha256.Sum256(append([]byte(fmt.Sprintf("%s:%d:", a.SessionID, messageID)), keyJSON...))
		a.RequestID = hex.EncodeToString(sum[:])
	}
	payload := map[string]any{"session_id": a.SessionID, "action": a.Action, "name": a.Name, "scope": a.Scope, "text": a.Text, "instructions": a.Instructions, "source_url": a.SourceURL, "source_job_id": a.SourceJobID, "transcript": a.Transcript, "idempotency_key": a.RequestID}
	payload["continuous"] = a.Continuous
	if a.VoiceID != "" {
		payload["voice_id"] = a.VoiceID
	}
	if a.TargetVoiceID != "" {
		payload["target_voice_id"] = a.TargetVoiceID
	}
	if a.Revision != nil {
		payload["revision"] = *a.Revision
	}
	if a.Start != nil {
		payload["start"] = *a.Start
	}
	if a.End != nil {
		payload["end"] = *a.End
	}
	if a.Track != nil {
		payload["track"] = *a.Track
	}
	if a.Channel != nil {
		payload["channel"] = *a.Channel
	}
	if a.Candidate != nil {
		payload["candidate"] = *a.Candidate
	}
	var raw []byte
	var code int
	var err error
	if source == nil {
		raw, code, err = s.shortsCall(ctx, "POST", "/v1/voice-jobs", payload)
	} else {
		payload["source_attachment"] = source
		payload["source_name"] = source.Name
		payload["source_mime"] = source.MIME
		f, e := s.media.Open(*source)
		if e != nil {
			return empty, e
		}
		defer f.Close()
		info, e := f.Stat()
		if e != nil {
			return empty, e
		}
		metadata, _ := json.Marshal(payload)
		var header bytes.Buffer
		writer := multipart.NewWriter(&header)
		_ = writer.WriteField("metadata", string(metadata))
		if _, e = writer.CreateFormFile("file", "source.bin"); e != nil {
			return empty, e
		}
		prefix := append([]byte(nil), header.Bytes()...)
		header.Reset()
		if e = writer.Close(); e != nil {
			return empty, e
		}
		suffix := append([]byte(nil), header.Bytes()...)
		request, e := http.NewRequestWithContext(ctx, "POST", s.shortsEndpoint()+"/v1/voice-jobs/upload", io.MultiReader(bytes.NewReader(prefix), f, bytes.NewReader(suffix)))
		if e != nil {
			return empty, e
		}
		request.ContentLength = int64(len(prefix)) + info.Size() + int64(len(suffix))
		request.Header.Set("Content-Type", writer.FormDataContentType())
		response, e := (&http.Client{Timeout: 10 * time.Minute}).Do(request)
		if e != nil {
			return empty, e
		}
		defer response.Body.Close()
		code = response.StatusCode
		raw, err = io.ReadAll(io.LimitReader(response.Body, 4<<20))
	}
	if err != nil {
		return empty, err
	}
	if code != 200 {
		return empty, fmt.Errorf("음성 작업 시작 HTTP %d: %s", code, raw)
	}
	var job voiceJob
	if err = json.Unmarshal(raw, &job); err != nil || !validShortsJobID(job.ID) || job.SessionID != a.SessionID {
		return empty, fmt.Errorf("잘못된 음성 작업 응답")
	}
	return job, nil
}

func (s *Server) conversationVoiceJob(ctx context.Context, session, id string) (voiceJob, error) {
	var job voiceJob
	path := "/v1/voice-jobs?session_id=" + url.QueryEscape(session)
	if id != "" {
		if !validShortsJobID(id) {
			return job, fmt.Errorf("잘못된 음성 작업 ID")
		}
		path = "/v1/voice-jobs/" + id + "?session_id=" + url.QueryEscape(session)
	}
	raw, code, err := s.shortsCall(ctx, "GET", path, nil)
	if err != nil {
		return job, err
	}
	if code != 200 {
		return job, fmt.Errorf("이 대화의 음성 작업을 찾을 수 없습니다")
	}
	if id == "" {
		var jobs []voiceJob
		if json.Unmarshal(raw, &jobs) != nil || len(jobs) == 0 {
			return job, fmt.Errorf("이 대화에 음성 작업이 없습니다")
		}
		job = jobs[0]
	} else if json.Unmarshal(raw, &job) != nil {
		return job, fmt.Errorf("잘못된 음성 작업 응답")
	}
	if job.SessionID != session || !validShortsJobID(job.ID) {
		return voiceJob{}, fmt.Errorf("다른 대화의 음성 작업입니다")
	}
	return job, nil
}

func voiceResultText(job voiceJob) string {
	var text strings.Builder
	switch job.Status {
	case "completed":
		if job.Result.Registered {
			fmt.Fprintf(&text, "‘%s’ 음성을 영구 등록했습니다. 음성 ID: `%s`, 참조 버전: %d.\n", job.Result.Voice.Name, job.Result.Voice.ID, job.Result.Voice.Revision)
		} else if job.Result.Temporary {
			fmt.Fprintf(&text, "임시 참조 음성으로 준비했습니다. 영구 음성 목록에는 등록하지 않았습니다. 음성 ID: `%s`.\n", job.Result.Voice.ID)
		} else {
			fmt.Fprintf(&text, "‘%s’ 음성으로 생성했습니다.\n", job.Result.Voice.Name)
		}
		if job.Result.Reference.End > job.Result.Reference.Start {
			fmt.Fprintf(&text, "\n사용한 원본 구간: %.2f–%.2f초.\n참조 대사: %s\n", job.Result.Reference.Start, job.Result.Reference.End, job.Result.Reference.Transcript)
		}
		fmt.Fprintf(&text, "\n생성 대사:\n%s\n", job.Result.Generation.Text)
		if job.Result.Generation.Instructions != "" {
			fmt.Fprintf(&text, "\n모델에 전달한 말투 지시: %s\n", job.Result.Generation.Instructions)
		}
		if job.Result.Generation.Validation.Exact {
			text.WriteString("\nASR 대사는 원문과 일치했습니다.\n")
		} else {
			fmt.Fprintf(&text, "\n최초 ASR 문자 오류율: %.1f%%. 인식 오류와 실제 발음·누락 오류는 구분해야 합니다.\n", 100*job.Result.Generation.Validation.CER)
		}
		if len(job.Result.Generation.Validation.Recovery) > 0 && string(job.Result.Generation.Validation.Recovery) != "null" {
			text.WriteString("조용한 구간을 따로 검사해 대사를 확인했습니다. 첨부 음성은 보정하지 않은 생성 결과입니다.\n")
		}
		fmt.Fprintf(&text, "\n다음에는 ‘%s로 다음 문장을 읽어줘: …’라고 요청하세요.\n", job.Result.Voice.Name)
		if job.Result.Temporary {
			text.WriteString("영구 저장하려면 ‘이 음성을 이름을 붙여 저장해’라고 요청하세요. 임시 참조와 대화 첨부파일의 보관은 별개입니다.\n")
		}
	case "needs_input":
		text.WriteString(job.Result.Question + "\n")
		for _, candidate := range job.Result.Candidates {
			fmt.Fprintf(&text, "\n후보 %d: %.2f–%.2f초 · %s\n", candidate.Index+1, candidate.Start, candidate.End, candidate.Transcript)
		}
		for _, stream := range job.Result.Source.Streams {
			fmt.Fprintf(&text, "\n오디오 트랙 %d: %s, %d채널\n", stream.Index, stream.Codec, stream.Channels)
		}
		text.WriteString("\n후보 번호를 고르거나 시작·끝 시간, 채널 또는 실제 참조 대사를 알려 주세요. 아직 등록 완료 상태는 아닙니다.\n")
	case "failed":
		text.WriteString("음성 처리에 실패했습니다. " + job.Error + "\n첨부가 있다면 검사 실패를 확인하기 위한 샘플입니다. 정상 완료 결과로 표시하지 않았습니다.\n")
	case "canceled":
		text.WriteString("음성 작업을 중지했습니다.\n")
	default:
		text.WriteString("음성 작업이 진행 중입니다. " + job.Stage + "\n")
	}
	for _, warning := range job.Result.Warnings {
		text.WriteString("\n- " + warning)
	}
	if job.Status == "completed" {
		text.WriteString("\n\n샘플을 참조한 생성이며 모델 재학습은 아닙니다. 대사 확인은 음색의 동일성이나 감정·태그 강도의 일관성을 보장하지 않습니다.")
	}
	fmt.Fprintf(&text, "\n\n작업 ID: `%s`", job.ID)
	return text.String()
}

func (s *Server) publishVoice(ctx context.Context, session, id string) (*db.Message, error) {
	s.voicePublishMu.Lock()
	defer s.voicePublishMu.Unlock()
	job, err := s.conversationVoiceJob(ctx, session, id)
	if err != nil {
		return nil, err
	}
	switch job.Status {
	case "completed", "needs_input", "failed", "canceled":
	default:
		return nil, fmt.Errorf("아직 게시할 음성 결과가 없습니다")
	}
	existing, err := s.db.VoicePublication(session, job.ID, job.Version)
	if err != nil || existing != nil {
		return existing, err
	}
	if len(job.Result.Files) > 8 {
		return nil, fmt.Errorf("음성 결과 파일 수가 너무 많습니다")
	}
	attachments := []db.Attachment{}
	for _, file := range job.Result.Files {
		if !regexp.MustCompile(`^(candidate-[0-9]|reference|preview|output|rejected-[12])\.wav$`).MatchString(file.File) || file.Bytes < 1 || file.Bytes > s.media.Limits().ForMIME("audio/wav") || len(file.SHA) != 64 {
			return nil, fmt.Errorf("잘못된 음성 결과 파일 정보")
		}
		request, err := http.NewRequestWithContext(ctx, "GET", s.shortsEndpoint()+"/v1/voice-jobs/"+job.ID+"/files/"+file.File+"?session_id="+url.QueryEscape(session), nil)
		if err != nil {
			return nil, err
		}
		response, err := http.DefaultClient.Do(request)
		if err != nil {
			return nil, err
		}
		if response.StatusCode != 200 {
			response.Body.Close()
			return nil, fmt.Errorf("음성 결과 파일을 읽지 못했습니다")
		}
		hash := sha256.New()
		a, saveErr := s.media.SaveReader(io.TeeReader(io.LimitReader(response.Body, file.Bytes+1), hash), file.Name, "audio/wav", s.media.Limits().ForMIME("audio/wav"))
		response.Body.Close()
		if saveErr != nil {
			return nil, saveErr
		}
		if a.Size != file.Bytes || hex.EncodeToString(hash.Sum(nil)) != file.SHA {
			return nil, fmt.Errorf("음성 결과 파일 검증 실패")
		}
		a.SourceURL = fmt.Sprintf("/api/voices/jobs/%s/files/%s?session_id=%s&version=%d", job.ID, file.File, url.QueryEscape(session), job.Version)
		attachments = append(attachments, a)
	}
	current, err := s.conversationVoiceJob(ctx, session, id)
	if err != nil {
		return nil, err
	}
	if current.Version != job.Version || current.Status != job.Status {
		return nil, fmt.Errorf("음성 작업 상태가 바뀌었습니다. 최신 결과를 다시 게시합니다")
	}
	message, err := s.db.PublishVoice(session, job.ID, job.Version, voiceResultText(job), attachments)
	if err != nil {
		return nil, err
	}
	_, _, _ = s.shortsCall(ctx, "POST", "/v1/voice-jobs/"+job.ID+"/published", map[string]any{"session_id": session, "version": job.Version})
	return message, nil
}

func (s *Server) transcribeVoiceReference(ctx context.Context, a voiceRequest) (any, error) {
	job, err := s.conversationVoiceJob(ctx, a.SessionID, a.JobID)
	if err != nil {
		return nil, err
	}
	if job.Status != "running" {
		return nil, fmt.Errorf("실행 중인 음성 작업이 아닙니다")
	}
	data, err := base64.StdEncoding.DecodeString(a.Audio)
	if err != nil || len(data) == 0 || len(data) > 48<<20 {
		return nil, fmt.Errorf("잘못된 참조 음성 검사 입력")
	}
	client, release, err := s.prepareASRWorkload(ctx, s.asrSnapshot())
	if err != nil {
		return nil, err
	}
	result, err := client.TranscribeReference(ctx, bytes.NewReader(data), "reference.wav", "audio/wav")
	noText := errors.Is(err, asr.ErrNoText)
	if noText {
		err = nil
	}
	cleanup := release(err)
	if err != nil {
		return nil, err
	}
	if cleanup != nil {
		return nil, cleanup
	}
	if noText {
		return asr.Result{Text: "", DiarizationStatus: "unavailable", Warning: "선택 구간의 대사를 인식하지 못했습니다."}, nil
	}
	return result, nil
}

func (s *Server) voicesAPI(w http.ResponseWriter, r *http.Request) {
	path := strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/voices"), "/")
	var a voiceRequest
	if r.Method == "POST" {
		decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<20))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&a); err != nil {
			http.Error(w, "잘못된 음성 요청", 400)
			return
		}
	} else {
		a.SessionID = r.URL.Query().Get("session_id")
	}
	cfg, _ := s.snapshot()
	if r.Method == "POST" && cfg.SupportDisabled("fish") && (a.Action == "prepare" || a.Action == "generate" || a.Action == "retry" || a.Action == "choose") {
		http.Error(w, "현재 모델 세트에서는 Fish 음성 생성을 제외했습니다", http.StatusConflict)
		return
	}
	if _, err := s.db.Session(a.SessionID); err != nil {
		http.Error(w, "대화가 없습니다", 404)
		return
	}
	if err := s.ensureShorts(r.Context()); err != nil {
		http.Error(w, err.Error(), 503)
		return
	}
	var result any
	var err error
	switch {
	case path == "publish" && r.Method == "POST":
		result, err = s.publishVoice(r.Context(), a.SessionID, a.JobID)
	case path == "transcribe" && r.Method == "POST":
		result, err = s.transcribeVoiceReference(r.Context(), a)
	case path == "" && r.Method == "GET":
		result, err = s.voiceCapabilities(r.Context(), a.SessionID)
	case path == "" && r.Method == "POST":
		switch a.Action {
		case "prepare", "generate":
			result, err = s.createVoiceJob(r.Context(), a)
		case "help", "capabilities", "list":
			result, err = s.voiceCapabilities(r.Context(), a.SessionID)
		case "inspect", "rename", "delete", "promote":
			raw, code, e := s.shortsCall(r.Context(), "POST", "/v1/voice-tools", map[string]any{"session_id": a.SessionID, "action": a.Action, "voice_id": a.VoiceID, "name": a.Name})
			shortsReply(w, raw, code, e)
			return
		default:
			http.Error(w, "지원하지 않는 음성 요청", 400)
			return
		}
	case path == "jobs" && r.Method == "GET":
		raw, code, e := s.shortsCall(r.Context(), "GET", "/v1/voice-jobs?session_id="+url.QueryEscape(a.SessionID), nil)
		shortsReply(w, raw, code, e)
		return
	default:
		parts := strings.Split(path, "/")
		if len(parts) < 2 || parts[0] != "jobs" || !validShortsJobID(parts[1]) {
			http.NotFound(w, r)
			return
		}
		job, e := s.conversationVoiceJob(r.Context(), a.SessionID, parts[1])
		if e != nil {
			http.Error(w, e.Error(), 404)
			return
		}
		if len(parts) == 2 && r.Method == "GET" {
			result = job
		} else if len(parts) == 3 && parts[2] == "action" && r.Method == "POST" {
			raw, code, e := s.shortsCall(r.Context(), "POST", "/v1/voice-jobs/"+job.ID+"/action", map[string]any{"session_id": a.SessionID, "action": a.Action, "changes": a.Changes})
			shortsReply(w, raw, code, e)
			return
		} else if len(parts) == 4 && parts[2] == "files" && r.Method == "GET" {
			if !regexp.MustCompile(`^(source\.bin|(candidate-[0-9]|reference|preview|output|rejected-[12])\.wav)$`).MatchString(parts[3]) {
				http.NotFound(w, r)
				return
			}
			req, _ := http.NewRequestWithContext(r.Context(), "GET", s.shortsEndpoint()+"/v1/voice-jobs/"+job.ID+"/files/"+parts[3]+"?session_id="+url.QueryEscape(a.SessionID), nil)
			if value := r.Header.Get("Range"); value != "" {
				req.Header.Set("Range", value)
			}
			response, e := http.DefaultClient.Do(req)
			if e != nil {
				http.Error(w, e.Error(), 502)
				return
			}
			defer response.Body.Close()
			for _, key := range []string{"Content-Type", "Content-Length", "Content-Range", "Accept-Ranges", "Content-Disposition"} {
				if v := response.Header.Get(key); v != "" {
					w.Header().Set(key, v)
				}
			}
			w.WriteHeader(response.StatusCode)
			_, _ = io.Copy(w, response.Body)
			return
		} else {
			http.NotFound(w, r)
			return
		}
	}
	if err != nil {
		http.Error(w, err.Error(), 502)
		return
	}
	writeJSON(w, 200, result)
}
