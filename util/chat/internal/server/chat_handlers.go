package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"unicode/utf8"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
)

func (s *Server) chat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	var req struct {
		SessionID       string          `json:"session_id"`
		Content         string          `json:"content"`
		Model           string          `json:"model"`
		ReasoningEffort string          `json:"reasoning_effort"`
		ToolsEnabled    bool            `json:"tools_enabled"`
		Attachments     []db.Attachment `json:"attachments"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&req); err != nil {
		http.Error(w, "invalid request", 400)
		return
	}
	req.Content = strings.TrimSpace(req.Content)
	if req.SessionID == "" || req.Content == "" {
		http.Error(w, "session_id and content are required", 400)
		return
	}
	cfg, client := s.snapshot()
	if req.Model == "" {
		req.Model = cfg.Model.DefaultModel
	}
	if req.ReasoningEffort == "" {
		req.ReasoningEffort = cfg.Model.ReasoningEffort
	}
	attachments, err := s.media.Validate(req.Attachments)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	count, _ := s.db.CompletedUserMessageCount(req.SessionID)
	pending, err := s.db.AddPendingMessage(req.SessionID, req.Content, attachments)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	_ = s.db.UpdateSession(req.SessionID, "", req.Model, req.ReasoningEffort)
	history, err := s.db.Messages(req.SessionID)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", 500)
		return
	}
	emit := func(kind string, payload any) error {
		data, _ := json.Marshal(payload)
		if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", kind, data); err != nil {
			return err
		}
		flusher.Flush()
		return nil
	}
	result, err := s.runContextCompletion(r.Context(), req.SessionID, modelHistory(history, pending.ID), req.Model, req.ReasoningEffort, cfg, client, req.ToolsEnabled, emit)
	if err != nil {
		status := db.MessageFailed
		failure := compactHistoryText(err.Error(), 2000)
		if errors.Is(err, context.Canceled) || errors.Is(r.Context().Err(), context.Canceled) {
			status = db.MessageCancelled
			failure = "사용자가 생성을 중지했습니다."
		}
		_ = s.db.FailPendingTurn(pending.ID, status, failure, result.Content, result.Reasoning, result.ToolTrace)
		payload, _ := json.Marshal(map[string]string{"error": failure})
		fmt.Fprintf(w, "event: error\ndata: %s\n\n", payload)
	} else {
		if _, completeErr := s.db.CompletePendingTurn(pending.ID, result.Content, result.Reasoning, result.ToolTrace); completeErr != nil {
			_ = s.db.FailPendingTurn(pending.ID, db.MessageFailed, compactHistoryText(completeErr.Error(), 2000), result.Content, result.Reasoning, result.ToolTrace)
			payload, _ := json.Marshal(map[string]string{"error": completeErr.Error()})
			fmt.Fprintf(w, "event: error\ndata: %s\n\n", payload)
			err = completeErr
		} else {
			fmt.Fprint(w, "event: done\ndata: {}\n\n")
		}
	}
	flusher.Flush()

	if err == nil && count == 0 {
		userText, sessionID, model := req.Content, req.SessionID, req.Model
		go s.generateTitle(client, sessionID, model, userText)
	}
}

func (s *Server) messageAction(w http.ResponseWriter, r *http.Request) {
	rest := strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/messages/"), "/")
	parts := strings.Split(rest, "/")
	if r.Method != http.MethodPost || len(parts) != 2 || (parts[1] != "retry" && parts[1] != "edit") {
		http.NotFound(w, r)
		return
	}
	messageID, err := strconv.ParseInt(parts[0], 10, 64)
	if err != nil {
		http.Error(w, "invalid message id", http.StatusBadRequest)
		return
	}
	if parts[1] == "edit" {
		s.editMessage(w, r, messageID)
		return
	}
	var req struct {
		Model           string `json:"model"`
		ReasoningEffort string `json:"reasoning_effort"`
		ToolsEnabled    bool   `json:"tools_enabled"`
		UserVariant     *int   `json:"user_variant"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&req); err != nil {
		http.Error(w, "invalid request", http.StatusBadRequest)
		return
	}
	userVariant := -1
	if req.UserVariant != nil {
		userVariant = *req.UserVariant
	}
	target, history, err := s.db.RetryContext(messageID, userVariant)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	cfg, client := s.snapshot()
	if req.Model == "" {
		req.Model = cfg.Model.DefaultModel
	}
	if req.ReasoningEffort == "" {
		req.ReasoningEffort = cfg.Model.ReasoningEffort
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", 500)
		return
	}
	emit := func(kind string, payload any) error {
		data, _ := json.Marshal(payload)
		if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", kind, data); err != nil {
			return err
		}
		flusher.Flush()
		return nil
	}
	result, err := s.runContextCompletion(r.Context(), target.SessionID, modelHistory(history, 0), req.Model, req.ReasoningEffort, cfg, client, req.ToolsEnabled, emit)
	if err == nil {
		err = s.db.ReplaceAssistant(target.ID, result.Content, result.Reasoning, result.ToolTrace, userVariant)
		_ = s.db.UpdateSession(target.SessionID, "", req.Model, req.ReasoningEffort)
	}
	if err != nil {
		payload, _ := json.Marshal(map[string]string{"error": err.Error()})
		fmt.Fprintf(w, "event: error\ndata: %s\n\n", payload)
	} else {
		fmt.Fprint(w, "event: done\ndata: {}\n\n")
	}
	flusher.Flush()
}

func (s *Server) editMessage(w http.ResponseWriter, r *http.Request, messageID int64) {
	var req struct {
		Content         string           `json:"content"`
		Model           string           `json:"model"`
		ReasoningEffort string           `json:"reasoning_effort"`
		ToolsEnabled    bool             `json:"tools_enabled"`
		Attachments     *[]db.Attachment `json:"attachments"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&req); err != nil {
		http.Error(w, "invalid request", http.StatusBadRequest)
		return
	}
	req.Content = strings.TrimSpace(req.Content)
	if req.Content == "" {
		http.Error(w, "content is required", http.StatusBadRequest)
		return
	}
	target, _, history, err := s.db.EditContext(messageID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	cfg, client := s.snapshot()
	if req.Model == "" {
		req.Model = cfg.Model.DefaultModel
	}
	if req.ReasoningEffort == "" {
		req.ReasoningEffort = cfg.Model.ReasoningEffort
	}
	attachments := target.Attachments
	if req.Attachments != nil {
		attachments, err = s.media.Validate(*req.Attachments)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
	}
	requestHistory := append(append([]db.Message{}, history...), db.Message{Role: "user", Content: req.Content, Attachments: attachments})
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	emit := func(kind string, payload any) error {
		data, _ := json.Marshal(payload)
		if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", kind, data); err != nil {
			return err
		}
		flusher.Flush()
		return nil
	}
	result, err := s.runContextCompletion(r.Context(), target.SessionID, modelHistory(requestHistory, 0), req.Model, req.ReasoningEffort, cfg, client, req.ToolsEnabled, emit)
	if err == nil {
		err = s.db.AppendEditedBranch(messageID, req.Content, attachments, result.Content, result.Reasoning, result.ToolTrace)
		_ = s.db.UpdateSession(target.SessionID, "", req.Model, req.ReasoningEffort)
	}
	if err != nil {
		payload, _ := json.Marshal(map[string]string{"error": err.Error()})
		fmt.Fprintf(w, "event: error\ndata: %s\n\n", payload)
	} else {
		fmt.Fprint(w, "event: done\ndata: {}\n\n")
		if len(history) == 0 {
			go s.generateTitle(client, target.SessionID, req.Model, req.Content)
		}
	}
	flusher.Flush()
}

func (s *Server) generateTitle(client *llm.Client, sessionID, model, userText string) {
	title, err := client.GenerateTitle(context.Background(), model, userText)
	if err != nil || title == "" {
		title = fallbackTitle(userText)
	}
	_ = s.db.UpdateSessionTitle(sessionID, title)
}

func (s *Server) llmMessages(ctx context.Context, items []db.Message, cfg config.Config) ([]llm.Message, error) {
	messages := make([]llm.Message, 0, len(items))
	for _, item := range items {
		if len(item.Attachments) == 0 {
			messages = append(messages, llm.Message{Role: item.Role, Content: item.Content})
			continue
		}
		parts := make([]map[string]any, 0, len(item.Attachments)+1)
		textParts := []string{item.Content}
		for _, attachment := range item.Attachments {
			isAudio := strings.HasPrefix(attachment.MIME, "audio/")
			isVideo := strings.HasPrefix(attachment.MIME, "video/")
			if !isAudio {
				dataURL, err := s.media.DataURL(attachment)
				if err != nil {
					return nil, fmt.Errorf("read media %s: %w", attachment.Name, err)
				}
				typeName, fieldName := "image_url", "image_url"
				if isVideo {
					typeName, fieldName = "video_url", "video_url"
				}
				parts = append(parts, map[string]any{"type": typeName, fieldName: map[string]string{"url": dataURL}})
			}
			if !isAudio && !isVideo {
				continue
			}
			if !cfg.ASR.Enabled {
				if isAudio {
					return nil, fmt.Errorf("음성 첨부를 처리하려면 설정에서 ASR을 활성화해야 합니다")
				}
				continue
			}
			cached, err := s.transcribeAttachment(ctx, attachment, cfg.ASR)
			if err != nil {
				if isVideo && isNoAudio(err) {
					continue
				}
				return nil, err
			}
			textParts = append(textParts, transcriptBlock(attachment, cached))
		}
		parts = append(parts, map[string]any{"type": "text", "text": strings.Join(textParts, "\n\n")})
		messages = append(messages, llm.Message{Role: item.Role, Content: parts})
	}
	return messages, nil
}

func fallbackTitle(text string) string {
	text = strings.Join(strings.Fields(text), " ")
	if utf8.RuneCountInString(text) <= 28 {
		return text
	}
	return string([]rune(text)[:28]) + "…"
}
