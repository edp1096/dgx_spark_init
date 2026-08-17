package server

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"io/fs"
	"net/http"
	"path"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/media"
)

type Server struct {
	mu         sync.RWMutex
	cfg        config.Config
	startup    config.ServerConfig
	configPath string
	db         *db.DB
	llm        *llm.Client
	media      *media.Store
	server     *http.Server
}

func New(cfg config.Config, configPath string, store *db.DB, client *llm.Client, embedded fs.FS) (*Server, error) {
	web, err := fs.Sub(embedded, "web/dist")
	if err != nil {
		return nil, err
	}
	mediaStore, err := media.New(cfg.Server.Database)
	if err != nil {
		return nil, fmt.Errorf("image storage: %w", err)
	}
	s := &Server{cfg: cfg, startup: cfg.Server, configPath: configPath, db: store, llm: client, media: mediaStore}
	mux := http.NewServeMux()
	mux.HandleFunc("/api/health", s.health)
	mux.HandleFunc("/api/config", s.configuration)
	mux.HandleFunc("/api/models", s.models)
	mux.HandleFunc("/api/images", s.uploadImage)
	mux.HandleFunc("/api/images/", s.image)
	mux.HandleFunc("/api/media", s.mediaUsage)
	mux.HandleFunc("/api/messages/", s.messageAction)
	mux.HandleFunc("/api/groups", s.groups)
	mux.HandleFunc("/api/groups/", s.group)
	mux.HandleFunc("/api/sessions", s.sessions)
	mux.HandleFunc("/api/sessions/", s.session)
	mux.HandleFunc("/api/chat", s.chat)
	mux.Handle("/", spaHandler(web))
	s.server = &http.Server{Addr: cfg.Server.ListenAddr, Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	return s, nil
}

func (s *Server) mediaUsage(w http.ResponseWriter, r *http.Request) {
	referenced, err := s.db.ReferencedAttachmentIDs()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	keep := make(map[string]struct{})
	if r.Method == http.MethodDelete {
		var req struct {
			KeepIDs []string `json:"keep_ids"`
		}
		if r.Body != nil {
			_ = json.NewDecoder(http.MaxBytesReader(w, r.Body, 64<<10)).Decode(&req)
		}
		for _, id := range req.KeepIDs {
			keep[id] = struct{}{}
		}
		removed, err := s.media.Cleanup(referenced, keep)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		usage, err := s.media.Usage(referenced, keep)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"removed": removed, "usage": usage})
		return
	}
	if r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	usage, err := s.media.Usage(referenced, keep)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusOK, usage)
}

func (s *Server) uploadImage(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	r.Body = http.MaxBytesReader(w, r.Body, 16<<20)
	if err := r.ParseMultipartForm(16 << 20); err != nil {
		http.Error(w, "invalid image upload or image is too large", http.StatusBadRequest)
		return
	}
	file, header, err := r.FormFile("image")
	if err != nil {
		http.Error(w, "image is required", http.StatusBadRequest)
		return
	}
	_ = file.Close()
	item, err := s.media.Save(header)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	writeJSON(w, http.StatusCreated, item)
}

func (s *Server) image(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	s.media.Serve(w, r, strings.TrimPrefix(r.URL.Path, "/api/images/"))
}

func (s *Server) ListenAndServe() error { return s.server.ListenAndServe() }

func (s *Server) snapshot() (config.Config, *llm.Client) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.cfg, s.llm
}

func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	cfg, client := s.snapshot()
	model, err := client.Health(r.Context())
	status := "ok"
	if err != nil {
		status = "degraded"
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"status": status, "endpoint": cfg.Model.Endpoint, "model": model, "error": errorText(err),
	})
}

func (s *Server) configuration(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		cfg, _ := s.snapshot()
		writeJSON(w, http.StatusOK, cfg.Public())
	case http.MethodPut:
		var req struct {
			Server      config.ServerConfig `json:"server"`
			Model       config.ModelConfig  `json:"model"`
			Tools       config.ToolsConfig  `json:"tools"`
			APIKey      string              `json:"api_key"`
			ClearAPIKey bool                `json:"clear_api_key"`
		}
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&req); err != nil {
			http.Error(w, "invalid config", http.StatusBadRequest)
			return
		}
		old, _ := s.snapshot()
		next := config.Config{Server: req.Server, Model: req.Model, Tools: req.Tools}
		if req.ClearAPIKey {
			next.Model.APIKey = ""
		} else if req.APIKey != "" {
			next.Model.APIKey = req.APIKey
		} else {
			next.Model.APIKey = old.Model.APIKey
		}
		next.Normalize()
		if err := config.Save(s.configPath, next); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		restartRequired := s.startup != next.Server
		s.mu.Lock()
		s.cfg = next
		s.llm = llm.New(next.Model.Endpoint, next.Model.DefaultModel, next.Model.APIKey)
		s.mu.Unlock()
		writeJSON(w, http.StatusOK, map[string]any{
			"config": next.Public(), "restart_required": restartRequired,
		})
	default:
		methodNotAllowed(w)
	}
}

func (s *Server) models(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	_, client := s.snapshot()
	models, err := client.Models(r.Context())
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	writeJSON(w, http.StatusOK, models)
}

func (s *Server) sessions(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		items, err := s.db.Sessions()
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		writeJSON(w, 200, items)
	case http.MethodPost:
		var req struct {
			Title     string `json:"title"`
			Model     string `json:"model"`
			Reasoning string `json:"reasoning_effort"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		if strings.TrimSpace(req.Title) == "" {
			req.Title = "새 대화"
		}
		cfg, _ := s.snapshot()
		if req.Model == "" {
			req.Model = cfg.Model.DefaultModel
		}
		if req.Reasoning == "" {
			req.Reasoning = cfg.Model.ReasoningEffort
		}
		item, err := s.db.CreateSession(newID(), req.Title, req.Model, req.Reasoning)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		writeJSON(w, 201, item)
	default:
		methodNotAllowed(w)
	}
}

func (s *Server) groups(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		items, err := s.db.Groups()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, http.StatusOK, items)
	case http.MethodPost:
		var req struct {
			Name string `json:"name"`
		}
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 16<<10)).Decode(&req); err != nil {
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		req.Name = strings.TrimSpace(req.Name)
		if req.Name == "" || utf8.RuneCountInString(req.Name) > 60 {
			http.Error(w, "group name must be between 1 and 60 characters", http.StatusBadRequest)
			return
		}
		item, err := s.db.CreateGroup(newID(), req.Name)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, http.StatusCreated, item)
	default:
		methodNotAllowed(w)
	}
}

func (s *Server) group(w http.ResponseWriter, r *http.Request) {
	rest := strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/groups/"), "/")
	parts := strings.Split(rest, "/")
	if len(parts) == 0 || parts[0] == "" {
		http.NotFound(w, r)
		return
	}
	id := parts[0]
	var err error
	switch {
	case len(parts) == 1 && r.Method == http.MethodPatch:
		var req struct {
			Name string `json:"name"`
		}
		if decodeErr := json.NewDecoder(http.MaxBytesReader(w, r.Body, 16<<10)).Decode(&req); decodeErr != nil {
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		req.Name = strings.TrimSpace(req.Name)
		if req.Name == "" || utf8.RuneCountInString(req.Name) > 60 {
			http.Error(w, "group name must be between 1 and 60 characters", http.StatusBadRequest)
			return
		}
		err = s.db.RenameGroup(id, req.Name)
		if err == nil {
			writeJSON(w, http.StatusOK, map[string]string{"name": req.Name})
			return
		}
	case len(parts) == 2 && parts[1] == "move" && r.Method == http.MethodPost:
		var req struct {
			Direction string `json:"direction"`
		}
		if decodeErr := json.NewDecoder(http.MaxBytesReader(w, r.Body, 16<<10)).Decode(&req); decodeErr != nil || (req.Direction != "up" && req.Direction != "down") {
			http.Error(w, "direction must be up or down", http.StatusBadRequest)
			return
		}
		err = s.db.MoveGroup(id, req.Direction)
		if err == nil {
			w.WriteHeader(http.StatusNoContent)
			return
		}
	case len(parts) == 1 && r.Method == http.MethodDelete:
		err = s.db.DeleteGroup(id)
		if err == nil {
			w.WriteHeader(http.StatusNoContent)
			return
		}
	default:
		http.NotFound(w, r)
		return
	}
	if err == sql.ErrNoRows {
		http.NotFound(w, r)
	} else {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func (s *Server) session(w http.ResponseWriter, r *http.Request) {
	rest := strings.TrimPrefix(r.URL.Path, "/api/sessions/")
	parts := strings.Split(strings.Trim(rest, "/"), "/")
	if len(parts) == 0 || parts[0] == "" {
		http.NotFound(w, r)
		return
	}
	id := parts[0]
	if len(parts) == 2 && parts[1] == "messages" && r.Method == http.MethodGet {
		items, err := s.db.Messages(id)
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		writeJSON(w, 200, items)
		return
	}
	if len(parts) == 2 && parts[1] == "group" && r.Method == http.MethodPut {
		var req struct {
			GroupID string `json:"group_id"`
		}
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 16<<10)).Decode(&req); err != nil {
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		if err := s.db.SetSessionGroup(id, strings.TrimSpace(req.GroupID)); err != nil {
			if err == sql.ErrNoRows {
				http.NotFound(w, r)
			} else {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		}
		w.WriteHeader(http.StatusNoContent)
		return
	}
	if len(parts) == 1 && r.Method == http.MethodDelete {
		if err := s.db.DeleteSession(id); err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		w.WriteHeader(http.StatusNoContent)
		return
	}
	if len(parts) == 1 && r.Method == http.MethodPatch {
		var req struct {
			Title string `json:"title"`
		}
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 16<<10)).Decode(&req); err != nil {
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		req.Title = strings.TrimSpace(req.Title)
		if req.Title == "" || utf8.RuneCountInString(req.Title) > 120 {
			http.Error(w, "title must be between 1 and 120 characters", http.StatusBadRequest)
			return
		}
		if err := s.db.RenameSession(id, req.Title); err != nil {
			if err == sql.ErrNoRows {
				http.NotFound(w, r)
			} else {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		}
		writeJSON(w, http.StatusOK, map[string]string{"title": req.Title})
		return
	}
	http.NotFound(w, r)
}

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
	count, _ := s.db.MessageCount(req.SessionID)
	if _, err := s.db.AddMessage(req.SessionID, "user", req.Content, "", nil, attachments); err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	_ = s.db.UpdateSession(req.SessionID, "", req.Model, req.ReasoningEffort)
	history, err := s.db.Messages(req.SessionID)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}
	messages, err := s.llmMessages(history)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
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
	result, err := runCompletionLoop(r.Context(), client, messages, req.Model, req.ReasoningEffort, cfg.Model.SystemPrompt, cfg.Tools, req.ToolsEnabled, emit)
	if result.Content != "" || result.Reasoning != "" || len(result.ToolTrace) > 0 {
		_, _ = s.db.AddMessage(req.SessionID, "assistant", result.Content, result.Reasoning, result.ToolTrace, nil)
	}
	if err != nil {
		payload, _ := json.Marshal(map[string]string{"error": err.Error()})
		fmt.Fprintf(w, "event: error\ndata: %s\n\n", payload)
	} else {
		fmt.Fprint(w, "event: done\ndata: {}\n\n")
	}
	flusher.Flush()

	if count == 0 {
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
	messages, err := s.llmMessages(history)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
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
	result, err := runCompletionLoop(r.Context(), client, messages, req.Model, req.ReasoningEffort, cfg.Model.SystemPrompt, cfg.Tools, req.ToolsEnabled, emit)
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
	messages, err := s.llmMessages(requestHistory)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

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
	result, err := runCompletionLoop(r.Context(), client, messages, req.Model, req.ReasoningEffort, cfg.Model.SystemPrompt, cfg.Tools, req.ToolsEnabled, emit)
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

func (s *Server) llmMessages(items []db.Message) ([]llm.Message, error) {
	messages := make([]llm.Message, 0, len(items))
	for _, item := range items {
		if len(item.Attachments) == 0 {
			messages = append(messages, llm.Message{Role: item.Role, Content: item.Content})
			continue
		}
		parts := make([]map[string]any, 0, len(item.Attachments)+1)
		for _, attachment := range item.Attachments {
			dataURL, err := s.media.DataURL(attachment)
			if err != nil {
				return nil, fmt.Errorf("read image %s: %w", attachment.Name, err)
			}
			parts = append(parts, map[string]any{
				"type": "image_url", "image_url": map[string]string{"url": dataURL},
			})
		}
		parts = append(parts, map[string]any{"type": "text", "text": item.Content})
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

func spaHandler(web fs.FS) http.Handler {
	files := http.FileServer(http.FS(web))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requested := strings.TrimPrefix(path.Clean(r.URL.Path), "/")
		if requested != "." {
			if _, err := fs.Stat(web, requested); err == nil {
				files.ServeHTTP(w, r)
				return
			}
		}
		index, err := fs.ReadFile(web, "index.html")
		if err != nil {
			http.Error(w, "web UI is not built; run make dist", 503)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write(index)
	})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func methodNotAllowed(w http.ResponseWriter) {
	http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
}
func errorText(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}
func newID() string { return fmt.Sprintf("chat-%d", time.Now().UnixNano()) }
