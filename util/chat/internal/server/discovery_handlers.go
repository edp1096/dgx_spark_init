package server

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"sparktalk/internal/db"
	"sparktalk/internal/skills"
)

func (s *Server) searchConversations(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	query := strings.TrimSpace(r.URL.Query().Get("q"))
	if query == "" || utf8.RuneCountInString(query) > 200 {
		http.Error(w, "search query must be between 1 and 200 characters", http.StatusBadRequest)
		return
	}
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	if limit < 1 || limit > 50 {
		limit = 20
	}
	items, err := s.db.SearchConversations(query, limit)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusOK, items)
}

type conversationSearchCursor struct {
	Sort      string  `json:"sort"`
	MessageID int64   `json:"message_id"`
	Rank      float64 `json:"rank,omitempty"`
}

func (s *Server) searchConversationPage(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	query := strings.TrimSpace(r.URL.Query().Get("q"))
	if query == "" || utf8.RuneCountInString(query) > 200 {
		http.Error(w, "search query must be between 1 and 200 characters", http.StatusBadRequest)
		return
	}
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	if limit < 1 || limit > 50 {
		limit = 20
	}
	sortMode := r.URL.Query().Get("sort")
	if sortMode != "recent" && sortMode != "relevance" {
		sortMode = "relevance"
	}
	scope := r.URL.Query().Get("scope")
	if scope != "title" && scope != "content" {
		scope = "all"
	}
	dateFrom, dateTo := r.URL.Query().Get("from"), r.URL.Query().Get("to")
	if !validSearchDate(dateFrom) || !validSearchDate(dateTo) || (dateFrom != "" && dateTo != "" && dateFrom > dateTo) {
		http.Error(w, "invalid search date range", http.StatusBadRequest)
		return
	}
	options := db.ConversationSearchOptions{Limit: limit, Sort: sortMode, Scope: scope, DateFrom: dateFrom, DateTo: dateTo}
	if raw := strings.TrimSpace(r.URL.Query().Get("cursor")); raw != "" {
		cursor, err := decodeConversationSearchCursor(raw)
		if err != nil || cursor.Sort != sortMode || cursor.MessageID < 1 {
			http.Error(w, "invalid search cursor", http.StatusBadRequest)
			return
		}
		options.CursorID, options.CursorRank = cursor.MessageID, cursor.Rank
	}
	items, next, err := s.db.SearchConversationPage(query, options)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	nextCursor := ""
	if next != nil {
		nextCursor = encodeConversationSearchCursor(conversationSearchCursor{Sort: sortMode, MessageID: next.MessageID, Rank: next.Rank})
	}
	writeJSON(w, http.StatusOK, map[string]any{"items": items, "next_cursor": nextCursor})
}

func validSearchDate(value string) bool {
	if value == "" {
		return true
	}
	_, err := time.Parse("2006-01-02", value)
	return err == nil
}

func encodeConversationSearchCursor(cursor conversationSearchCursor) string {
	data, _ := json.Marshal(cursor)
	return base64.RawURLEncoding.EncodeToString(data)
}

func decodeConversationSearchCursor(value string) (conversationSearchCursor, error) {
	var cursor conversationSearchCursor
	data, err := base64.RawURLEncoding.DecodeString(value)
	if err == nil {
		err = json.Unmarshal(data, &cursor)
	}
	return cursor, err
}

type publicSkill struct {
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Toolsets    []string `json:"toolsets"`
	Available   bool     `json:"available"`
}

func (s *Server) skillCatalog(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	cfg, _ := s.snapshot()
	active := map[string]bool{
		"web":   cfg.Tools.Enabled,
		"media": cfg.Tools.MediaImportEnabled,
		"image": cfg.Image.Enabled,
		"ssh":   cfg.Extra.SSHEnabled,
	}
	items := make([]publicSkill, 0, len(skills.Catalog()))
	for _, skill := range skills.Catalog() {
		available := cfg.Tools.SkillsEnabled
		for _, toolset := range skill.Toolsets {
			available = available && active[toolset]
		}
		items = append(items, publicSkill{Name: skill.Name, Description: skill.Description, Toolsets: skill.Toolsets, Available: available})
	}
	writeJSON(w, http.StatusOK, items)
}

func (s *Server) toolAudits(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		methodNotAllowed(w)
		return
	}
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	items, err := s.db.ToolAudits(limit)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	writeJSON(w, http.StatusOK, items)
}
