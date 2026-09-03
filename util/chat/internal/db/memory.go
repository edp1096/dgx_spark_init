package db

import (
	"database/sql"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
)

func (d *DB) Memories() ([]Memory, error) {
	rows, err := d.conn.Query(`SELECT id,kind,priority,title,content,enabled,source_session_id,source_message_id,created_at,updated_at FROM memories ORDER BY kind,id`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	items := []Memory{}
	for rows.Next() {
		var item Memory
		if err := rows.Scan(&item.ID, &item.Kind, &item.Priority, &item.Title, &item.Content, &item.Enabled, &item.SourceSessionID, &item.SourceMessageID, &item.CreatedAt, &item.UpdatedAt); err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	return items, rows.Err()
}

// FindMemories searches both enabled and disabled records for conversational
// memory management. It intentionally uses bounded SQL substring matching so
// a disabled record, which is absent from memory_search, can still be found.
func (d *DB) FindMemories(query string, limit int) ([]Memory, error) {
	if limit < 1 {
		limit = 8
	}
	if limit > 20 {
		limit = 20
	}
	terms := memoryLookupTerms(query)
	if len(terms) == 0 {
		rows, err := d.conn.Query(`SELECT id,kind,priority,title,content,enabled,source_session_id,source_message_id,created_at,updated_at FROM memories ORDER BY updated_at DESC LIMIT ?`, limit)
		if err != nil {
			return nil, err
		}
		return scanMemories(rows)
	}
	where := make([]string, 0, len(terms))
	args := make([]any, 0, len(terms)*2+1)
	for _, term := range terms {
		where = append(where, `(instr(lower(title),?)>0 OR instr(lower(content),?)>0)`)
		args = append(args, term, term)
	}
	args = append(args, min(limit*10, 200))
	rows, err := d.conn.Query(`SELECT id,kind,priority,title,content,enabled,source_session_id,source_message_id,created_at,updated_at
		FROM memories WHERE `+strings.Join(where, " OR ")+` ORDER BY updated_at DESC LIMIT ?`, args...)
	if err != nil {
		return nil, err
	}
	items, err := scanMemories(rows)
	if err != nil {
		return nil, err
	}
	sort.SliceStable(items, func(left, right int) bool {
		leftScore := memoryRecordScore(items[left], terms)
		rightScore := memoryRecordScore(items[right], terms)
		if leftScore == rightScore {
			return items[left].UpdatedAt.After(items[right].UpdatedAt)
		}
		return leftScore > rightScore
	})
	if len(items) > limit {
		items = items[:limit]
	}
	return items, nil
}

func scanMemories(rows *sql.Rows) ([]Memory, error) {
	defer rows.Close()
	items := []Memory{}
	for rows.Next() {
		var item Memory
		if err := rows.Scan(&item.ID, &item.Kind, &item.Priority, &item.Title, &item.Content, &item.Enabled, &item.SourceSessionID, &item.SourceMessageID, &item.CreatedAt, &item.UpdatedAt); err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	return items, rows.Err()
}

func memoryLookupTerms(value string) []string {
	parts := strings.FieldsFunc(strings.ToLower(value), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
	seen := make(map[string]struct{}, len(parts))
	terms := make([]string, 0, 8)
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" || isMemoryManagementTerm(part) {
			continue
		}
		if _, ok := seen[part]; ok {
			continue
		}
		seen[part] = struct{}{}
		terms = append(terms, part)
		if len(terms) == 8 {
			break
		}
	}
	return terms
}

func isMemoryManagementTerm(term string) bool {
	for _, prefix := range []string{"기억", "잊어", "잊고", "삭제", "제거", "수정", "변경", "보여", "알려", "확인", "목록", "사용", "활성", "비활성", "항상", "관련"} {
		if strings.HasPrefix(term, prefix) {
			return true
		}
	}
	for _, exact := range []string{"내", "내가", "뭘", "무엇을", "있는지", "있지", "해줘", "해라", "그거", "그것"} {
		if term == exact {
			return true
		}
	}
	return false
}

func memoryRecordScore(item Memory, terms []string) int {
	title := strings.ToLower(item.Title)
	content := strings.ToLower(item.Content)
	score := 0
	for _, term := range terms {
		weight := min(utf8.RuneCountInString(term), 10)
		if strings.Contains(title, term) {
			score += 20 + weight
		}
		if count := strings.Count(content, term); count > 0 {
			score += 5 + weight + min(count, 3)
		}
	}
	return score
}

func (d *DB) AddMemory(kind, priority, title, content, sourceSessionID string, sourceMessageID int64) (Memory, error) {
	if sourceMessageID > 0 {
		var existing Memory
		err := d.conn.QueryRow(`SELECT id,kind,priority,title,content,enabled,source_session_id,source_message_id,created_at,updated_at
			FROM memories WHERE source_message_id=? AND source_session_id=? ORDER BY id DESC LIMIT 1`, sourceMessageID, sourceSessionID).
			Scan(&existing.ID, &existing.Kind, &existing.Priority, &existing.Title, &existing.Content, &existing.Enabled, &existing.SourceSessionID, &existing.SourceMessageID, &existing.CreatedAt, &existing.UpdatedAt)
		if err == nil {
			return existing, nil
		}
		if !errors.Is(err, sql.ErrNoRows) {
			return Memory{}, err
		}
	}
	now := time.Now()
	result, err := d.conn.Exec(`INSERT INTO memories(kind,priority,title,content,enabled,source_session_id,source_message_id,created_at,updated_at) VALUES(?,?,?,?,1,?,?,?,?)`,
		kind, priority, title, content, sourceSessionID, sourceMessageID, now, now)
	if err != nil {
		return Memory{}, err
	}
	id, _ := result.LastInsertId()
	return Memory{ID: id, Kind: kind, Priority: priority, Title: title, Content: content, Enabled: true, SourceSessionID: sourceSessionID, SourceMessageID: sourceMessageID, CreatedAt: now, UpdatedAt: now}, nil
}

func (d *DB) UpdateMemory(id int64, kind, priority, title, content string, enabled bool) (Memory, error) {
	now := time.Now()
	result, err := d.conn.Exec(`UPDATE memories SET kind=?,priority=?,title=?,content=?,enabled=?,updated_at=? WHERE id=?`, kind, priority, title, content, enabled, now, id)
	if err != nil {
		return Memory{}, err
	}
	if changed, _ := result.RowsAffected(); changed == 0 {
		return Memory{}, sql.ErrNoRows
	}
	return d.Memory(id)
}

func (d *DB) Memory(id int64) (Memory, error) {
	var item Memory
	err := d.conn.QueryRow(`SELECT id,kind,priority,title,content,enabled,source_session_id,source_message_id,created_at,updated_at FROM memories WHERE id=?`, id).
		Scan(&item.ID, &item.Kind, &item.Priority, &item.Title, &item.Content, &item.Enabled, &item.SourceSessionID, &item.SourceMessageID, &item.CreatedAt, &item.UpdatedAt)
	return item, err
}

func (d *DB) DeleteMemory(id int64) error {
	result, err := d.conn.Exec(`DELETE FROM memories WHERE id=?`, id)
	if err != nil {
		return err
	}
	if changed, _ := result.RowsAffected(); changed == 0 {
		return sql.ErrNoRows
	}
	return nil
}

func (d *DB) UserMemories(limit int) ([]RecallItem, error) {
	if limit <= 0 {
		limit = 20
	}
	rows, err := d.conn.Query(`SELECT priority,title,content,source_session_id,source_message_id,updated_at FROM memories WHERE enabled=1 AND kind='user' ORDER BY priority='preferred' DESC,updated_at DESC LIMIT ?`, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	items := []RecallItem{}
	for rows.Next() {
		var item RecallItem
		item.Kind = "user"
		if err := rows.Scan(&item.Priority, &item.Title, &item.Content, &item.SessionID, &item.MessageID, &item.CreatedAt); err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	return items, rows.Err()
}

func (d *DB) SearchMemories(query string, limit int) ([]RecallItem, error) {
	terms := recallSearchTerms(query)
	match := ftsMatchQuery(terms)
	if match == "" || limit <= 0 {
		return nil, nil
	}
	candidateLimit := min(limit*8, 100)
	rows, err := d.conn.Query(`
		SELECT memory_row.priority,memory_row.title,memory_row.content,memory_row.source_session_id,memory_row.source_message_id,memory_row.updated_at
		FROM memory_search JOIN memories AS memory_row ON memory_row.id=memory_search.rowid
		WHERE memory_search MATCH ? AND memory_row.enabled=1 AND memory_row.kind='memory'
		ORDER BY bm25(memory_search),memory_row.updated_at DESC LIMIT ?`, match, candidateLimit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	items := []scoredRecall{}
	for rows.Next() {
		var item RecallItem
		item.Kind = "memory"
		if err := rows.Scan(&item.Priority, &item.Title, &item.Content, &item.SessionID, &item.MessageID, &item.CreatedAt); err != nil {
			return nil, err
		}
		if !relevantRecallMatch(item.Title, item.Content, terms) {
			continue
		}
		items = append(items, scoredRecall{item: item, score: recallRelevanceScore(item, terms)})
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return bestRecallItems(items, limit, false), nil
}

func (d *DB) SearchMessages(query, excludeSessionID string, limit int) ([]RecallItem, error) {
	terms := recallSearchTerms(query)
	match := ftsMatchQuery(terms)
	if match == "" || limit <= 0 {
		return nil, nil
	}
	candidateLimit := min(limit*8, 100)
	rows, err := d.conn.Query(`
		SELECT message_search.title,message_search.role,message_search.content,message_search.session_id,message_row.id,message_row.created_at
		FROM message_search JOIN messages AS message_row ON message_row.id=message_search.rowid
		WHERE message_search MATCH ? AND message_search.session_id<>? AND message_row.status='completed'
		ORDER BY bm25(message_search),message_row.created_at DESC LIMIT ?`, match, excludeSessionID, candidateLimit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	items := []scoredRecall{}
	for rows.Next() {
		var item RecallItem
		item.Kind = "session"
		if err := rows.Scan(&item.Title, &item.Role, &item.Content, &item.SessionID, &item.MessageID, &item.CreatedAt); err != nil {
			return nil, err
		}
		if !relevantRecallMatch(item.Title, item.Content, terms) {
			continue
		}
		items = append(items, scoredRecall{item: item, score: recallRelevanceScore(item, terms)})
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return bestRecallItems(items, limit, true), nil
}

type scoredRecall struct {
	item  RecallItem
	score float64
}

func bestRecallItems(items []scoredRecall, limit int, uniqueSessions bool) []RecallItem {
	sort.SliceStable(items, func(left, right int) bool {
		if items[left].score == items[right].score {
			return items[left].item.CreatedAt.After(items[right].item.CreatedAt)
		}
		return items[left].score > items[right].score
	})
	result := make([]RecallItem, 0, min(limit, len(items)))
	seenSessions := make(map[string]struct{})
	for _, candidate := range items {
		if uniqueSessions {
			if _, seen := seenSessions[candidate.item.SessionID]; seen {
				continue
			}
			seenSessions[candidate.item.SessionID] = struct{}{}
		}
		result = append(result, candidate.item)
		if len(result) == limit {
			break
		}
	}
	return result
}

func recallRelevanceScore(item RecallItem, terms []string) float64 {
	title := strings.ToLower(item.Title)
	content := strings.ToLower(item.Content)
	score := 0.0
	for _, term := range terms {
		termRunes := utf8.RuneCountInString(term)
		weight := float64(min(termRunes, 10))
		if strings.Contains(title, term) {
			score += 8 + weight
		}
		count := strings.Count(content, term)
		if count > 0 {
			score += 3 + weight + float64(min(count, 3)-1)
		}
	}
	age := time.Since(item.CreatedAt)
	switch {
	case age < 7*24*time.Hour:
		score += 2
	case age < 30*24*time.Hour:
		score += 1
	}
	return score
}

func (d *DB) SearchConversations(query string, limit int) ([]RecallItem, error) {
	query = strings.TrimSpace(query)
	if query == "" || limit <= 0 {
		return nil, nil
	}
	items, err := d.SearchMessages(query, "", limit)
	if err != nil {
		return nil, err
	}
	if items == nil {
		items = []RecallItem{}
	}
	seenSessions := make(map[string]struct{}, len(items))
	for _, item := range items {
		seenSessions[item.SessionID] = struct{}{}
	}
	// FTS5 trigram cannot serve one- and two-rune searches. The bounded LIKE
	// fallback also makes exact short model names and Korean nouns searchable.
	if len(items) == 0 {
		rows, queryErr := d.conn.Query(`
		SELECT session_row.title,message_row.role,message_row.content,message_row.session_id,message_row.id,message_row.created_at
		FROM messages AS message_row JOIN sessions AS session_row ON session_row.id=message_row.session_id
		WHERE message_row.status='completed'
		  AND (instr(lower(message_row.content),lower(?))>0 OR instr(lower(session_row.title),lower(?))>0)
		ORDER BY message_row.created_at DESC LIMIT ?`, query, query, min(limit*4, 100))
		if queryErr != nil {
			return nil, queryErr
		}
		for rows.Next() {
			var item RecallItem
			item.Kind = "session"
			if scanErr := rows.Scan(&item.Title, &item.Role, &item.Content, &item.SessionID, &item.MessageID, &item.CreatedAt); scanErr != nil {
				rows.Close()
				return nil, scanErr
			}
			if _, seen := seenSessions[item.SessionID]; seen {
				continue
			}
			seenSessions[item.SessionID] = struct{}{}
			items = append(items, item)
			if len(items) == limit {
				break
			}
		}
		if rowsErr := rows.Err(); rowsErr != nil {
			rows.Close()
			return nil, rowsErr
		}
		rows.Close()
	}
	if len(items) == limit {
		return items, nil
	}
	// A newly created or intentionally empty conversation has no FTS row yet,
	// but its visible title must still be discoverable.
	rows, err := d.conn.Query(`
		SELECT session_row.title,session_row.id,session_row.created_at
		FROM sessions AS session_row
		WHERE instr(lower(session_row.title),lower(?))>0
		ORDER BY session_row.updated_at DESC LIMIT ?`, query, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	for rows.Next() {
		var item RecallItem
		item.Kind = "session"
		if err := rows.Scan(&item.Title, &item.SessionID, &item.CreatedAt); err != nil {
			return nil, err
		}
		if _, seen := seenSessions[item.SessionID]; seen {
			continue
		}
		seenSessions[item.SessionID] = struct{}{}
		items = append(items, item)
		if len(items) == limit {
			break
		}
	}
	return items, rows.Err()
}

// SearchConversationPage returns one bounded page without calculating a total
// match count. Recent pages use a message-id cursor; relevance pages add the
// exact FTS rank so deep result sets are not traversed with OFFSET.
func (d *DB) SearchConversationPage(query string, options ConversationSearchOptions) ([]RecallItem, *ConversationSearchCursor, error) {
	query = strings.TrimSpace(query)
	if query == "" {
		return []RecallItem{}, nil, nil
	}
	if options.Limit < 1 || options.Limit > 50 {
		options.Limit = 20
	}
	if options.Sort != "recent" {
		options.Sort = "relevance"
	}
	if options.Scope != "title" && options.Scope != "content" {
		options.Scope = "all"
	}
	match := ftsMatchQuery(manualSearchTerms(query))
	if match == "" {
		return d.searchConversationPageSubstring(query, options)
	}
	if options.Scope != "all" {
		match = options.Scope + " : (" + match + ")"
	}

	where := []string{"message_search MATCH ?", "message_row.status='completed'"}
	args := []any{match}
	appendSearchDateFilters(&where, &args, options)
	if options.CursorID > 0 {
		if options.Sort == "recent" {
			where = append(where, "message_row.id < ?")
			args = append(args, options.CursorID)
		} else {
			where = append(where, "(message_search.rank > ? OR (message_search.rank = ? AND message_row.id < ?))")
			args = append(args, options.CursorRank, options.CursorRank, options.CursorID)
		}
	}
	order := "message_search.rank ASC,message_row.id DESC"
	if options.Sort == "recent" {
		order = "message_row.id DESC"
	}
	args = append(args, options.Limit+1)
	statement := fmt.Sprintf(`
		SELECT message_search.title,message_search.role,
			snippet(message_search,4,'','',' … ',32),message_search.session_id,
			message_row.id,message_row.created_at,message_search.rank
		FROM message_search JOIN messages AS message_row ON message_row.id=message_search.rowid
		WHERE %s ORDER BY %s LIMIT ?`, strings.Join(where, " AND "), order)
	rows, err := d.conn.Query(statement, args...)
	if err != nil {
		return nil, nil, err
	}
	defer rows.Close()
	items := make([]RecallItem, 0, options.Limit+1)
	ranks := make([]float64, 0, options.Limit+1)
	for rows.Next() {
		var item RecallItem
		var rank float64
		item.Kind = "session"
		if err := rows.Scan(&item.Title, &item.Role, &item.Content, &item.SessionID, &item.MessageID, &item.CreatedAt, &rank); err != nil {
			return nil, nil, err
		}
		items = append(items, item)
		ranks = append(ranks, rank)
	}
	if err := rows.Err(); err != nil {
		return nil, nil, err
	}
	if len(items) <= options.Limit {
		if len(items) == 0 && options.CursorID == 0 && options.Scope != "content" {
			return d.emptyConversationTitleMatches(query, options)
		}
		return items, nil, nil
	}
	items = items[:options.Limit]
	last := len(items) - 1
	return items, &ConversationSearchCursor{MessageID: items[last].MessageID, Rank: ranks[last]}, nil
}

func (d *DB) searchConversationPageSubstring(query string, options ConversationSearchOptions) ([]RecallItem, *ConversationSearchCursor, error) {
	where := []string{"message_row.status='completed'"}
	args := []any{}
	switch options.Scope {
	case "title":
		where = append(where, "instr(lower(session_row.title),lower(?))>0")
		args = append(args, query)
	case "content":
		where = append(where, "instr(lower(message_row.content),lower(?))>0")
		args = append(args, query)
	default:
		where = append(where, "(instr(lower(session_row.title),lower(?))>0 OR instr(lower(message_row.content),lower(?))>0)")
		args = append(args, query, query)
	}
	appendSearchDateFilters(&where, &args, options)
	if options.CursorID > 0 {
		where = append(where, "message_row.id < ?")
		args = append(args, options.CursorID)
	}
	args = append(args, options.Limit+1)
	statement := fmt.Sprintf(`
		SELECT session_row.title,message_row.role,substr(message_row.content,1,600),
			message_row.session_id,message_row.id,message_row.created_at
		FROM messages AS message_row JOIN sessions AS session_row ON session_row.id=message_row.session_id
		WHERE %s ORDER BY message_row.id DESC LIMIT ?`, strings.Join(where, " AND "))
	rows, err := d.conn.Query(statement, args...)
	if err != nil {
		return nil, nil, err
	}
	defer rows.Close()
	items := make([]RecallItem, 0, options.Limit+1)
	for rows.Next() {
		var item RecallItem
		item.Kind = "session"
		if err := rows.Scan(&item.Title, &item.Role, &item.Content, &item.SessionID, &item.MessageID, &item.CreatedAt); err != nil {
			return nil, nil, err
		}
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		return nil, nil, err
	}
	if len(items) <= options.Limit {
		if len(items) == 0 && options.CursorID == 0 && options.Scope != "content" {
			return d.emptyConversationTitleMatches(query, options)
		}
		return items, nil, nil
	}
	items = items[:options.Limit]
	return items, &ConversationSearchCursor{MessageID: items[len(items)-1].MessageID}, nil
}

func (d *DB) emptyConversationTitleMatches(query string, options ConversationSearchOptions) ([]RecallItem, *ConversationSearchCursor, error) {
	where := []string{"instr(lower(session_row.title),lower(?))>0", "NOT EXISTS(SELECT 1 FROM messages WHERE messages.session_id=session_row.id AND messages.status='completed')"}
	args := []any{query}
	if options.DateFrom != "" {
		where = append(where, "substr(session_row.created_at,1,10)>=?")
		args = append(args, options.DateFrom)
	}
	if options.DateTo != "" {
		where = append(where, "substr(session_row.created_at,1,10)<=?")
		args = append(args, options.DateTo)
	}
	args = append(args, options.Limit)
	statement := fmt.Sprintf(`SELECT session_row.title,session_row.id,session_row.created_at
		FROM sessions AS session_row WHERE %s ORDER BY session_row.updated_at DESC LIMIT ?`, strings.Join(where, " AND "))
	rows, err := d.conn.Query(statement, args...)
	if err != nil {
		return nil, nil, err
	}
	defer rows.Close()
	items := []RecallItem{}
	for rows.Next() {
		var item RecallItem
		item.Kind = "session"
		if err := rows.Scan(&item.Title, &item.SessionID, &item.CreatedAt); err != nil {
			return nil, nil, err
		}
		items = append(items, item)
	}
	return items, nil, rows.Err()
}

func appendSearchDateFilters(where *[]string, args *[]any, options ConversationSearchOptions) {
	if options.DateFrom != "" {
		*where = append(*where, "substr(message_row.created_at,1,10)>=?")
		*args = append(*args, options.DateFrom)
	}
	if options.DateTo != "" {
		*where = append(*where, "substr(message_row.created_at,1,10)<=?")
		*args = append(*args, options.DateTo)
	}
}

func manualSearchTerms(value string) []string {
	parts := strings.FieldsFunc(strings.ToLower(value), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
	seen := make(map[string]struct{}, len(parts))
	terms := make([]string, 0, 10)
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if utf8.RuneCountInString(part) < 3 {
			continue
		}
		if _, ok := seen[part]; ok {
			continue
		}
		seen[part] = struct{}{}
		terms = append(terms, part)
		if len(terms) == 10 {
			break
		}
	}
	return terms
}

func recallSearchTerms(value string) []string {
	parts := strings.FieldsFunc(strings.ToLower(value), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
	seen := make(map[string]struct{}, len(parts))
	terms := make([]string, 0, 10)
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if utf8.RuneCountInString(part) < 3 || isRecallIntentTerm(part) {
			continue
		}
		if _, ok := seen[part]; ok {
			continue
		}
		seen[part] = struct{}{}
		terms = append(terms, part)
		if len(terms) == 10 {
			break
		}
	}
	return terms
}

func ftsMatchQuery(terms []string) string {
	quoted := make([]string, 0, len(terms))
	for _, term := range terms {
		quoted = append(quoted, `"`+strings.ReplaceAll(term, `"`, `""`)+`"`)
	}
	return strings.Join(quoted, " OR ")
}

func isRecallIntentTerm(term string) bool {
	for _, prefix := range []string{"알려", "답해", "답변", "설명", "확인", "찾아", "보여", "작성", "참고"} {
		if strings.HasPrefix(term, prefix) {
			return true
		}
	}
	for _, prefix := range []string{"내용", "질문", "과거", "이전", "대화", "그거", "그건", "그게", "그걸", "그때", "아까", "저번", "지난번", "말했던", "하던거"} {
		if strings.HasPrefix(term, prefix) {
			return true
		}
	}
	return false
}

func relevantRecallMatch(title, content string, terms []string) bool {
	if len(terms) == 0 {
		return false
	}
	haystack := strings.ToLower(title + "\n" + content)
	matched, strongest := 0, 0
	for _, term := range terms {
		if !strings.Contains(haystack, term) {
			continue
		}
		matched++
		strongest = max(strongest, utf8.RuneCountInString(term))
	}
	if len(terms) == 1 {
		return matched == 1
	}
	return strongest >= 5 || matched >= 2
}

func IsMemoryNotFound(err error) bool { return errors.Is(err, sql.ErrNoRows) }
