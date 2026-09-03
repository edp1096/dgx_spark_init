package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"sparktalk/internal/db"
	"sparktalk/internal/llm"
)

const maxKnowledgeImportURLs = 8

func knowledgeImportSystemPrompt(collections []db.KnowledgeCollection, webSearchAvailable bool) string {
	names := make([]string, 0, len(collections))
	for _, collection := range collections {
		if collection.Enabled {
			names = append(names, collection.Name)
		}
	}
	prompt := "Use knowledge_import only when the user explicitly asks to save or add external sources to a knowledge collection. " +
		"For an import request, select primary, authoritative source URLs; never import a search-results page, tracking URL, or a page merely mentioning a source. " +
		"Direct PDF URLs are supported by knowledge_import and must not be sent to web_fetch. " +
		"Call import_urls with the exact collection name and all selected URLs in one call when possible. " +
		"The UI always asks the user to approve the exact collection and URL list before anything is stored. Do not claim an import succeeded until the tool result confirms it. "
	if webSearchAvailable {
		prompt += "When the user gives search terms or a search-page URL, use web_search first to discover the actual source URLs, then import those URLs. "
	} else {
		prompt += "Web search is unavailable, so only import explicit source URLs supplied by the user. "
	}
	return prompt + "Available knowledge collections: " + strings.Join(names, ", ") + "."
}

func knowledgeImportToolDefinition(collections []db.KnowledgeCollection) llm.Tool {
	names := make([]string, 0, len(collections))
	for _, collection := range collections {
		if collection.Enabled {
			names = append(names, collection.Name)
		}
	}
	properties := map[string]any{
		"action": map[string]any{
			"type": "string", "enum": []string{"list_collections", "import_urls"},
			"description": "List available collections or import source URLs after user approval",
		},
		"urls": map[string]any{
			"type": "array", "minItems": 1, "maxItems": maxKnowledgeImportURLs,
			"items":       map[string]any{"type": "string"},
			"description": "Actual source URLs to retain; never pass a search-results page",
		},
		"mode": map[string]any{
			"type": "string", "enum": []string{"auto", "direct", "browser"},
			"description": "Use auto unless a JavaScript viewer specifically requires browser mode",
		},
	}
	if len(names) > 0 {
		properties["collection"] = map[string]any{
			"type": "string", "enum": names, "description": "Exact target knowledge collection name",
		}
	}
	parameters, _ := json.Marshal(map[string]any{
		"type": "object", "properties": properties,
		"required": []string{"action"}, "additionalProperties": false,
	})
	return llm.Tool{Type: "function", Function: llm.ToolFunction{
		Name:        "knowledge_import",
		Description: "List knowledge collections or import authoritative web sources into one collection after explicit UI approval.",
		Parameters:  parameters,
	}}
}

type knowledgeImportRequest struct {
	Action     string   `json:"action"`
	Collection string   `json:"collection,omitempty"`
	URLs       []string `json:"urls,omitempty"`
	Mode       string   `json:"mode,omitempty"`
}

type knowledgeImportItem struct {
	URL       string                `json:"url"`
	Status    string                `json:"status"`
	Duplicate bool                  `json:"duplicate,omitempty"`
	Document  *db.KnowledgeDocument `json:"document,omitempty"`
	Error     string                `json:"error,omitempty"`
}

func (s *Server) executeKnowledgeImport(ctx context.Context, sessionID string, call llm.ToolCall, emit eventEmitter) (string, error) {
	var request knowledgeImportRequest
	if err := json.Unmarshal([]byte(call.Function.Arguments), &request); err != nil {
		return "", errors.New("knowledge_import received invalid arguments")
	}
	request.Action = strings.ToLower(strings.TrimSpace(request.Action))
	switch request.Action {
	case "list_collections":
		return s.listKnowledgeCollectionsForTool(sessionID)
	case "import_urls":
		return s.importKnowledgeURLsFromTool(ctx, sessionID, call.ID, request, emit)
	default:
		return "", errors.New("knowledge_import action must be list_collections or import_urls")
	}
}

func (s *Server) listKnowledgeCollectionsForTool(sessionID string) (string, error) {
	collections, err := s.db.KnowledgeCollections()
	if err != nil {
		return "", fmt.Errorf("list knowledge collections: %w", err)
	}
	type view struct {
		ID          int64  `json:"id"`
		Name        string `json:"name"`
		Description string `json:"description,omitempty"`
		Documents   int    `json:"documents"`
	}
	items := make([]view, 0, len(collections))
	for _, collection := range collections {
		if collection.Enabled {
			items = append(items, view{collection.ID, collection.Name, collection.Description, collection.Documents})
		}
	}
	_ = s.db.AddToolAudit(sessionID, "knowledge_import", "", "list_collections", "executed", fmt.Sprintf("%d collection(s)", len(items)))
	data, _ := json.Marshal(map[string]any{"action": "list_collections", "collections": items})
	return string(data), nil
}

func (s *Server) importKnowledgeURLsFromTool(ctx context.Context, sessionID, callID string, request knowledgeImportRequest, emit eventEmitter) (string, error) {
	request.Collection = strings.TrimSpace(request.Collection)
	if request.Collection == "" {
		return "", errors.New("knowledge_import requires the exact target collection name")
	}
	collectionID, err := s.db.KnowledgeCollectionIDByName(request.Collection)
	if err != nil {
		return "", fmt.Errorf("knowledge collection not found: %s; list collections first", request.Collection)
	}
	mode := strings.ToLower(strings.TrimSpace(request.Mode))
	if mode == "" {
		mode = "auto"
	}
	if mode != "auto" && mode != "direct" && mode != "browser" {
		return "", errors.New("knowledge import mode must be auto, direct, or browser")
	}
	urls, err := normalizeKnowledgeImportURLs(request.URLs)
	if err != nil {
		return "", err
	}
	resource := strconv.FormatInt(collectionID, 10)
	payload := map[string]any{
		"name": "knowledge_import", "approval_kind": "knowledge_import", "action": "import_urls",
		"collection_id": collectionID, "collection_name": request.Collection, "urls": urls, "mode": mode,
	}
	decision, err := s.awaitToolApproval(ctx, callID, payload, emit)
	if err != nil {
		_ = s.db.AddToolAudit(sessionID, "knowledge_import", resource, "import_urls", string(decision), compactHistoryText(err.Error(), 300))
		return "", err
	}

	items := make([]knowledgeImportItem, 0, len(urls))
	imported, duplicates, failed := 0, 0, 0
	for _, sourceURL := range urls {
		result, status, importErr := s.importKnowledgeSource(ctx, knowledgeSourceRequest{CollectionID: collectionID, URL: sourceURL, Mode: mode})
		if importErr != nil {
			failed++
			items = append(items, knowledgeImportItem{URL: sourceURL, Status: "failed", Error: compactHistoryText(importErr.Error(), 500)})
			continue
		}
		duplicate := status == http.StatusOK
		if duplicate {
			duplicates++
		} else {
			imported++
		}
		document := result.Document
		itemStatus := document.Status
		if duplicate {
			itemStatus = "already_present"
		}
		items = append(items, knowledgeImportItem{URL: sourceURL, Status: itemStatus, Duplicate: duplicate, Document: &document})
	}
	decisionName := "executed"
	if failed > 0 {
		decisionName = "partial"
	}
	_ = s.db.AddToolAudit(sessionID, "knowledge_import", resource, "import_urls", decisionName, fmt.Sprintf("imported=%d duplicate=%d failed=%d", imported, duplicates, failed))
	data, _ := json.Marshal(map[string]any{
		"action": "import_urls", "collection": request.Collection,
		"imported": imported, "already_present": duplicates, "failed": failed, "results": items,
	})
	return string(data), nil
}

func normalizeKnowledgeImportURLs(values []string) ([]string, error) {
	if len(values) < 1 || len(values) > maxKnowledgeImportURLs {
		return nil, fmt.Errorf("knowledge_import requires between 1 and %d source URLs", maxKnowledgeImportURLs)
	}
	seen := make(map[string]struct{}, len(values))
	result := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		parsed, err := url.Parse(value)
		if err != nil || parsed.Host == "" || parsed.Scheme != "http" && parsed.Scheme != "https" {
			return nil, fmt.Errorf("invalid HTTP(S) source URL: %s", value)
		}
		if _, exists := seen[value]; exists {
			continue
		}
		seen[value] = struct{}{}
		result = append(result, value)
	}
	if len(result) == 0 {
		return nil, errors.New("knowledge_import requires at least one unique source URL")
	}
	return result, nil
}
