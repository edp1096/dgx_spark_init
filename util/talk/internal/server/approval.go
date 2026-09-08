package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"time"
)

type approvalDecision string

const (
	approvalReject       approvalDecision = "reject"
	approvalOnce         approvalDecision = "once"
	approvalConversation approvalDecision = "conversation"
)

type toolApproval struct{ decision chan approvalDecision }

func (s *Server) awaitToolApproval(ctx context.Context, callID string, payload map[string]any, emit eventEmitter) (approvalDecision, error) {
	random := make([]byte, 12)
	if _, err := rand.Read(random); err != nil {
		return "", err
	}
	id := hex.EncodeToString(random)
	pending := &toolApproval{decision: make(chan approvalDecision, 1)}
	s.approvalsMu.Lock()
	s.approvals[id] = pending
	s.approvalsMu.Unlock()
	defer func() {
		s.approvalsMu.Lock()
		delete(s.approvals, id)
		s.approvalsMu.Unlock()
	}()
	payload["id"] = callID
	payload["approval_id"] = id
	if err := emit("tool_approval", payload); err != nil {
		return "", err
	}
	timer := time.NewTimer(5 * time.Minute)
	defer timer.Stop()
	select {
	case decision := <-pending.decision:
		approved := decision != approvalReject
		if err := emit("tool_approval_resolved", map[string]any{"id": callID, "approved": approved, "decision": decision}); err != nil {
			return "", err
		}
		if !approved {
			return decision, errors.New("tool action was rejected by the user")
		}
		return decision, nil
	case <-timer.C:
		return "", errors.New("tool action approval timed out")
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

func (s *Server) toolApproval(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	id := strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/tool-approvals/"), "/")
	if id == "" {
		http.Error(w, "approval id is required", http.StatusBadRequest)
		return
	}
	var request struct {
		Decision string `json:"decision"`
		Approved *bool  `json:"approved,omitempty"`
	}
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1024)).Decode(&request); err != nil {
		http.Error(w, "invalid approval response", http.StatusBadRequest)
		return
	}
	decision := approvalDecision(strings.TrimSpace(request.Decision))
	if decision == "" && request.Approved != nil {
		if *request.Approved {
			decision = approvalOnce
		} else {
			decision = approvalReject
		}
	}
	if decision != approvalReject && decision != approvalOnce && decision != approvalConversation {
		http.Error(w, "decision must be reject, once, or conversation", http.StatusBadRequest)
		return
	}
	s.approvalsMu.Lock()
	pending := s.approvals[id]
	s.approvalsMu.Unlock()
	if pending == nil {
		http.Error(w, "approval request is no longer active", http.StatusNotFound)
		return
	}
	select {
	case pending.decision <- decision:
		writeJSON(w, http.StatusOK, map[string]any{"accepted": true, "approved": decision != approvalReject, "decision": decision})
	default:
		http.Error(w, "approval request was already answered", http.StatusConflict)
	}
}
