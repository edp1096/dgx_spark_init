package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"

	"sparktalk/internal/db"
)

type activeTurnKey struct{}
type activeTurn struct {
	mu              sync.Mutex
	token           string
	session         string
	anchor          int64
	base            string
	accepting       bool
	pending         []db.TurnInput
	applied         []db.TurnInput
	cancelInference context.CancelFunc
	ctx             context.Context
}

func (s *Server) claimTurn(ctx context.Context, session string) (context.Context, func(), error) {
	if _, ok := ctx.Value(activeTurnKey{}).(*activeTurn); ok {
		return ctx, func() {}, nil
	}
	s.turnMu.Lock()
	defer s.turnMu.Unlock()
	if s.turns == nil {
		s.turns = make(map[string]*activeTurn)
	}
	if s.turns[session] != nil {
		return ctx, nil, fmt.Errorf("이 대화는 이미 응답 중입니다. 추가 입력으로 보내세요")
	}
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return ctx, nil, err
	}
	t := &activeTurn{token: hex.EncodeToString(b), session: session, ctx: ctx}
	s.turns[session] = t
	return context.WithValue(ctx, activeTurnKey{}, t), func() {
		s.turnMu.Lock()
		defer s.turnMu.Unlock()
		t.mu.Lock()
		t.accepting = false
		if t.cancelInference != nil {
			t.cancelInference()
		}
		t.mu.Unlock()
		if s.turns[session] == t {
			delete(s.turns, session)
		}
	}, nil
}

func turnFrom(ctx context.Context) *activeTurn {
	t, _ := ctx.Value(activeTurnKey{}).(*activeTurn)
	return t
}

// Only the inference child is cancelled. Tools keep using the parent context.
func (t *activeTurn) inferContext(ctx context.Context) (context.Context, context.CancelFunc, []db.TurnInput) {
	child, cancel := context.WithCancel(ctx)
	if t == nil {
		return child, cancel, nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	inputs := t.pending
	t.applied = append(t.applied, inputs...)
	t.pending = nil
	t.cancelInference = cancel
	return child, cancel, inputs
}
func (t *activeTurn) inferenceEnded(final bool) bool {
	if t == nil {
		return false
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t.cancelInference = nil
	if len(t.pending) > 0 {
		return true
	}
	if final {
		t.accepting = false
	}
	return false
}
func (t *activeTurn) hasPending() bool {
	if t == nil {
		return false
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.pending) > 0
}

func (s *Server) steerChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		methodNotAllowed(w)
		return
	}
	var req struct {
		SessionID string `json:"session_id"`
		TurnID    string `json:"turn_id"`
		InputID   string `json:"input_id"`
		Content   string `json:"content"`
	}
	if json.NewDecoder(http.MaxBytesReader(w, r.Body, 128<<10)).Decode(&req) != nil {
		http.Error(w, "invalid additional input", 400)
		return
	}
	req.Content = strings.TrimSpace(req.Content)
	if req.Content == "" || len(req.Content) > 32768 || len(req.InputID) < 8 || len(req.InputID) > 128 {
		http.Error(w, "추가 입력은 1~32768바이트여야 합니다", 400)
		return
	}
	s.turnMu.Lock()
	t := s.turns[req.SessionID]
	s.turnMu.Unlock()
	if t == nil || t.token != req.TurnID {
		http.Error(w, "응답이 종료됐습니다. 입력을 다시 전송하세요", 409)
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	if !t.accepting || t.ctx.Err() != nil {
		http.Error(w, "응답이 종료됐습니다. 입력을 다시 전송하세요", 409)
		return
	}
	if len(t.pending) >= 32 {
		http.Error(w, "추가 입력이 너무 많습니다. 반영 후 다시 전송하세요", 429)
		return
	}
	input := db.TurnInput{ID: req.InputID, Content: req.Content}
	added, err := s.db.SaveTurnInput(t.session, t.anchor, t.base, input)
	if err != nil {
		http.Error(w, "추가 입력 저장 실패: "+err.Error(), 500)
		return
	}
	if added {
		t.pending = append(t.pending, input)
		if t.cancelInference != nil {
			t.cancelInference()
		}
	}
	writeJSON(w, http.StatusAccepted, input)
}

func userTurnContent(item db.Message) string {
	text := item.Content
	for _, input := range item.TurnInputs {
		text += "\n\n추가 지시:\n" + input.Content
	}
	return text
}

func (t *activeTurn) appliedInputs() []db.TurnInput {
	if t == nil {
		return nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]db.TurnInput(nil), t.applied...)
}
