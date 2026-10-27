package server

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"
	"time"
)

var errChatStopped = errors.New("chat explicitly stopped by user")

// The task owns its context. A browser is only a subscriber. Events are spooled
// to a private temporary file so long generations do not grow a RAM buffer.
type chatStream struct {
	mu        sync.Mutex
	file      *os.File
	size      int64
	notify    chan struct{}
	done      bool
	cancel    context.CancelFunc
	operation string
	messageID int64
	input     string
	started   time.Time
}

func (s *Server) claimBrowserTurn(parent context.Context, session, operation string, messageID int64, input string) (context.Context, func(), error) {
	ctx, finish, err := s.tasks.Track(context.WithoutCancel(parent))
	if err != nil {
		return nil, nil, err
	}
	ctx, cancel := context.WithCancelCause(ctx)
	ctx, release, err := s.claimTurn(ctx, session)
	if err != nil {
		cancel(nil)
		finish()
		return nil, nil, err
	}
	file, err := os.CreateTemp("", "sparktalk-chat-*.sse")
	if err != nil {
		release()
		cancel(nil)
		finish()
		return nil, nil, err
	}
	stream := &chatStream{file: file, notify: make(chan struct{}), cancel: func() { cancel(errChatStopped) }, operation: operation, messageID: messageID, input: input, started: time.Now()}
	turn := turnFrom(ctx)
	turn.mu.Lock()
	turn.stream = stream
	turn.mu.Unlock()
	return ctx, func() {
		stream.mu.Lock()
		stream.done = true
		close(stream.notify)
		stream.mu.Unlock()
		release()
		cancel(nil)
		finish()
		file.Close()
		os.Remove(file.Name())
	}, nil
}

func setChatStreamMessage(ctx context.Context, id int64) {
	if t := turnFrom(ctx); t != nil && t.stream != nil {
		t.stream.mu.Lock()
		t.stream.messageID = id
		t.stream.mu.Unlock()
	}
}

func browserTurnEmitter(w http.ResponseWriter, browser context.Context, task context.Context) eventEmitter {
	var mu sync.Mutex
	connected := true
	return func(kind string, payload any) error {
		mu.Lock()
		defer mu.Unlock()
		data, err := json.Marshal(payload)
		if err != nil {
			return err
		}
		raw := []byte(fmt.Sprintf("event: %s\ndata: %s\n\n", kind, data))
		if t := turnFrom(task); t != nil && t.stream != nil {
			stream := t.stream
			stream.mu.Lock()
			n, writeErr := stream.file.Write(raw)
			stream.size += int64(n)
			close(stream.notify)
			stream.notify = make(chan struct{})
			stream.mu.Unlock()
			if writeErr != nil {
				return writeErr
			}
		}
		if connected && browser.Err() == nil {
			controller := http.NewResponseController(w)
			_ = controller.SetWriteDeadline(time.Now().Add(5 * time.Second))
			_, err = w.Write(raw)
			if err == nil {
				err = controller.Flush()
			}
			_ = controller.SetWriteDeadline(time.Time{})
			if err != nil {
				connected = false
			}
		}
		return nil
	}
}

func (s *Server) activeChat(w http.ResponseWriter, r *http.Request) {
	session := r.URL.Query().Get("session_id")
	token := r.URL.Query().Get("turn_id")
	if r.Method == http.MethodPost {
		var a struct {
			Session string `json:"session_id"`
			Token   string `json:"turn_id"`
		}
		if json.NewDecoder(http.MaxBytesReader(w, r.Body, 4096)).Decode(&a) != nil {
			http.Error(w, "invalid cancellation", 400)
			return
		}
		session, token = a.Session, a.Token
	}
	s.turnMu.Lock()
	turn := s.turns[session]
	s.turnMu.Unlock()
	if turn == nil {
		if r.Method == http.MethodGet && r.URL.Path == "/api/chat/active" {
			writeJSON(w, 200, map[string]any{"active": false})
			return
		}
		http.NotFound(w, r)
		return
	}
	turn.mu.Lock()
	stream := turn.stream
	turnID := turn.token
	turn.mu.Unlock()
	if stream == nil {
		writeJSON(w, 200, map[string]any{"active": false})
		return
	}
	if r.Method == http.MethodGet && r.URL.Path == "/api/chat/active" {
		stream.mu.Lock()
		state := map[string]any{"active": !stream.done, "ready": stream.size > 0, "turn_id": turnID, "session_id": session, "operation": stream.operation, "message_id": stream.messageID, "input": stream.input, "started_at": stream.started}
		stream.mu.Unlock()
		writeJSON(w, 200, state)
		return
	}
	if token == "" || token != turnID {
		http.NotFound(w, r)
		return
	}
	if r.Method == http.MethodPost && r.URL.Path == "/api/chat/active/cancel" {
		stream.cancel()
		w.WriteHeader(http.StatusAccepted)
		return
	}
	if r.Method != http.MethodGet || r.URL.Path != "/api/chat/active/events" {
		http.NotFound(w, r)
		return
	}
	file, err := os.Open(stream.file.Name())
	if err != nil {
		http.NotFound(w, r)
		return
	}
	defer file.Close()
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", 500)
		return
	}
	var offset int64
	buffer := make([]byte, 64<<10)
	for {
		stream.mu.Lock()
		size, done, notify := stream.size, stream.done, stream.notify
		stream.mu.Unlock()
		for offset < size {
			n, err := file.ReadAt(buffer[:min(int64(len(buffer)), size-offset)], offset)
			if err != nil && err != io.EOF {
				return
			}
			if n == 0 {
				return
			}
			_ = http.NewResponseController(w).SetWriteDeadline(time.Now().Add(5 * time.Second))
			if _, err = w.Write(buffer[:n]); err != nil {
				return
			}
			flusher.Flush()
			offset += int64(n)
			_ = http.NewResponseController(w).SetWriteDeadline(time.Time{})
		}
		if done {
			return
		}
		select {
		case <-r.Context().Done():
			return
		case <-notify:
		}
	}
}

func (s *Server) cancelSessionChat(ctx context.Context, session string) error {
	s.turnMu.Lock()
	turn := s.turns[session]
	s.turnMu.Unlock()
	if turn == nil {
		return nil
	}
	turn.mu.Lock()
	stream := turn.stream
	turn.mu.Unlock()
	if stream == nil {
		return nil
	}
	stream.cancel()
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	for {
		stream.mu.Lock()
		done, notify := stream.done, stream.notify
		stream.mu.Unlock()
		if done {
			return nil
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("대화 작업 중지를 기다리고 있습니다. 잠시 후 다시 삭제해 주세요")
		case <-notify:
		}
	}
}
