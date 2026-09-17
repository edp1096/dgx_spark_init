// Package browserbridge connects Talk to the user's explicitly paired Chrome extension.
package browserbridge

import (
	"archive/zip"
	"context"
	"crypto/rand"
	"crypto/subtle"
	"embed"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io/fs"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/websocket"
)

//go:embed extension/*
var assets embed.FS

type Bridge struct {
	protocol int
	mu       sync.Mutex
	sendMu   sync.Mutex
	token    string
	path     string
	conn     *websocket.Conn
	pending  map[string]chan json.RawMessage
}

func New(path string) (*Bridge, error) {
	b := &Bridge{path: path, pending: make(map[string]chan json.RawMessage)}
	data, err := os.ReadFile(path)
	if err != nil && !os.IsNotExist(err) {
		return nil, err
	}
	b.token = strings.TrimSpace(string(data))
	return b, nil
}
func randomID() string {
	var v [32]byte
	if _, err := rand.Read(v[:]); err != nil {
		panic(err)
	}
	return hex.EncodeToString(v[:])
}
func (b *Bridge) Connected() bool      { b.mu.Lock(); defer b.mu.Unlock(); return b.conn != nil }
func (b *Bridge) ProtocolVersion() int { b.mu.Lock(); defer b.mu.Unlock(); return b.protocol }
func (b *Bridge) Register(mux *http.ServeMux) {
	mux.HandleFunc("/api/browser", b.settings)
	mux.HandleFunc("/api/browser/extension.zip", b.download)
	mux.Handle("/api/browser/connect", websocket.Server{Handshake: func(c *websocket.Config, r *http.Request) error {
		origin, err := url.Parse(r.Header.Get("Origin"))
		if err != nil || origin.Scheme != "chrome-extension" || len(origin.Host) != 32 {
			return errors.New("extension origin required")
		}
		return nil
	}, Handler: b.serve})
}
func sameOrigin(r *http.Request) bool {
	u, e := url.Parse(r.Header.Get("Origin"))
	return e == nil && u.Host == r.Host && (u.Scheme == "http" || u.Scheme == "https")
}
func (b *Bridge) settings(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("Content-Type", "application/json")
	b.mu.Lock()
	defer b.mu.Unlock()
	switch r.Method {
	case "GET":
		json.NewEncoder(w).Encode(map[string]any{"connected": b.conn != nil, "paired": b.token != "", "extension_protocol": b.protocol, "update_required": b.conn != nil && b.protocol < 11})
	case "POST", "DELETE":
		if !sameOrigin(r) {
			http.Error(w, "same-origin request required", 403)
			return
		}
		token := ""
		if r.Method == "POST" {
			token = randomID()
		}
		if err := os.MkdirAll(filepath.Dir(b.path), 0700); err != nil {
			http.Error(w, "pairing storage unavailable", 500)
			return
		}
		if err := os.WriteFile(b.path+".tmp", []byte(token), 0600); err != nil {
			http.Error(w, "pairing storage unavailable", 500)
			return
		}
		if err := os.Rename(b.path+".tmp", b.path); err != nil {
			http.Error(w, "pairing storage unavailable", 500)
			return
		}
		b.token = token
		if b.conn != nil {
			b.conn.Close()
			b.conn = nil
			for id, ch := range b.pending {
				close(ch)
				delete(b.pending, id)
			}
		}
		json.NewEncoder(w).Encode(map[string]any{"token": token, "connected": false})
	default:
		w.WriteHeader(405)
	}
}
func (b *Bridge) download(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		w.WriteHeader(405)
		return
	}
	w.Header().Set("Content-Type", "application/zip")
	w.Header().Set("Content-Disposition", `attachment; filename="sparktalk-browser.zip"`)
	z := zip.NewWriter(w)
	defer z.Close()
	fs.WalkDir(assets, "extension", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		v, err := assets.ReadFile(path)
		if err != nil {
			return err
		}
		out, err := z.Create(strings.TrimPrefix(path, "extension/"))
		if err != nil {
			return err
		}
		_, err = out.Write(v)
		return err
	})
}
func (b *Bridge) serve(ws *websocket.Conn) {
	defer ws.Close()
	ws.MaxPayloadBytes = 1 << 20
	ws.SetReadDeadline(time.Now().Add(10 * time.Second))
	var auth struct {
		Token    string `json:"token"`
		Protocol int    `json:"protocol"`
	}
	if websocket.JSON.Receive(ws, &auth) != nil {
		return
	}
	b.mu.Lock()
	if b.token == "" || subtle.ConstantTimeCompare([]byte(auth.Token), []byte(b.token)) != 1 || b.conn != nil {
		b.mu.Unlock()
		return
	}
	b.conn = ws
	b.protocol = auth.Protocol
	b.mu.Unlock()
	b.sendMu.Lock()
	websocket.JSON.Send(ws, map[string]any{"ready": true})
	b.sendMu.Unlock()
	defer func() {
		b.mu.Lock()
		if b.conn == ws {
			b.conn = nil
			for id, ch := range b.pending {
				close(ch)
				delete(b.pending, id)
			}
		}
		b.mu.Unlock()
	}()
	for {
		ws.SetReadDeadline(time.Now().Add(65 * time.Second))
		var msg struct {
			ID     string          `json:"id"`
			Result json.RawMessage `json:"result"`
		}
		if websocket.JSON.Receive(ws, &msg) != nil {
			return
		}
		if msg.ID == "" {
			continue
		}
		b.mu.Lock()
		ch := b.pending[msg.ID]
		if ch != nil {
			ch <- msg.Result
			delete(b.pending, msg.ID)
		}
		b.mu.Unlock()
	}
}
func (b *Bridge) Call(ctx context.Context, action string, args any) (json.RawMessage, error) {
	id := randomID()
	ch := make(chan json.RawMessage, 1)
	b.mu.Lock()
	ws := b.conn
	if ws == nil {
		b.mu.Unlock()
		return nil, errors.New("브라우저 확장이 연결되지 않았습니다. 설정 > 기능 > 브라우저에서 연결하세요.")
	}
	b.pending[id] = ch
	b.mu.Unlock()
	defer func() { b.mu.Lock(); delete(b.pending, id); b.mu.Unlock() }()
	b.sendMu.Lock()
	ws.SetWriteDeadline(time.Now().Add(5 * time.Second))
	expires := time.Now().Add(time.Minute)
	if deadline, ok := ctx.Deadline(); ok {
		expires = deadline
	}
	err := websocket.JSON.Send(ws, map[string]any{"id": id, "action": action, "args": args, "expires": expires.UnixMilli()})
	b.sendMu.Unlock()
	if err != nil {
		return nil, err
	}
	select {
	case v, ok := <-ch:
		if !ok {
			return nil, errors.New("브라우저 연결이 끊겼습니다. 등록 결과는 불확실하며 자동 재시도하지 마세요.")
		}
		return v, nil
	case <-ctx.Done():
		b.sendMu.Lock()
		ws.SetWriteDeadline(time.Now().Add(time.Second))
		websocket.JSON.Send(ws, map[string]any{"cancel": id})
		b.sendMu.Unlock()
		return nil, errors.New("브라우저 응답 대기 종료. 등록 요청이었다면 결과를 확인하기 전 재시도하지 마세요.")
	}
}
