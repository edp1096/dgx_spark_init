// Package pluginsdk implements Talk's local, newline-delimited JSON-RPC 2.0
// protocol. It uses only the Go standard library.
package pluginsdk

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
)

const MaxMessage = 2 << 20
const Protocol = 1

type Handler func(context.Context, string, json.RawMessage) (any, error)
type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}
type message struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      string          `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
}
type Peer struct {
	writer   io.Writer
	writeMu  sync.Mutex
	mu       sync.Mutex
	pending  map[string]chan message
	incoming map[string]context.CancelFunc
	ctx      context.Context
	cancel   context.CancelFunc
	seq      atomic.Uint64
	done     chan struct{}
}

func NewPeer(ctx context.Context, r io.Reader, w io.Writer, handler Handler) *Peer {
	ctx, cancel := context.WithCancel(ctx)
	p := &Peer{writer: w, pending: map[string]chan message{}, incoming: map[string]context.CancelFunc{}, ctx: ctx, cancel: cancel, done: make(chan struct{})}
	context.AfterFunc(ctx, func() {
		if c, ok := r.(io.Closer); ok {
			_ = c.Close()
		}
		if c, ok := w.(io.Closer); ok {
			_ = c.Close()
		}
	})
	go p.read(r, handler)
	return p
}
func (p *Peer) Done() <-chan struct{} { return p.done }
func (p *Peer) Close()                { p.cancel() }
func (p *Peer) send(m message) error {
	m.JSONRPC = "2.0"
	b, err := json.Marshal(m)
	if err != nil {
		return err
	}
	if len(b) > MaxMessage {
		return errors.New("RPC message too large")
	}
	p.writeMu.Lock()
	defer p.writeMu.Unlock()
	_, err = p.writer.Write(append(b, '\n'))
	return err
}
func (p *Peer) Call(ctx context.Context, method string, in, out any) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if p.ctx.Err() != nil {
		return errors.New("plugin connection closed")
	}
	b, err := json.Marshal(in)
	if err != nil {
		return err
	}
	id := fmt.Sprint(p.seq.Add(1))
	ch := make(chan message, 1)
	p.mu.Lock()
	p.pending[id] = ch
	p.mu.Unlock()
	defer func() { p.mu.Lock(); delete(p.pending, id); p.mu.Unlock() }()
	// The reader must continue consuming replies while a pipe write is blocked.
	sent := make(chan error, 1)
	go func() { sent <- p.send(message{ID: id, Method: method, Params: b}) }()
	select {
	case err = <-sent:
		if err != nil {
			return err
		}
	case <-ctx.Done():
		// The request may already be handled before Write's completion reaches
		// this goroutine. Send cancellation after that write, never before it.
		go func() {
			if err := <-sent; err == nil {
				_ = p.send(message{Method: "$/cancel", Params: json.RawMessage(fmt.Sprintf(`{"id":%q}`, id))})
			}
		}()
		return ctx.Err()
	case <-p.ctx.Done():
		return errors.New("plugin connection closed")
	}
	select {
	case m := <-ch:
		if m.Error != nil {
			return errors.New(m.Error.Message)
		}
		if out != nil {
			return json.Unmarshal(m.Result, out)
		}
		return nil
	case <-ctx.Done():
		// Caller owns hard termination if a peer ignores cancellation.
		go p.send(message{Method: "$/cancel", Params: json.RawMessage(fmt.Sprintf(`{"id":%q}`, id))})
		return ctx.Err()
	case <-p.ctx.Done():
		return errors.New("plugin connection closed")
	}
}
func (p *Peer) read(r io.Reader, handler Handler) {
	defer close(p.done)
	defer p.cancel()
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 4096), MaxMessage+1)
	slots := make(chan struct{}, 16)
	for scanner.Scan() {
		var m message
		if len(scanner.Bytes()) > MaxMessage || json.Unmarshal(scanner.Bytes(), &m) != nil || m.JSONRPC != "2.0" {
			return
		}
		if m.Method == "" {
			if m.ID == "" || (m.Error == nil && len(m.Result) == 0) || (m.Error != nil && len(m.Result) != 0) {
				return
			}
			p.mu.Lock()
			ch := p.pending[m.ID]
			p.mu.Unlock()
			if ch != nil {
				select {
				case ch <- m:
				default:
					return
				}
			}
			continue
		}
		if m.Method == "$/cancel" {
			var v struct {
				ID string `json:"id"`
			}
			if json.Unmarshal(m.Params, &v) != nil {
				return
			}
			p.mu.Lock()
			c := p.incoming[v.ID]
			p.mu.Unlock()
			if c != nil {
				c()
			}
			continue
		}
		if m.Error != nil || len(m.Result) != 0 || m.ID == "" || len(m.ID) > 128 || len(m.Method) > 128 {
			return
		}
		select {
		case slots <- struct{}{}:
		default:
			return
		}
		ctx, cancel := context.WithCancel(p.ctx)
		p.mu.Lock()
		if p.incoming[m.ID] != nil {
			p.mu.Unlock()
			cancel()
			return
		}
		p.incoming[m.ID] = cancel
		p.mu.Unlock()
		go func(m message) {
			defer func() { cancel(); p.mu.Lock(); delete(p.incoming, m.ID); p.mu.Unlock(); <-slots }()
			var out any
			var err error
			func() {
				defer func() {
					if v := recover(); v != nil {
						err = fmt.Errorf("plugin handler panic: %v", v)
					}
				}()
				if handler == nil {
					err = errors.New("method unavailable")
				} else {
					out, err = handler(ctx, m.Method, m.Params)
				}
			}()
			reply := message{ID: m.ID}
			if err == nil {
				reply.Result, err = json.Marshal(out)
			}
			if err != nil {
				reply.Error = &rpcError{Code: -32000, Message: err.Error()}
			}
			if p.send(reply) != nil {
				p.cancel()
			}
		}(m)
	}
}
