// Package tasklife owns task cancellation and draining, not retry policy.
package tasklife

import (
	"context"
	"errors"
	"sync"
)

var ErrStopping = errors.New("server is shutting down")

// Group's zero value accepts work. Stop permanently closes admission.
type Group struct {
	root       context.Context
	cancelRoot context.CancelFunc
	mu         sync.Mutex
	closed     bool
	next       uint64
	tasks      map[uint64]context.CancelFunc
	done       chan struct{}
}

func (g *Group) Track(parent context.Context) (context.Context, func(), error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed {
		return nil, nil, ErrStopping
	}
	if g.tasks == nil {
		g.tasks = map[uint64]context.CancelFunc{}
		g.done = make(chan struct{})
	}
	ctx, cancel := context.WithCancel(parent)
	g.next++
	id := g.next
	g.tasks[id] = cancel
	var once sync.Once
	finish := func() {
		once.Do(func() {
			cancel()
			g.mu.Lock()
			defer g.mu.Unlock()
			delete(g.tasks, id)
			if g.closed && len(g.tasks) == 0 {
				close(g.done)
			}
		})
	}
	return ctx, finish, nil
}
func (g *Group) Stop() {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.closed {
		return
	}
	g.closed = true
	if g.cancelRoot != nil {
		g.cancelRoot()
	}
	if g.done == nil {
		g.done = make(chan struct{})
	}
	for _, cancel := range g.tasks {
		cancel()
	}
	if len(g.tasks) == 0 {
		close(g.done)
	}
}
func (g *Group) Wait(ctx context.Context) error {
	g.Stop()
	g.mu.Lock()
	done := g.done
	g.mu.Unlock()
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Context supplies the HTTP server's lifetime without counting it as a task.
func (g *Group) Context() context.Context {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.root == nil {
		g.root, g.cancelRoot = context.WithCancel(context.Background())
		if g.closed {
			g.cancelRoot()
		}
	}
	return g.root
}
