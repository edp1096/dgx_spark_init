package orchestrator

import (
	"context"
	"strings"
	"sync"
)

type recipeProgressKey struct{}

// Keep partial lines private until redaction; stderr progress often uses CR.
type recipeOutput struct {
	mu      sync.Mutex
	pending string
	tail    string
	token   string
	report  func(string)
}

func (w *recipeOutput) emit(line string) {
	if w.token != "" {
		line = strings.ReplaceAll(line, w.token, "[redacted]")
	}
	line = strings.TrimSpace(line)
	if line == "" {
		return
	}
	w.tail += line + "\n"
	if len(w.tail) > 16384 {
		w.tail = w.tail[len(w.tail)-16384:]
	}
	w.report(line)
}
func (w *recipeOutput) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.pending += string(p)
	for {
		i := strings.IndexAny(w.pending, "\r\n")
		if i < 0 {
			break
		}
		w.emit(w.pending[:i])
		w.pending = w.pending[i+1:]
	}
	// Discard overlong unterminated output instead of exposing partial secrets.
	if len(w.pending) > 65536 {
		w.pending = ""
	}
	return len(p), nil
}
func (w *recipeOutput) finish() string {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.emit(w.pending)
	w.pending = ""
	return w.tail
}
func recipeReporter(ctx context.Context) func(string) {
	report, _ := ctx.Value(recipeProgressKey{}).(func(string))
	return report
}
