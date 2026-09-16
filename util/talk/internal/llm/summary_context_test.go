package llm

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

type summaryRoundTripFunc func(*http.Request) (*http.Response, error)

func (f summaryRoundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestSummaryHonorsLongCallerDeadline(t *testing.T) {
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(6*time.Hour))
	defer cancel()
	expected, _ := ctx.Deadline()
	client := New("http://model.invalid", "model", "")
	client.http.Transport = summaryRoundTripFunc(func(r *http.Request) (*http.Response, error) {
		got, ok := r.Context().Deadline()
		if !ok || !got.Equal(expected) {
			t.Fatalf("summary shortened caller deadline: %v", got)
		}
		text := ""
		for _, h := range []string{"Objective", "Decisions", "Constraints", "Facts", "Artifacts", "Completed", "Unresolved", "Next Steps"} {
			text += "## " + h + "\nNone.\n"
		}
		body, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"finish_reason": "stop", "message": map[string]any{"content": text}}}})
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(string(body))), Header: make(http.Header), Request: r}, nil
	})
	if _, err := client.SummarizeContext(ctx, "model", "", "transcript"); err != nil {
		t.Fatal(err)
	}
}

func TestSummaryRemainsCancelable(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	started := make(chan struct{})
	done := make(chan error, 1)
	client := New("http://model.invalid", "model", "")
	client.http.Transport = summaryRoundTripFunc(func(r *http.Request) (*http.Response, error) {
		close(started)
		<-r.Context().Done()
		return nil, r.Context().Err()
	})
	go func() { _, err := client.SummarizeContext(ctx, "model", "", "transcript"); done <- err }()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("request not started")
	}
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("cancellation lost: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("summary ignored cancellation")
	}
}
