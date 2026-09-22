package pluginsdk

import (
	"context"
	"encoding/json"
	"io"
	"net"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestBidirectionalAndCancellation(t *testing.T) {
	a, b := net.Pipe()
	defer a.Close()
	defer b.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	p := NewPeer(ctx, a, a, func(_ context.Context, _ string, raw json.RawMessage) (any, error) { return raw, nil })
	started := make(chan struct{})
	stopped := make(chan struct{})
	var q *Peer
	ready := make(chan struct{})
	q = NewPeer(ctx, b, b, func(ctx context.Context, method string, raw json.RawMessage) (any, error) {
		<-ready
		if method == "hang" {
			close(started)
			<-ctx.Done()
			close(stopped)
			return nil, ctx.Err()
		}
		var out json.RawMessage
		err := q.Call(ctx, "host.echo", raw, &out)
		return out, err
	})
	close(ready)
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			var out int
			if err := p.Call(ctx, "echo", i, &out); err != nil || out != i {
				t.Errorf("echo %d %d %v", i, out, err)
			}
		}(i)
	}
	wg.Wait()
	c, stop := context.WithCancel(ctx)
	result := make(chan error, 1)
	go func() { result <- p.Call(c, "hang", nil, nil) }()
	<-started
	stop()
	if <-result == nil {
		t.Fatal("cancel succeeded")
	}
	select {
	case <-stopped:
	case <-ctx.Done():
		t.Fatal("cancel not forwarded")
	}
}
func TestMalformedAndOversizedMessages(t *testing.T) {
	for _, raw := range []string{`{"jsonrpc":"1.0","method":"hello","id":"1"}` + "\n", `{"jsonrpc":"2.0","id":"1","result":{},"error":{"code":1,"message":"bad"}}` + "\n", strings.Repeat("x", MaxMessage+2)} {
		p := NewPeer(context.Background(), strings.NewReader(raw), &strings.Builder{}, nil)
		select {
		case <-p.Done():
		case <-time.After(time.Second):
			t.Fatal("invalid frame hung peer")
		}
	}
}

func TestCanceledCallDoesNotSend(t *testing.T) {
	r, w := io.Pipe()
	defer w.Close()
	var output strings.Builder
	p := NewPeer(context.Background(), r, &output, nil)
	defer p.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := p.Call(ctx, "operation", struct{}{}, nil); err != context.Canceled {
		t.Fatal(err)
	}
	if output.Len() != 0 {
		t.Fatal("already canceled request was sent")
	}
}
