package pluginsdk

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sync"
)

// Request carries the initiating Talk conversation and operation input.
type Request struct {
	SessionID string          `json:"session_id,omitempty"`
	Input     json.RawMessage `json:"input"`
}
type HandlerFunc func(context.Context, *Client, string, Request) (json.RawMessage, error)
type Plugin struct {
	Stop    func(context.Context) error
	Migrate func(context.Context, int, int, map[string]json.RawMessage) (map[string]json.RawMessage, error)
	ID      string
	Version string
	Start   func(context.Context, *Client, json.RawMessage) error
	Handle  HandlerFunc
}
type Client struct{ peer *Peer }

func (c *Client) Host(ctx context.Context, method string, input, output any) error {
	return c.peer.Call(ctx, "host."+method, input, output)
}

// Serve reserves stdout for RPC. Use stderr for diagnostic logging.
func Serve(ctx context.Context, plugin Plugin) error {
	workers, cancelWorkers := context.WithCancel(ctx)
	defer cancelWorkers()
	var stopOnce sync.Once
	var stopErr error
	c := &Client{}
	ready := make(chan struct{})
	c.peer = NewPeer(ctx, os.Stdin, os.Stdout, func(ctx context.Context, method string, raw json.RawMessage) (any, error) {
		<-ready
		switch method {
		case "hello":
			return map[string]any{"protocol": Protocol, "id": plugin.ID, "version": plugin.Version}, nil
		case "start":
			var v struct {
				Config json.RawMessage `json:"config"`
			}
			if err := json.Unmarshal(raw, &v); err != nil {
				return nil, err
			}
			if plugin.Start != nil {
				return nil, plugin.Start(workers, c, v.Config)
			}
			return nil, nil
		case "stop":
			stopOnce.Do(func() {
				cancelWorkers()
				if plugin.Stop != nil {
					stopErr = plugin.Stop(ctx)
				}
			})
			return nil, stopErr
		case "migrate":
			var v struct {
				From   int                        `json:"from"`
				To     int                        `json:"to"`
				Values map[string]json.RawMessage `json:"values"`
			}
			if err := json.Unmarshal(raw, &v); err != nil {
				return nil, err
			}
			if plugin.Migrate == nil {
				return nil, fmt.Errorf("migration not implemented")
			}
			return plugin.Migrate(ctx, v.From, v.To, v.Values)
		case "handle":
			var v struct {
				Operation string  `json:"operation"`
				Request   Request `json:"request"`
			}
			if err := json.Unmarshal(raw, &v); err != nil {
				return nil, err
			}
			if plugin.Handle == nil {
				return nil, fmt.Errorf("no handler")
			}
			return plugin.Handle(ctx, c, v.Operation, v.Request)
		default:
			return nil, fmt.Errorf("unknown method: %s", method)
		}
	})
	close(ready)
	select {
	case <-c.peer.Done():
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
