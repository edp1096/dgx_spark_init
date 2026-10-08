package main

import (
	"context"
	"encoding/json"
	"os"
	"sparktalk/pluginsdk"
)

func main() {
	err := pluginsdk.Serve(context.Background(), pluginsdk.Plugin{ID: "example", Version: "1.0.0", Handle: func(ctx context.Context, c *pluginsdk.Client, op string, r pluginsdk.Request) (json.RawMessage, error) {
		return r.Input, nil
	}})
	if err != nil {
		os.Exit(1)
	}
}
