package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/exec"
	"sparktalk/pluginsdk"
	"strings"
	"time"
)

var version = "1.0.0"

func main() {
	err := pluginsdk.Serve(context.Background(), pluginsdk.Plugin{ID: "example", Version: version,
		Start: func(ctx context.Context, c *pluginsdk.Client, config json.RawMessage) error {
			if strings.Contains(string(config), "reject") {
				return fmt.Errorf("rejected configuration")
			}
			return nil
		},
		Migrate: func(ctx context.Context, from, to int, values map[string]json.RawMessage) (map[string]json.RawMessage, error) {
			if version == "3.0.0" {
				values["corrupt"] = json.RawMessage(`true`)
				return nil, fmt.Errorf("migration deliberately failed")
			}
			values["schema"] = json.RawMessage(fmt.Sprint(to))
			return values, nil
		}, Handle: func(ctx context.Context, c *pluginsdk.Client, op string, r pluginsdk.Request) (json.RawMessage, error) {
			switch op {
			case "echo":
				return r.Input, nil
			case "storage":
				var out json.RawMessage
				if err := c.Host(ctx, "put", map[string]any{"key": "example", "value": r.Input}, nil); err != nil {
					return nil, err
				}
				err := c.Host(ctx, "get", map[string]string{"key": "example"}, &out)
				return out, err
			case "isolation":
				_, fileErr := os.ReadFile("/etc/passwd")
				conn, netErr := net.DialTimeout("tcp", "127.0.0.1:9", time.Second)
				if conn != nil {
					conn.Close()
				}
				return json.Marshal(map[string]any{"file_denied": fileErr != nil, "network_error": fmt.Sprint(netErr)})
			case "spawn":
				err := exec.Command(os.Args[0]).Run()
				return json.Marshal(map[string]bool{"denied": err != nil})
			case "secret":
				return json.Marshal(map[string]string{"secret": os.Getenv("TALK_PLUGIN_TEST_SECRET")})
			case "oversize":
				os.Stdout.Write([]byte(strings.Repeat("x", 3<<20)))
				return nil, nil
			case "hang":
				for {
					time.Sleep(time.Hour)
				}
			case "crash":
				os.Exit(7)
			}
			return json.RawMessage(`{}`), nil
		}})
	if err != nil {
		os.Exit(1)
	}
}
