package plugins

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
	"time"

	"sparktalk/pluginsdk"
)

type processPlugin struct {
	shutdown     func()
	expectedHash string
	mu           sync.Mutex
	cmd          *exec.Cmd
	peer         *pluginsdk.Peer
	done         chan struct{}
	stop         context.CancelFunc
	manifest     Manifest
	executable   string
	diagnostic   *boundedLog
	exitErr      error
}

func externalDefinition(manifest Manifest, path, expectedHash string) Definition {
	p := &processPlugin{manifest: manifest, executable: path, expectedHash: expectedHash}
	return Definition{Health: p.health, Manifest: manifest, Start: p.start, Stop: p.close, Handle: p.handle}
}
func (p *processPlugin) start(ctx context.Context, h Host) error {
	if p.expectedHash != "" {
		digest, err := fileDigest(p.executable)
		if err != nil {
			return err
		}
		if digest != p.expectedHash {
			return fmt.Errorf("installed executable checksum mismatch")
		}
	}
	self, err := os.Executable()
	if err != nil {
		return err
	}
	cmd := exec.Command(self, "--talk-plugin-sandbox-v1", p.executable)
	cmd.Env = []string{"LANG=C.UTF-8"}
	if err = configurePluginProcess(cmd); err != nil {
		return err
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		stdin.Close()
		return err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		stdin.Close()
		stdout.Close()
		return err
	}
	if err = cmd.Start(); err != nil {
		stdin.Close()
		stdout.Close()
		stderr.Close()
		return err
	}
	life, cancel := context.WithCancel(context.Background())
	peer := pluginsdk.NewPeer(life, stdout, stdin, func(callCtx context.Context, method string, raw json.RawMessage) (any, error) {
		return dispatchHost(callCtx, h, method, raw)
	})
	done := make(chan struct{})
	p.mu.Lock()
	p.cmd = cmd
	p.peer = peer
	p.done = done
	p.stop = cancel
	p.mu.Unlock()
	log := &boundedLog{}
	p.mu.Lock()
	p.diagnostic = log
	p.mu.Unlock()
	go func() { io.Copy(log, stderr) }() // Drain without retaining unbounded/untrusted logs.
	go func() {
		err := cmd.Wait()
		p.mu.Lock()
		p.exitErr = err
		p.mu.Unlock()
		cancel()
		stdin.Close()
		peer.Close()
		close(done)
	}()
	var stopOnce sync.Once
	shutdown := func() {
		stopOnce.Do(func() {
			stopCtx, c := context.WithTimeout(context.Background(), 500*time.Millisecond)
			defer c()
			_ = peer.Call(stopCtx, "stop", struct{}{}, nil)
			cancel()
		})
	}
	p.mu.Lock()
	p.shutdown = shutdown
	p.mu.Unlock()
	go func() {
		select {
		case <-ctx.Done():
			shutdown()
		case <-life.Done():
		case <-peer.Done():
			cancel()
		}
		_ = cmd.Process.Kill()
	}()
	var hello struct {
		Protocol int    `json:"protocol"`
		ID       string `json:"id"`
		Version  string `json:"version"`
	}
	handshake, c := context.WithTimeout(ctx, 5*time.Second)
	defer c()
	if err = peer.Call(handshake, "hello", struct{}{}, &hello); err == nil && (hello.Protocol != pluginsdk.Protocol || hello.ID != p.manifest.ID || hello.Version != p.manifest.Version) {
		err = fmt.Errorf("plugin handshake mismatch")
	}
	if err == nil {
		err = peer.Call(handshake, "start", map[string]any{"config": h.Config()}, nil)
	}
	if err != nil {
		cancel()
		<-done
		return fmt.Errorf("plugin start/sandbox failed: %w (%s)", err, log.String())
	}
	return nil
}
func (p *processPlugin) close(ctx context.Context) error {
	p.mu.Lock()
	cancel, done, shutdown := p.stop, p.done, p.shutdown
	p.mu.Unlock()
	if cancel == nil {
		return nil
	}
	if shutdown != nil {
		shutdown()
	} else {
		cancel()
	}
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
func (p *processPlugin) handle(ctx context.Context, h Host, op string, r Request) (json.RawMessage, error) {
	p.mu.Lock()
	peer, cancel := p.peer, p.stop
	p.mu.Unlock()
	if peer == nil {
		return nil, ErrDisabled
	}
	var out json.RawMessage
	err := peer.Call(ctx, "handle", map[string]any{"operation": op, "request": r}, &out)
	if ctx.Err() != nil {
		cancel()
	} // No abandoned execution after a timeout/cancel.
	return out, err
}
func dispatchHost(ctx context.Context, h Host, method string, raw json.RawMessage) (any, error) {
	var in struct {
		Key       string          `json:"key"`
		Value     json.RawMessage `json:"value"`
		Service   string          `json:"service"`
		Request   Request         `json:"request"`
		Operation string          `json:"operation"`
		RunID     string          `json:"run_id"`
	}
	if len(raw) > MaxPayload || json.Unmarshal(raw, &in) != nil {
		return nil, fmt.Errorf("invalid host request")
	}
	switch method {
	case "host.ready":
		select {
		case <-h.Ready():
			return true, nil
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	case "host.get":
		return h.Get(ctx, in.Key)
	case "host.put":
		return nil, h.Put(ctx, in.Key, in.Value)
	case "host.delete":
		return nil, h.Delete(ctx, in.Key)
	case "host.call":
		return h.Call(ctx, in.Service, in.Request)
	case "host.submit":
		return h.Submit(ctx, in.Operation, in.Key, in.Request)
	case "host.runs":
		return h.Runs(ctx)
	case "host.cancel":
		return nil, h.Cancel(ctx, in.RunID)
	default:
		return nil, ErrNotFound
	}
}

func (p *processPlugin) health() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.done != nil {
		select {
		case <-p.done:
			return fmt.Errorf("plugin process exited: %v; %s", p.exitErr, p.diagnostic.String())
		default:
		}
	}
	return nil
}

type boundedLog struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (l *boundedLog) Write(b []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	n := len(b)
	remaining := 8192 - l.b.Len()
	if remaining > 0 {
		if len(b) > remaining {
			b = b[:remaining]
		}
		l.b.Write(b)
	}
	return n, nil
}
func (l *boundedLog) String() string { l.mu.Lock(); defer l.mu.Unlock(); return l.b.String() }

func fileDigest(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err = io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
