package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/crypto/ssh"
	"golang.org/x/crypto/ssh/knownhosts"
)

var keyIDPattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`)

type api struct {
	cfg          config
	sem          chan struct{}
	knownHostsMu sync.Mutex
	keysMu       sync.Mutex
}

type targetRequest struct {
	Host  string `json:"host"`
	Port  int    `json:"port"`
	User  string `json:"user"`
	KeyID string `json:"key_id"`
}

type execRequest struct {
	targetRequest
	Command        string `json:"command"`
	TimeoutSeconds int    `json:"timeout_seconds"`
}

type streamEvent struct {
	Type       string `json:"type"`
	Data       string `json:"data,omitempty"`
	ExitCode   *int   `json:"exit_code,omitempty"`
	DurationMS int64  `json:"duration_ms,omitempty"`
	Truncated  bool   `json:"truncated,omitempty"`
	Error      string `json:"error,omitempty"`
}

type observedHostKey struct {
	Fingerprint string `json:"fingerprint"`
	PublicKey   string `json:"public_key"`
}

type hostKeyError struct {
	Err      error
	Observed observedHostKey
}

func (e *hostKeyError) Error() string { return e.Err.Error() }
func (e *hostKeyError) Unwrap() error { return e.Err }

func newAPI(cfg config) *api { return &api{cfg: cfg, sem: make(chan struct{}, cfg.MaxConcurrency)} }

func (a *api) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", a.health)
	mux.HandleFunc("POST /v1/ssh/check", a.check)
	mux.HandleFunc("POST /v1/ssh/trust", a.trust)
	mux.HandleFunc("POST /v1/ssh/exec", a.execute)
	mux.HandleFunc("GET /v1/ssh/keys", a.listKeys)
	mux.HandleFunc("POST /v1/ssh/keys/generate", a.generateKey)
	mux.HandleFunc("POST /v1/ssh/keys/import", a.importKey)
	mux.HandleFunc("DELETE /v1/ssh/keys/{id}", a.deleteKey)
	return requestLog(mux)
}

func (a *api) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"status": "ok", "active": len(a.sem), "max_concurrency": cap(a.sem),
		"known_hosts": a.cfg.KnownHostsPath, "key_dir": a.cfg.KeyDir,
	})
}

func (a *api) check(w http.ResponseWriter, r *http.Request) {
	var req targetRequest
	if err := decodeJSON(w, r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), nil)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 15*time.Second)
	defer cancel()
	client, err := a.dial(ctx, req)
	if err != nil {
		a.writeConnectError(w, err)
		return
	}
	defer client.Close()
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok"})
}

func (a *api) trust(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Host      string `json:"host"`
		Port      int    `json:"port"`
		PublicKey string `json:"public_key"`
	}
	if err := decodeJSON(w, r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), nil)
		return
	}
	address, err := targetAddress(req.Host, req.Port)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), nil)
		return
	}
	wanted, _, _, _, err := ssh.ParseAuthorizedKey([]byte(strings.TrimSpace(req.PublicKey)))
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid SSH public key", nil)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 15*time.Second)
	defer cancel()
	observed, err := scanHostKey(ctx, address)
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error(), nil)
		return
	}
	if !bytes.Equal(wanted.Marshal(), observed.Marshal()) {
		writeError(w, http.StatusConflict, "the SSH host key changed before it was trusted", map[string]any{"fingerprint": ssh.FingerprintSHA256(observed)})
		return
	}
	line := knownhosts.Line([]string{knownhosts.Normalize(address)}, observed)
	if err := a.withStore(func(s *keyStore) error { return s.trust(line) }); err != nil {
		writeError(w, http.StatusConflict, err.Error(), nil)
		return
	}

	writeJSON(w, http.StatusOK, map[string]any{"status": "trusted", "fingerprint": ssh.FingerprintSHA256(observed)})
}

func (a *api) execute(w http.ResponseWriter, r *http.Request) {
	var req execRequest
	if err := decodeJSON(w, r, &req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), nil)
		return
	}
	req.Command = strings.TrimSpace(req.Command)
	if req.Command == "" || len(req.Command) > 8192 || strings.ContainsRune(req.Command, 0) {
		writeError(w, http.StatusBadRequest, "command must contain 1 to 8192 valid characters", nil)
		return
	}
	select {
	case a.sem <- struct{}{}:
		defer func() { <-a.sem }()
	case <-r.Context().Done():
		return
	}
	timeout := a.cfg.CommandTimeout
	if req.TimeoutSeconds > 0 && time.Duration(req.TimeoutSeconds)*time.Second < timeout {
		timeout = time.Duration(req.TimeoutSeconds) * time.Second
	}
	ctx, cancel := context.WithTimeout(r.Context(), timeout)
	defer cancel()
	client, err := a.dial(ctx, req.targetRequest)
	if err != nil {
		a.writeConnectError(w, err)
		return
	}
	defer client.Close()
	session, err := client.NewSession()
	if err != nil {
		writeError(w, http.StatusBadGateway, "create SSH session: "+err.Error(), nil)
		return
	}
	defer session.Close()
	stdout, err := session.StdoutPipe()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), nil)
		return
	}
	stderr, err := session.StderrPipe()
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error(), nil)
		return
	}
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	flusher, _ := w.(http.Flusher)
	encoder := json.NewEncoder(w)
	emit := func(event streamEvent) bool {
		if err := encoder.Encode(event); err != nil {
			return false
		}
		if flusher != nil {
			flusher.Flush()
		}
		return true
	}
	started := time.Now()
	if !emit(streamEvent{Type: "start"}) {
		return
	}
	if err := session.Start(req.Command); err != nil {
		_ = emit(streamEvent{Type: "exit", Error: "start command: " + err.Error()})
		return
	}
	type chunk struct{ stream, data string }
	chunks := make(chan chunk, 16)
	var readers sync.WaitGroup
	readPipe := func(name string, reader io.Reader) {
		defer readers.Done()
		buffer := make([]byte, 8192)
		for {
			n, readErr := reader.Read(buffer)
			if n > 0 {
				select {
				case chunks <- chunk{name, string(buffer[:n])}:
				case <-ctx.Done():
					return
				}
			}
			if readErr != nil {
				return
			}
		}
	}
	readers.Add(2)
	go readPipe("stdout", stdout)
	go readPipe("stderr", stderr)
	waitResult := make(chan error, 1)
	go func() {
		waitResult <- session.Wait()
		readers.Wait()
		close(chunks)
	}()
	go func() {
		<-ctx.Done()
		_ = client.Close()
	}()
	var outputBytes int64
	truncated := false
	for item := range chunks {
		remaining := a.cfg.MaxOutputBytes - outputBytes
		if remaining <= 0 {
			truncated = true
			continue
		}
		data := item.data
		if int64(len(data)) > remaining {
			data = data[:remaining]
			truncated = true
		}
		outputBytes += int64(len(data))
		if !emit(streamEvent{Type: item.stream, Data: data}) {
			return
		}
	}
	waitErr := <-waitResult
	exitCode := 0
	errorMessage := ""
	if waitErr != nil {
		var exitErr *ssh.ExitError
		if errors.As(waitErr, &exitErr) {
			exitCode = exitErr.ExitStatus()
			errorMessage = waitErr.Error()
		} else if ctx.Err() != nil {
			exitCode = -1
			errorMessage = ctx.Err().Error()
		} else {
			exitCode = -1
			errorMessage = waitErr.Error()
		}
	}
	_ = emit(streamEvent{Type: "exit", ExitCode: &exitCode, DurationMS: time.Since(started).Milliseconds(), Truncated: truncated, Error: errorMessage})
}

func (a *api) dial(ctx context.Context, req targetRequest) (*ssh.Client, error) {
	address, err := targetAddress(req.Host, req.Port)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(req.User) == "" || len(req.User) > 128 {
		return nil, errors.New("SSH user is required")
	}
	if !keyIDPattern.MatchString(req.KeyID) {
		return nil, errors.New("invalid SSH key id")
	}
	var signer ssh.Signer
	var knownCallback ssh.HostKeyCallback
	err = a.withStore(func(s *keyStore) error {
		key, ok := s.m.Keys[req.KeyID]
		if !ok {
			return fmt.Errorf("SSH key %q is unavailable", req.KeyID)
		}
		data, err := os.ReadFile(s.objectPath(key.Hash))
		if err != nil {
			return err
		}
		if hashBytes(data) != key.Hash {
			return errors.New("key hash mismatch")
		}
		signer, err = ssh.ParsePrivateKey(data)
		if err != nil {
			return err
		}
		trustPath := s.objectPath(s.m.KnownHosts)
		if s.m.KnownHosts == "" {
			trustPath = filepath.Join(a.cfg.KeyDir, ".empty-known-hosts")
			if err = atomicPrivate(trustPath, nil); err != nil {
				return err
			}
		}
		knownCallback, err = knownhosts.New(trustPath)
		return err
	})
	if err != nil {
		return nil, err
	}

	var observed ssh.PublicKey
	var hostKeyCallbackErr error
	callback := func(hostname string, remote net.Addr, key ssh.PublicKey) error {
		observed = key
		hostKeyCallbackErr = knownCallback(hostname, remote, key)
		return hostKeyCallbackErr
	}
	sshConfig := &ssh.ClientConfig{User: strings.TrimSpace(req.User), Auth: []ssh.AuthMethod{ssh.PublicKeys(signer)}, HostKeyCallback: callback, Timeout: 15 * time.Second}
	dialer := net.Dialer{Timeout: 15 * time.Second}
	connection, err := dialer.DialContext(ctx, "tcp", address)
	if err != nil {
		return nil, fmt.Errorf("dial SSH server: %w", err)
	}
	conn, channels, requests, err := ssh.NewClientConn(connection, address, sshConfig)
	if err != nil {
		connection.Close()
		var keyErr *knownhosts.KeyError
		if observed != nil && errors.As(hostKeyCallbackErr, &keyErr) && len(keyErr.Want) == 0 {
			return nil, &hostKeyError{Err: fmt.Errorf("verify SSH host key: %w", hostKeyCallbackErr), Observed: observedHostKey{Fingerprint: ssh.FingerprintSHA256(observed), PublicKey: strings.TrimSpace(string(ssh.MarshalAuthorizedKey(observed)))}}
		}
		if hostKeyCallbackErr != nil {
			return nil, fmt.Errorf("verify SSH host key: %w", hostKeyCallbackErr)
		}
		return nil, fmt.Errorf("SSH handshake: %w", err)
	}
	return ssh.NewClient(conn, channels, requests), nil
}

func scanHostKey(ctx context.Context, address string) (ssh.PublicKey, error) {
	connection, err := (&net.Dialer{Timeout: 15 * time.Second}).DialContext(ctx, "tcp", address)
	if err != nil {
		return nil, err
	}
	defer connection.Close()
	var observed ssh.PublicKey
	captured := errors.New("host key captured")
	sshConfig := &ssh.ClientConfig{User: "sparktalk-key-scan", HostKeyCallback: func(_ string, _ net.Addr, key ssh.PublicKey) error {
		observed = key
		return captured
	}, Timeout: 15 * time.Second}
	_, _, _, err = ssh.NewClientConn(connection, address, sshConfig)
	if observed == nil {
		return nil, fmt.Errorf("scan SSH host key: %w", err)
	}
	return observed, nil
}

func targetAddress(host string, port int) (string, error) {
	host = strings.TrimSpace(host)
	if host == "" || len(host) > 253 || strings.ContainsAny(host, " /\\\t\r\n") {
		return "", errors.New("invalid SSH host")
	}
	if port == 0 {
		port = 22
	}
	if port < 1 || port > 65535 {
		return "", errors.New("invalid SSH port")
	}
	return net.JoinHostPort(host, strconv.Itoa(port)), nil
}

func (a *api) writeConnectError(w http.ResponseWriter, err error) {
	var keyErr *hostKeyError
	if errors.As(err, &keyErr) {
		writeError(w, http.StatusConflict, err.Error(), map[string]any{"host_key": keyErr.Observed})
		return
	}
	writeError(w, http.StatusBadGateway, err.Error(), nil)
}

func decodeJSON(w http.ResponseWriter, r *http.Request, target any) error {
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 32*1024))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return errors.New("invalid JSON request")
	}
	return nil
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func writeError(w http.ResponseWriter, status int, message string, extra map[string]any) {
	payload := map[string]any{"error": message}
	for key, value := range extra {
		payload[key] = value
	}
	writeJSON(w, status, payload)
}

func requestLog(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		started := time.Now()
		next.ServeHTTP(w, r)
		fmt.Printf("%s %s %s\n", r.Method, r.URL.Path, time.Since(started).Round(time.Millisecond))
	})
}
