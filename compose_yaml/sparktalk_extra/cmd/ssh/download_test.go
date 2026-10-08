package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"net"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/pkg/sftp"
	"golang.org/x/crypto/ssh"
	"golang.org/x/crypto/ssh/knownhosts"
)

// Exercise the real SSH handshake and SFTP subsystem, not a mocked file reader.
func TestDownloadSFTP(t *testing.T) {
	a := testStore(t)
	addKey(t, a, "test")
	key, _ := generatePrivateKey()
	signer, err := ssh.ParsePrivateKey(key)
	if err != nil {
		t.Fatal(err)
	}
	cfg := &ssh.ServerConfig{PublicKeyCallback: func(ssh.ConnMetadata, ssh.PublicKey) (*ssh.Permissions, error) { return nil, nil }}
	cfg.AddHostKey(signer)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	go func() {
		for {
			conn, e := listener.Accept()
			if e != nil {
				return
			}
			go func() {
				sc, chans, reqs, e := ssh.NewServerConn(conn, cfg)
				if e != nil {
					conn.Close()
					return
				}
				defer sc.Close()
				go ssh.DiscardRequests(reqs)
				for ch := range chans {
					if ch.ChannelType() != "session" {
						ch.Reject(ssh.UnknownChannelType, "session required")
						continue
					}
					channel, requests, e := ch.Accept()
					if e != nil {
						continue
					}
					go func() {
						defer channel.Close()
						for r := range requests {
							var subsystem struct{ Name string }
							_ = ssh.Unmarshal(r.Payload, &subsystem)
							if r.Type != "subsystem" || subsystem.Name != "sftp" {
								r.Reply(false, nil)
								continue
							}
							r.Reply(true, nil)
							srv, e := sftp.NewServer(channel, sftp.ReadOnly())
							if e == nil {
								_ = srv.Serve()
								srv.Close()
							}
							return
						}
					}()
				}
			}()
		}
	}()
	host, portString, _ := net.SplitHostPort(listener.Addr().String())
	port, _ := strconv.Atoi(portString)
	target := targetRequest{Host: host, Port: port, User: "test", KeyID: "test"}
	dir := t.TempDir()
	filename := filepath.Join(dir, "한글 '$(never-run)' file.zip")
	data := []byte{0, 1, 255, 10, 13, 0, 128}
	if e := os.WriteFile(filename, data, 0600); e != nil {
		t.Fatal(e)
	}
	request := func(p string) *httptest.ResponseRecorder {
		b, _ := json.Marshal(downloadRequest{targetRequest: target, Path: p, MaxBytes: 256 << 20, TimeoutSeconds: 3})
		w := httptest.NewRecorder()
		a.routes().ServeHTTP(w, httptest.NewRequest("POST", "/v1/ssh/download", bytes.NewReader(b)))
		return w
	}
	if w := request(filename); w.Code != 409 {
		t.Fatalf("untrusted key status %d", w.Code)
	}
	line := knownhosts.Line([]string{knownhosts.Normalize(listener.Addr().String())}, signer.PublicKey())
	if err := a.withStore(func(s *keyStore) error { return s.trust(line) }); err != nil {
		t.Fatal(err)
	}
	w := request(filename)
	if w.Code != 200 || !bytes.Equal(w.Body.Bytes(), data) || w.Header().Get("X-Content-SHA256") != fmt.Sprintf("%x", sha256.Sum256(data)) {
		t.Fatalf("download status=%d body=%q", w.Code, w.Body.Bytes())
	}
	empty := filepath.Join(dir, "empty")
	_ = os.WriteFile(empty, nil, 0600)
	link := filepath.Join(dir, "link")
	if err := os.Symlink(filename, link); err != nil {
		t.Fatal(err)
	}
	huge := filepath.Join(dir, "huge")
	f, err := os.Create(huge)
	if err != nil {
		t.Fatal(err)
	}
	_ = f.Truncate(256<<20 + 1)
	_ = f.Close()
	for _, p := range []string{dir, link, empty, huge, filepath.Join(dir, "missing"), "relative.zip", "/tmp/bad\nname"} {
		if w := request(p); w.Code == 200 {
			t.Fatalf("invalid file accepted: %q", p)
		}
	}
	// A changed known host key must fail rather than offering automatic trust.
	other, _ := generatePrivateKey()
	otherSigner, _ := ssh.ParsePrivateKey(other)
	cfg2 := knownhosts.Line([]string{knownhosts.Normalize(listener.Addr().String())}, otherSigner.PublicKey())
	b := testStore(t)
	addKey(t, b, "test")
	if e := b.withStore(func(s *keyStore) error { return s.trust(cfg2) }); e != nil {
		t.Fatal(e)
	}
	a = b
	if w := request(filename); w.Code != 502 {
		t.Fatalf("changed key status=%d", w.Code)
	}
}
