package main

import (
	"context"
	"crypto/sha256"
	"fmt"
	"io"
	"mime"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/pkg/sftp"
)

const maxTransferBytes int64 = 4096 << 20

type downloadRequest struct {
	targetRequest
	Path           string `json:"path"`
	MaxBytes       int64  `json:"max_bytes"`
	TimeoutSeconds int    `json:"timeout_seconds"`
}

// Transfer bytes over SFTP, never through a shell or command-output stream.
// Stage the complete file before sending HTTP headers so failures cannot look
// like successful attachments. Temporary files are private and always removed.
func (a *api) download(w http.ResponseWriter, r *http.Request) {
	var req downloadRequest
	if err := decodeJSON(w, r, &req); err != nil {
		writeError(w, 400, err.Error(), nil)
		return
	}
	if req.MaxBytes < 1 || req.MaxBytes > maxTransferBytes {
		writeError(w, 400, "max_bytes must be 1 to 4096 MiB", nil)
		return
	}
	if !validDownloadPath(req.Path) {
		writeError(w, 400, "path must be an absolute file path without control characters", nil)
		return
	}
	select {
	case a.sem <- struct{}{}:
		defer func() { <-a.sem }()
	case <-r.Context().Done():
		return
	}
	timeout := 5 * time.Minute
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
	stop := context.AfterFunc(ctx, func() { _ = client.Close() })
	defer stop()
	remote, err := sftp.NewClient(client)
	if err != nil {
		writeError(w, 502, "open SFTP: "+err.Error(), nil)
		return
	}
	defer remote.Close()
	info, err := remote.Lstat(req.Path)
	if err != nil {
		writeError(w, 502, "stat remote file: "+err.Error(), nil)
		return
	}
	if !info.Mode().IsRegular() || info.Size() < 1 || info.Size() > req.MaxBytes {
		writeError(w, 400, fmt.Sprintf("download requires a regular, non-symlink file of 1 byte to %d MiB", req.MaxBytes>>20), nil)
		return
	}
	file, err := remote.Open(req.Path)
	if err != nil {
		writeError(w, 502, "open remote file: "+err.Error(), nil)
		return
	}
	defer file.Close()
	opened, err := file.Stat()
	if err != nil || !opened.Mode().IsRegular() || opened.Size() != info.Size() {
		writeError(w, 409, "remote file changed before download", nil)
		return
	}
	dir := filepath.Join(filepath.Dir(a.cfg.KnownHostsPath), "transfers")
	if err := os.MkdirAll(dir, 0700); err != nil {
		writeError(w, 500, "create transfer staging directory", nil)
		return
	}
	staged, err := os.CreateTemp(dir, "sparktalk-ssh-download-*")
	if err != nil {
		writeError(w, 500, "create transfer staging file", nil)
		return
	}
	defer os.Remove(staged.Name())
	defer staged.Close()
	hash := sha256.New()
	n, err := io.Copy(io.MultiWriter(staged, hash), io.LimitReader(file, info.Size()+1))
	after, statErr := file.Stat()
	if err != nil || ctx.Err() != nil || n != info.Size() || statErr != nil || after.Size() != info.Size() || !after.ModTime().Equal(opened.ModTime()) {
		writeError(w, 502, "remote file transfer failed or file changed; no attachment created", nil)
		return
	}
	if _, err := staged.Seek(0, io.SeekStart); err != nil {
		writeError(w, 500, "read staged file", nil)
		return
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	w.Header().Set("Content-Disposition", mime.FormatMediaType("attachment", map[string]string{"filename": path.Base(req.Path)}))
	w.Header().Set("Content-Length", strconv.FormatInt(n, 10))
	w.Header().Set("X-Content-SHA256", fmt.Sprintf("%x", hash.Sum(nil)))
	w.Header().Set("Cache-Control", "no-store")
	_, _ = io.Copy(w, staged)
}

func validDownloadPath(p string) bool {
	return path.IsAbs(p) && len(p) <= 4096 && !strings.ContainsFunc(p, func(r rune) bool { return r < 32 || r == 127 })
}
