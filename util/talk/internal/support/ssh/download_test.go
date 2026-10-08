package ssh

import (
	"context"
	"crypto/sha256"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sparktalk/internal/attachment"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestDownloadIntegrity(t *testing.T) {
	for _, mode := range []string{"ok", "truncated", "checksum", "oversize", "filename", "error", "cancel"} {
		t.Run(mode, func(t *testing.T) {
			data := []byte("binary\x00\xffpayload")
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if mode == "cancel" {
					<-r.Context().Done()
					return
				}
				if mode == "error" {
					http.Error(w, "missing file", 502)
					return
				}
				if r.URL.Path != "/v1/ssh/download" || r.Method != "POST" {
					t.Error("wrong route")
				}
				w.Header().Set("Content-Disposition", `attachment; filename="result.zip"`)
				w.Header().Set("Content-Length", strconv.Itoa(len(data)))
				w.Header().Set("X-Content-SHA256", fmt.Sprintf("%x", sha256.Sum256(data)))
				if mode == "checksum" {
					w.Header().Set("X-Content-SHA256", strings.Repeat("0", 64))
				}
				if mode == "filename" {
					w.Header().Set("Content-Disposition", `attachment; filename="other.zip"`)
				}
				if mode == "oversize" {
					w.Header().Set("Content-Length", strconv.FormatInt((attachment.Limits{}).MaxBytes()+1, 10))
				}
				if mode == "truncated" {
					data = data[:3]
				}
				_, _ = w.Write(data)
			}))
			defer server.Close()
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			if mode == "cancel" {
				cancel()
			}
			result, err := New(server.URL).Download(ctx, DownloadRequest{Path: "/remote/result.zip"})
			var got []byte
			if err == nil {
				got, err = io.ReadAll(result.Body)
				result.Body.Close()
			}
			if mode == "ok" {
				if err != nil || string(got) != string(data) {
					t.Fatalf("%+v %v", result, err)
				}
			} else if err == nil {
				t.Fatalf("invalid transfer accepted: %+v %v", result, err)
			}
		})
	}
}
