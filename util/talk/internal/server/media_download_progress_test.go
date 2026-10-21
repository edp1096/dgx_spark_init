package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"sparktalk/internal/config"
	"sparktalk/internal/media"
)

func TestMediaDownloadProgressArrivesBeforeBinaryResponse(t *testing.T) {
	var mu sync.Mutex
	id := ""
	received := make(chan struct{})
	var once sync.Once
	remote := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "GET" {
			mu.Lock()
			match := id != "" && id == r.URL.Query().Get("id")
			mu.Unlock()
			if !match {
				http.NotFound(w, r)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"stage":"downloading","streams":[{"id":"video","kind":"video","status":"finished","downloaded_bytes":100},{"id":"audio","kind":"audio","status":"downloading","downloaded_bytes":50,"total_bytes":100,"speed_bps":10,"eta_seconds":5}]}`))
			return
		}
		mu.Lock()
		id = r.Header.Get("X-Media-Download-ID")
		mu.Unlock()
		select {
		case <-received:
		case <-r.Context().Done():
			return
		}
		w.Header().Set("Content-Type", "audio/mpeg")
		w.Header().Set("Content-Disposition", `attachment; filename="test.mp3"`)
		w.Write([]byte("ID3sample audio"))
	}))
	defer remote.Close()
	s := titleTestServer(t)
	var err error
	s.media, err = media.New(t.TempDir() + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	s.cfg = config.Config{ASR: config.ASRConfig{FFmpegEndpoint: remote.URL, Timeout: "5s"}}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	ctx = context.WithValue(ctx, mediaProgressKey{}, mediaProgressSink(func(p mediaDownloadProgress) error {
		if len(p.Streams) != 2 || p.Streams[1].ETA == nil || *p.Streams[1].ETA != 5 {
			t.Errorf("invalid progress: %+v", p)
		}
		once.Do(func() { close(received) })
		return nil
	}))
	item, err := s.importMediaSource(ctx, "https://example.com/watch")
	if err != nil || item.ID == "" {
		t.Fatalf("download failed: %+v %v", item, err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(id) != 32 {
		t.Fatal("missing download correlation ID")
	}
}

func TestMediaDownloadProgressPollStopsAndRedacts(t *testing.T) {
	remote := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"stage":"downloading","streams":[{"id":"a","kind":"audio","status":"downloading","downloaded_bytes":20,"filename":"private-path","url":"private-url"}]}`))
	}))
	defer remote.Close()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	called := make(chan struct{}, 10)
	stop := pollMediaDownload(ctx, remote.URL, "test", func(p mediaDownloadProgress) error {
		raw, _ := json.Marshal(p)
		if strings.Contains(string(raw), "private-") {
			t.Error("leaked raw service data")
		}
		called <- struct{}{}
		return nil
	}, cancel)
	select {
	case <-called:
	case <-time.After(3 * time.Second):
		t.Fatal("no progress")
	}
	stop()
	select {
	case <-called:
		t.Fatal("poll continued after stop")
	default:
	}
}
