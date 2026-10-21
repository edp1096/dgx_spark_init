package server

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"time"
)

type mediaDownloadStream struct {
	ID              string   `json:"id"`
	Kind            string   `json:"kind"`
	Status          string   `json:"status"`
	DownloadedBytes int64    `json:"downloaded_bytes"`
	TotalBytes      int64    `json:"total_bytes,omitempty"`
	Estimated       bool     `json:"estimated,omitempty"`
	Fragment        int      `json:"fragment,omitempty"`
	Fragments       int      `json:"fragments,omitempty"`
	Speed           float64  `json:"speed_bps,omitempty"`
	ETA             *float64 `json:"eta_seconds"`
	Stalled         bool     `json:"stalled,omitempty"`
}

type mediaDownloadProgress struct {
	Stage   string                `json:"stage"`
	Streams []mediaDownloadStream `json:"streams,omitempty"`
	Elapsed float64               `json:"elapsed_seconds,omitempty"`
}

type mediaProgressKey struct{}
type mediaProgressSink func(mediaDownloadProgress) error

func mediaProgressFrom(ctx context.Context) mediaProgressSink {
	sink, _ := ctx.Value(mediaProgressKey{}).(mediaProgressSink)
	return sink
}

// Poll independently of the binary response: its headers arrive only after
// extraction, both downloads and remuxing. Join before the next SSE producer.
func pollMediaDownload(ctx context.Context, endpoint, id string, sink mediaProgressSink, cancelDownload context.CancelFunc) func() {
	ctx, cancel := context.WithCancel(ctx)
	done := make(chan struct{})
	go func() {
		defer close(done)
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		client := &http.Client{Timeout: 3 * time.Second}
		for {
			req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint+"/v1/source/progress?id="+id, nil)
			if err != nil {
				return
			}
			response, err := client.Do(req)
			if err == nil {
				var progress mediaDownloadProgress
				if response.StatusCode == http.StatusOK && json.NewDecoder(io.LimitReader(response.Body, 64<<10)).Decode(&progress) == nil {
					err = sink(progress)
					if err != nil {
						response.Body.Close()
						cancelDownload()
						return
					}
				}
				response.Body.Close()
			}
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
			}
		}
	}()
	return func() { cancel(); <-done }
}

func newMediaDownloadID() (string, error) {
	var token [16]byte
	if _, err := rand.Read(token[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(token[:]), nil
}
