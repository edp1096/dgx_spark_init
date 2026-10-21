package main

import (
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"os/exec"
	"strings"
	"time"
)

const progressPrefix = "SPARKTALK_DOWNLOAD:"

func downloadProgressArgs() []string {
	return []string{"--progress", "--newline", "--no-colors", "--progress-delta", "1", "--progress-template", `download:` + progressPrefix + `{"format_id":%(info.format_id)j,"progress":%(progress)j}`}
}

type downloadStream struct {
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
	updated         time.Time
}

type downloadProgress struct {
	Stage   string           `json:"stage"`
	Streams []downloadStream `json:"streams"`
	Elapsed float64          `json:"elapsed_seconds"`
	started time.Time
	updated time.Time
}

func validProgressID(id string) bool {
	b, err := hex.DecodeString(id)
	return err == nil && len(b) == 16 && len(id) == 32
}

func (a *api) beginDownload(id string) bool {
	if id == "" {
		return true
	}
	a.progressMu.Lock()
	defer a.progressMu.Unlock()
	if a.downloads == nil {
		a.downloads = make(map[string]downloadProgress)
	}
	for key, p := range a.downloads {
		if (p.Stage == "downloaded" || p.Stage == "failed") && time.Since(p.updated) > 10*time.Minute {
			delete(a.downloads, key)
		}
	}
	if _, exists := a.downloads[id]; exists {
		return false
	}
	a.downloads[id] = downloadProgress{Stage: "queued", started: time.Now(), updated: time.Now(), Streams: []downloadStream{}}
	return true
}

func (a *api) updateDownload(id string, update func(*downloadProgress)) {
	if id == "" {
		return
	}
	a.progressMu.Lock()
	defer a.progressMu.Unlock()
	p, ok := a.downloads[id]
	if !ok {
		return
	}
	update(&p)
	p.updated = time.Now()
	a.downloads[id] = p
}

func (a *api) downloadStage(id, stage string) {
	a.updateDownload(id, func(p *downloadProgress) { p.Stage = stage })
}

func (a *api) selectDownloadStreams(id string, info sourceInfo, format string) {
	a.updateDownload(id, func(p *downloadProgress) {
		p.Stage = "downloading"
		for _, selected := range strings.Split(format, "+") {
			for _, f := range info.Formats {
				if f.ID != selected {
					continue
				}
				kind := "video"
				if f.Height <= 0 && (f.VideoCodec == "" || f.VideoCodec == "none") {
					kind = "audio"
				} else if hasFormatAudio(f) {
					kind = "video_audio"
				}
				size := f.FileSize
				if size == 0 {
					size = f.FileSizeApprox
				}
				p.Streams = append(p.Streams, downloadStream{ID: f.ID, Kind: kind, Status: "queued", TotalBytes: size, Estimated: f.FileSize == 0})
			}
		}
	})
}

func (a *api) sourceProgress(w http.ResponseWriter, r *http.Request) {
	id := r.URL.Query().Get("id")
	if !validProgressID(id) {
		http.Error(w, "invalid download ID", 400)
		return
	}
	a.progressMu.Lock()
	p, ok := a.downloads[id]
	p.Streams = append([]downloadStream{}, p.Streams...)
	a.progressMu.Unlock()
	if !ok {
		http.NotFound(w, r)
		return
	}
	p.Elapsed = time.Since(p.started).Seconds()
	for i := range p.Streams {
		stream := &p.Streams[i]
		if stream.Status == "downloading" && time.Since(stream.updated) > 15*time.Second {
			stream.Speed, stream.ETA, stream.Stalled = 0, nil, true
		}
	}
	w.Header().Set("Cache-Control", "no-store")
	writeJSON(w, 200, p)
}

func (a *api) downloadLine(id string, line []byte) {
	var event struct {
		FormatID string `json:"format_id"`
		Progress struct {
			Status     string   `json:"status"`
			Downloaded int64    `json:"downloaded_bytes"`
			Total      int64    `json:"total_bytes"`
			Estimate   float64  `json:"total_bytes_estimate"`
			Speed      float64  `json:"speed"`
			ETA        *float64 `json:"eta"`
			Fragment   int      `json:"fragment_index"`
			Fragments  int      `json:"fragment_count"`
		} `json:"progress"`
	}
	if json.Unmarshal(line, &event) != nil {
		return
	}
	a.updateDownload(id, func(p *downloadProgress) {
		for i := range p.Streams {
			s := &p.Streams[i]
			if s.ID != event.FormatID {
				continue
			}
			e := event.Progress
			s.Status = "downloading"
			if e.Status == "finished" {
				s.Status = "finished"
			}
			s.DownloadedBytes = e.Downloaded
			if e.Total > 0 {
				s.TotalBytes, s.Estimated = e.Total, false
			} else if e.Estimate > 0 && e.Estimate < 1<<62 {
				s.TotalBytes, s.Estimated = int64(e.Estimate), true
			}
			s.Speed, s.ETA = e.Speed, e.ETA
			s.Fragment, s.Fragments = e.Fragment, e.Fragments
			s.updated = time.Now()
			if s.Status == "finished" {
				s.Speed, s.ETA = 0, nil
			}
		}
		allFinished := len(p.Streams) > 0
		for _, s := range p.Streams {
			allFinished = allFinished && s.Status == "finished"
		}
		if allFinished {
			p.Stage = "processing"
		}
	})
}

// Consume structured progress without exposing signed URLs, filenames or raw
// extractor output. Keep diagnostics bounded even on malformed/long lines.
type downloadProgressWriter struct {
	a           *api
	id          string
	pending     []byte
	dropping    bool
	diagnostics limitedBuffer
}

func (w *downloadProgressWriter) Write(data []byte) (int, error) {
	for _, c := range data {
		if c == '\n' || c == '\r' {
			if !w.dropping {
				w.line(w.pending)
			}
			w.pending, w.dropping = w.pending[:0], false
		} else if len(w.pending) < 64<<10 && !w.dropping {
			w.pending = append(w.pending, c)
		} else {
			w.dropping = true
		}
	}
	return len(data), nil
}

func (w *downloadProgressWriter) line(line []byte) {
	if bytes.HasPrefix(line, []byte(progressPrefix)) {
		w.a.downloadLine(w.id, line[len(progressPrefix):])
		return
	}
	w.diagnostics.Write(line)
	w.diagnostics.Write([]byte{'\n'})
}

func (a *api) runDownload(ctx context.Context, id string, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, a.currentYtdlp(), args...)
	stdout := &downloadProgressWriter{a: a, id: id, diagnostics: limitedBuffer{Limit: 8 << 20}}
	stderr := &downloadProgressWriter{a: a, id: id, diagnostics: limitedBuffer{Limit: 64 << 10}}
	cmd.Stdout, cmd.Stderr = stdout, stderr
	err := cmd.Run()
	if len(stdout.pending) > 0 && !stdout.dropping {
		stdout.line(stdout.pending)
	}
	if len(stderr.pending) > 0 && !stderr.dropping {
		stderr.line(stderr.pending)
	}
	return stderr.diagnostics.Bytes(), err
}
