package asr

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"mime/multipart"
	"net/http"
	"os"
	"sort"
	"strings"
)

type Word struct {
	Text  string  `json:"word"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
}
type SpeakerSegment struct {
	Start   float64 `json:"start"`
	End     float64 `json:"end"`
	Speaker int     `json:"speaker"`
}
type Turn struct {
	Start    float64 `json:"start"`
	End      float64 `json:"end"`
	Speakers []int   `json:"speakers"`
	Text     string  `json:"text"`
}

func SpeakerLabel(ids []int) string {
	if len(ids) == 0 {
		return "화자 미확인"
	}
	labels := make([]string, len(ids))
	for i, id := range ids {
		labels[i] = fmt.Sprintf("화자 %d", id)
	}
	if len(ids) > 1 {
		return strings.Join(labels, "·") + " (겹침·구분 불확실)"
	}
	return labels[0]
}

func (c *Client) diarize(ctx context.Context, audio *os.File) ([]SpeakerSegment, error) {
	if _, err := audio.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}
	reader, writer := io.Pipe()
	form := multipart.NewWriter(writer)
	done := make(chan error, 1)
	go func() {
		err := form.WriteField("mode", "streaming")
		if err == nil {
			var part io.Writer
			part, err = form.CreateFormFile("file", "audio.wav")
			if err == nil {
				_, err = io.Copy(part, audio)
			}
		}
		if closeErr := form.Close(); err == nil {
			err = closeErr
		}
		writer.CloseWithError(err)
		done <- err
	}()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.cfg.Endpoint+"/v1/audio/diarizations", reader)
	if err != nil {
		reader.CloseWithError(err)
		<-done
		return nil, err
	}
	req.Header.Set("Content-Type", form.FormDataContentType())
	resp, err := c.http.Do(req)
	reader.Close()
	writeErr := <-done
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ASR 화자 API HTTP %d", resp.StatusCode)
	}
	if writeErr != nil {
		return nil, writeErr
	}
	var payload struct {
		Segments []SpeakerSegment `json:"segments"`
	}
	if err = json.NewDecoder(io.LimitReader(resp.Body, 8<<20)).Decode(&payload); err != nil {
		return nil, err
	}
	for _, s := range payload.Segments {
		if s.Speaker < 1 || s.Speaker > 8 || !validTime(s.Start, s.End) {
			return nil, fmt.Errorf("invalid speaker segment")
		}
	}
	if len(payload.Segments) == 0 {
		return nil, fmt.Errorf("음성 화자 구간이 없습니다")
	}
	return payload.Segments, nil
}
func validTime(start, end float64) bool {
	return !math.IsNaN(start) && !math.IsNaN(end) && !math.IsInf(start, 0) && !math.IsInf(end, 0) && start >= 0 && end > start
}

// Preserve ambiguity: a word spanning two active speakers is never assigned
// confidently to one of them. Labels are local to this recording, arrival ordered.
func alignSpeakers(words []Word, segments []SpeakerSegment) []Turn {
	sort.SliceStable(segments, func(i, j int) bool { return segments[i].Start < segments[j].Start })
	ids := map[int]int{}
	for _, s := range segments {
		if _, ok := ids[s.Speaker]; !ok {
			ids[s.Speaker] = len(ids) + 1
		}
	}
	var turns []Turn
	words = append([]Word(nil), words...)
	sort.SliceStable(words, func(i, j int) bool { return words[i].Start < words[j].Start })
	next := 0
	candidates := []SpeakerSegment{}
	for _, w := range words {
		if !validTime(w.Start, w.End) || strings.TrimSpace(w.Text) == "" {
			continue
		}
		active := map[int]bool{}
		for next < len(segments) && segments[next].Start < w.End {
			candidates = append(candidates, segments[next])
			next++
		}
		kept := candidates[:0]
		for _, s := range candidates {
			if s.End > w.Start {
				kept = append(kept, s)
			}
		}
		candidates = kept
		for _, s := range candidates {
			overlap := math.Min(w.End, s.End) - math.Max(w.Start, s.Start)
			if overlap > 0 {
				active[ids[s.Speaker]] = true
			}
		}
		speakers := make([]int, 0, len(active))
		for id := range active {
			speakers = append(speakers, id)
		}
		sort.Ints(speakers)
		// Limit readable turns to 12 seconds, and preserve pauses and speaker changes.
		if n := len(turns); n > 0 && fmt.Sprint(turns[n-1].Speakers) == fmt.Sprint(speakers) && w.Start-turns[n-1].End < 1.5 && w.End-turns[n-1].Start <= 12 {
			turns[n-1].End = w.End
			turns[n-1].Text += " " + strings.TrimSpace(w.Text)
		} else {
			turns = append(turns, Turn{Start: w.Start, End: w.End, Speakers: speakers, Text: strings.TrimSpace(w.Text)})
		}
	}
	return turns
}
