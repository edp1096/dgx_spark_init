package server

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
)

type subtitleSpeakerSegment struct {
	Start   float64 `json:"start"`
	End     float64 `json:"end"`
	Speaker int     `json:"speaker"`
}

// A single full-recording pass preserves speaker identity across ASR chunks.
func (s *Server) subtitleSpeakers(jobID, dir string, manifest preparedManifest) ([]subtitleSpeakerSegment, error) {
	audio := filepath.Join(dir, "diarization.wav")
	if err := joinSpeakerAudio(audio, dir, manifest.Segments); err != nil {
		return nil, err
	}
	defer os.Remove(audio)
	output := filepath.Join(dir, "diarization.response.json")
	defer os.Remove(output)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Minute)
	defer cancel()
	go func() {
		ticker := time.NewTicker(500 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				if s.jobCancelled(jobID) {
					cancel()
					return
				}
			}
		}
	}()
	endpoint := strings.TrimRight(s.config().Recognition.DiarizationEndpoint, "/") + "/v1/audio/diarizations"
	if err := s.callMultipartToFileStreamingContext(ctx, endpoint, map[string]string{"mode": "streaming"}, "file", []string{audio}, output); err != nil {
		return nil, err
	}
	f, err := os.Open(output)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var response struct {
		Segments []subtitleSpeakerSegment `json:"segments"`
	}
	if err = json.NewDecoder(io.LimitReader(f, 8<<20)).Decode(&response); err != nil {
		return nil, err
	}
	if len(response.Segments) == 0 {
		return nil, fmt.Errorf("화자 구간이 없습니다")
	}
	for _, x := range response.Segments {
		if x.Speaker < 1 || x.Speaker > 8 || math.IsNaN(x.Start) || math.IsNaN(x.End) || math.IsInf(x.Start, 0) || math.IsInf(x.End, 0) || x.Start < 0 || x.End <= x.Start {
			return nil, fmt.Errorf("잘못된 화자 구간")
		}
	}
	sort.SliceStable(response.Segments, func(i, j int) bool { return response.Segments[i].Start < response.Segments[j].Start })
	ids := map[int]int{}
	for i, x := range response.Segments {
		if ids[x.Speaker] == 0 {
			ids[x.Speaker] = len(ids) + 1
		}
		response.Segments[i].Speaker = ids[x.Speaker]
	}
	return response.Segments, nil
}

// Prepared audio is PCM16 mono 16 kHz WAV. Write at absolute sample offsets,
// not byte-concatenate WAV containers (which would reset timestamps).
func joinSpeakerAudio(output, dir string, segments []preparedSegment) error {
	out, err := os.Create(output)
	if err != nil {
		return err
	}
	defer out.Close()
	header := make([]byte, 44)
	if _, err = out.Write(header); err != nil {
		return err
	}
	var end int64
	for _, seg := range segments {
		if math.IsNaN(seg.Start) || math.IsInf(seg.Start, 0) || seg.Start < 0 || seg.Start > 4*3600 {
			return fmt.Errorf("화자 구분 입력 시간 범위 초과")
		}
		f, err := os.Open(filepath.Join(dir, seg.Name))
		if err != nil {
			return err
		}
		offset, size, err := speakerPCM(f)
		if err != nil {
			f.Close()
			return err
		}
		position := int64(math.Round(seg.Start*16000)) * 2
		if position+size > 512<<20 {
			f.Close()
			return fmt.Errorf("화자 구분 PCM 한도 512MiB 초과")
		}
		if _, err = out.Seek(44+position, io.SeekStart); err == nil {
			_, err = io.CopyN(out, io.NewSectionReader(f, offset, size), size)
		}
		f.Close()
		if err != nil {
			return err
		}
		if position+size > end {
			end = position + size
		}
	}
	if end == 0 {
		return fmt.Errorf("화자 구분용 음성이 없습니다")
	}
	copy(header, "RIFF")
	binary.LittleEndian.PutUint32(header[4:], uint32(end+36))
	copy(header[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(header[16:], 16)
	binary.LittleEndian.PutUint16(header[20:], 1)
	binary.LittleEndian.PutUint16(header[22:], 1)
	binary.LittleEndian.PutUint32(header[24:], 16000)
	binary.LittleEndian.PutUint32(header[28:], 32000)
	binary.LittleEndian.PutUint16(header[32:], 2)
	binary.LittleEndian.PutUint16(header[34:], 16)
	copy(header[36:], "data")
	binary.LittleEndian.PutUint32(header[40:], uint32(end))
	_, err = out.WriteAt(header, 0)
	return err
}
func speakerPCM(f *os.File) (int64, int64, error) {
	var head [12]byte
	if _, err := io.ReadFull(f, head[:]); err != nil {
		return 0, 0, err
	}
	if string(head[:4]) != "RIFF" || string(head[8:]) != "WAVE" {
		return 0, 0, fmt.Errorf("화자 구분에는 PCM WAV가 필요합니다")
	}
	valid := false
	for i := 0; i < 128; i++ {
		var h [8]byte
		if _, err := io.ReadFull(f, h[:]); err != nil {
			return 0, 0, err
		}
		size := int64(binary.LittleEndian.Uint32(h[4:]))
		start, _ := f.Seek(0, io.SeekCurrent)
		if string(h[:4]) == "fmt " {
			var fmtData [16]byte
			if size < 16 {
				return 0, 0, fmt.Errorf("invalid WAV fmt")
			}
			if _, err := io.ReadFull(f, fmtData[:]); err != nil {
				return 0, 0, err
			}
			valid = binary.LittleEndian.Uint16(fmtData[:]) == 1 && binary.LittleEndian.Uint16(fmtData[2:]) == 1 && binary.LittleEndian.Uint32(fmtData[4:]) == 16000 && binary.LittleEndian.Uint16(fmtData[14:]) == 16
		}
		if string(h[:4]) == "data" {
			if !valid || size%2 != 0 {
				return 0, 0, fmt.Errorf("화자 구분에는 16kHz mono PCM16 WAV가 필요합니다")
			}
			info, err := f.Stat()
			if err != nil {
				return 0, 0, err
			}
			if start+size > info.Size() {
				return 0, 0, fmt.Errorf("truncated WAV")
			}
			return start, size, nil
		}
		if _, err := f.Seek(start+size+(size%2), io.SeekStart); err != nil {
			return 0, 0, err
		}
	}
	return 0, 0, fmt.Errorf("WAV data missing")
}
func subtitleSpeakerIDs(start, end float64, segments []subtitleSpeakerSegment) []int {
	found := map[int]bool{}
	for _, s := range segments {
		if s.Start >= end {
			break
		}
		if s.End > start {
			found[s.Speaker] = true
		}
	}
	ids := make([]int, 0, len(found))
	for id := range found {
		ids = append(ids, id)
	}
	sort.Ints(ids)
	return ids
}
func subtitleSpeakerLabel(cue subtitleCue) string {
	if !cue.Diarized {
		return ""
	}
	if len(cue.Speakers) == 0 {
		return "화자 미확인"
	}
	labels := make([]string, 0, len(cue.Speakers))
	for _, id := range cue.Speakers {
		label := cue.SpeakerNames[strconv.Itoa(id)]
		if label == "" {
			label = fmt.Sprintf("화자 %d", id)
		}
		labels = append(labels, label)
	}
	label := strings.Join(labels, "·")
	if len(labels) > 1 {
		label += " (겹침·구분 불확실)"
	}
	return label
}
func validateSpeakerNames(names map[string]string, cues []subtitleCue) error {
	allowed := map[string]bool{}
	for _, c := range cues {
		for _, id := range c.Speakers {
			allowed[strconv.Itoa(id)] = true
		}
	}
	for id, name := range names {
		if !allowed[id] || utf8.RuneCountInString(name) > 40 || strings.ContainsAny(name, "<>&[]") {
			return fmt.Errorf("잘못된 화자 이름 (최대 40자)")
		}
		for _, r := range name {
			if unicode.IsControl(r) {
				return fmt.Errorf("화자 이름에는 줄바꿈이나 제어 문자를 사용할 수 없습니다")
			}
		}
	}
	return nil
}
