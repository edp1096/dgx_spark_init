package server

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"mediaapp/internal/config"
	"mediaapp/internal/jobs"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func speakerTestWAV(t *testing.T, path string) {
	t.Helper()
	h := make([]byte, 44+32000)
	copy(h, "RIFF")
	binary.LittleEndian.PutUint32(h[4:], uint32(len(h)-8))
	copy(h[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(h[16:], 16)
	binary.LittleEndian.PutUint16(h[20:], 1)
	binary.LittleEndian.PutUint16(h[22:], 1)
	binary.LittleEndian.PutUint32(h[24:], 16000)
	binary.LittleEndian.PutUint32(h[28:], 32000)
	binary.LittleEndian.PutUint16(h[32:], 2)
	binary.LittleEndian.PutUint16(h[34:], 16)
	copy(h[36:], "data")
	binary.LittleEndian.PutUint32(h[40:], 32000)
	if err := os.WriteFile(path, h, 0600); err != nil {
		t.Fatal(err)
	}
}
func TestSpeakerAudioJoinAndCueBoundaries(t *testing.T) {
	dir := t.TempDir()
	speakerTestWAV(t, filepath.Join(dir, "a.wav"))
	speakerTestWAV(t, filepath.Join(dir, "b.wav"))
	out := filepath.Join(dir, "joined.wav")
	if err := joinSpeakerAudio(out, dir, []preparedSegment{{Name: "a.wav", Start: 0}, {Name: "b.wav", Start: 2}}); err != nil {
		t.Fatal(err)
	}
	f, _ := os.Open(out)
	defer f.Close()
	_, size, err := speakerPCM(f)
	if err != nil || size != 96000 {
		t.Fatalf("join size=%d err=%v", size, err)
	}
	words := []timedWord{{Text: "안녕", Start: 0, End: 1, Diarized: true, Speakers: []int{1}}, {Text: "반가워", Start: 1, End: 2, Diarized: true, Speakers: []int{2}}}
	cues := cuesFromTimestamps("안녕 반가워", words, 0)
	if len(cues) != 2 || !strings.Contains(renderSRT(cues, "none"), "[화자 2] 반가워") {
		t.Fatalf("wrong cues %+v", cues)
	}
}
func TestSpeakerNamesRegenerateWithoutInference(t *testing.T) {
	dir := t.TempDir()
	store, _ := jobs.New(dir)
	s := New(config.Config{DataDir: dir}, store, nil)
	job := jobs.Job{ID: "diar-rename", Kind: "recognition", Status: "completed", CreatedAt: time.Now(), Params: map[string]any{"speaker_ids": []int{1, 2}}, Outputs: map[string]string{}}
	store.Save(job)
	cues := []subtitleCue{{Start: 0, End: 1, Text: "안녕", Diarized: true, Speakers: []int{1}}, {Start: 1, End: 2, Text: "Hello", Diarized: true, Speakers: []int{2}}}
	s.writeSubtitleCueArchive(job.ID, cues)
	for _, name := range []string{"김철수", "박영희", ""} {
		body, _ := json.Marshal(subtitleRegenerateRequest{TranslationMode: "none", OutputFormats: []string{"srt", "vtt"}, SpeakerNames: map[string]string{"1": name}})
		req := httptest.NewRequest(http.MethodPost, "/api/jobs/"+job.ID+"/subtitle-regenerate", bytes.NewReader(body))
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, req)
		if rec.Code != 200 {
			t.Fatalf("rename %d %s", rec.Code, rec.Body.String())
		}
		data, _ := os.ReadFile(store.OutputPath(job.ID + ".srt"))
		want := name
		if want == "" {
			want = "화자 1"
		}
		if !strings.Contains(string(data), "["+want+"] 안녕") || !strings.Contains(string(data), "[화자 2] Hello") {
			t.Fatalf("bad output %s", data)
		}
		actual, _, _ := s.loadSubtitleCueArchive(job)
		if actual[0].Speakers[0] != 1 || actual[0].Text != "안녕" {
			t.Fatal("identity or original text changed")
		}
	}
	for _, names := range []map[string]string{{"3": "외부"}, {"1": "잘못\n된 이름"}, {"1": "<b>이름</b>"}} {
		if validateSpeakerNames(names, cues) == nil {
			t.Fatal("invalid names accepted")
		}
	}
}

func TestSubtitleDiarizationFullPipeline(t *testing.T) {
	for _, failure := range []bool{false, true} {
		t.Run(map[bool]string{false: "speakers", true: "preserve_plain_subtitles"}[failure], func(t *testing.T) {
			calls := 0
			backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method == http.MethodGet {
					w.Write([]byte(`{"status":"ok"}`))
					return
				}
				if err := r.ParseMultipartForm(1 << 20); err != nil {
					t.Error(err)
					return
				}
				defer r.MultipartForm.RemoveAll()
				if r.URL.Path == "/v1/audio/diarizations" {
					calls++
					if failure {
						http.Error(w, "missing model", 503)
						return
					}
					json.NewEncoder(w).Encode(map[string]any{"segments": []subtitleSpeakerSegment{{0, 1, 4}, {1, 2, 7}}})
					return
				}
				json.NewEncoder(w).Encode(map[string]any{"text": "안녕하세요", "language": "Korean", "timestamps": []timedWord{{Text: "안녕하세요", Start: 0, End: .8}}})
			}))
			defer backend.Close()
			dir := t.TempDir()
			input := filepath.Join(dir, "inputs", "job")
			prepared := filepath.Join(input, "prepared")
			os.MkdirAll(prepared, 0700)
			speakerTestWAV(t, filepath.Join(prepared, "a.wav"))
			speakerTestWAV(t, filepath.Join(prepared, "b.wav"))
			manifest := preparedManifest{Segments: []preparedSegment{{Name: "a.wav", Start: 0, End: 1, Duration: 1}, {Name: "b.wav", Start: 1, End: 2, Duration: 1}}}
			data, _ := json.Marshal(manifest)
			os.WriteFile(filepath.Join(prepared, "manifest.json"), data, 0600)
			store, _ := jobs.New(dir)
			s := New(config.Config{DataDir: dir, Engines: map[string]config.Engine{"recognition": {Endpoint: backend.URL}}, Recognition: config.Recognition{Model: "test", DiarizationEndpoint: backend.URL}}, store, nil)
			job := jobs.Job{ID: "job", Kind: "recognition", Status: "running", CreatedAt: time.Now(), Params: map[string]any{"diarization": true}}
			store.Save(job)
			s.runSubtitle(job, input, "", "", "Korean", "", []string{"srt"}, "none", "", "", "")
			finished, _ := store.Get("job")
			if finished.Status != "completed" || calls != 1 {
				t.Fatalf("pipeline %+v calls=%d", finished, calls)
			}
			output, _ := os.ReadFile(store.OutputPath("job.srt"))
			if failure {
				if finished.Params["diarization_warning"] == nil || strings.Contains(string(output), "화자") {
					t.Fatalf("lost warning/plain text %s %+v", output, finished.Params)
				}
			} else {
				if !strings.Contains(string(output), "[화자 1]") || !strings.Contains(string(output), "[화자 2]") {
					t.Fatalf("unstable speakers %s", output)
				}
			}
		})
	}
}

func TestLiveSubtitleDiarization(t *testing.T) {
	path := os.Getenv("MEDIA_DIAR_LIVE_AUDIO")
	if path == "" {
		t.Skip("explicit live model integration")
	}
	dir := t.TempDir()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	os.WriteFile(filepath.Join(dir, "meeting.wav"), data, 0600)
	store, _ := jobs.New(dir)
	store.Save(jobs.Job{ID: "live-diar", Kind: "recognition", Status: "running", CreatedAt: time.Now()})
	s := New(config.Config{DataDir: dir, Recognition: config.Recognition{DiarizationEndpoint: "http://127.0.0.1:8693"}}, store, nil)
	segments, err := s.subtitleSpeakers("live-diar", dir, preparedManifest{Segments: []preparedSegment{{Name: "meeting.wav", Start: 0}}})
	if err != nil {
		t.Fatal(err)
	}
	ids := map[int]bool{}
	for _, x := range segments {
		ids[x.Speaker] = true
	}
	if len(ids) < 2 {
		t.Fatalf("expected meeting speakers, got %d", len(ids))
	}
	t.Logf("segments=%d speakers=%d", len(segments), len(ids))
}
