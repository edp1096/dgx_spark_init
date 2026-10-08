package asr

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sparktalk/internal/config"
	"strings"
	"testing"
)

func TestSpeakerAlignmentPreservesOverlapUnknownAndArrivalOrder(t *testing.T) {
	turns := alignSpeakers([]Word{{"안녕", 0, 1}, {"Hello", 1, 2}, {"네", 2, 3}, {"끝", 4, 5}}, []SpeakerSegment{{0, 1.5, 7}, {1, 3, 2}})
	if len(turns) != 4 || SpeakerLabel(turns[0].Speakers) != "화자 1" || SpeakerLabel(turns[2].Speakers) != "화자 2" || len(turns[1].Speakers) != 2 || len(turns[3].Speakers) != 0 {
		t.Fatalf("wrong turns %+v", turns)
	}
}
func TestMediaDiarizationAndVoiceIsolation(t *testing.T) {
	for _, failure := range []bool{false, true} {
		t.Run(map[bool]string{false: "success", true: "fallback"}[failure], func(t *testing.T) {
			diarCalls := 0
			backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/v1/audio/extract" {
					io.Copy(io.Discard, r.Body)
					io.WriteString(w, "wav")
					return
				}
				if err := r.ParseMultipartForm(1 << 20); err != nil {
					t.Error(err)
					return
				}
				defer r.MultipartForm.RemoveAll()
				f, _, err := r.FormFile("file")
				if err != nil {
					t.Error(err)
					return
				}
				data, _ := io.ReadAll(f)
				f.Close()
				if string(data) != "wav" {
					t.Errorf("wrong audio %q", data)
				}
				if r.URL.Path == "/v1/audio/diarizations" {
					diarCalls++
					if failure {
						http.Error(w, "not loaded", 503)
						return
					}
					io.WriteString(w, `{"segments":[{"start":0,"end":1,"speaker":3}]}`)
					return
				}
				json.NewEncoder(w).Encode(Result{Text: "안녕", Words: []Word{{"안녕", 0, 1}}})
			}))
			defer backend.Close()
			client := New(config.ASRConfig{Enabled: true, Diarization: true, Endpoint: backend.URL, FFmpegEndpoint: backend.URL, Timeout: "5s"})
			r, err := client.Transcribe(context.Background(), strings.NewReader("x"), "audio.wav", "audio/wav")
			if err != nil || r.Text != "안녕" || diarCalls != 1 {
				t.Fatalf("%+v %v calls=%d", r, err, diarCalls)
			}
			if failure && (r.Warning == "" || len(r.Turns) != 0) {
				t.Fatal("failure erased or mislabeled")
			}
			if !failure && (len(r.Turns) != 1 || r.Turns[0].Speakers[0] != 1) {
				t.Fatalf("no speaker %+v", r)
			}
			_, err = client.TranscribeVoice(context.Background(), strings.NewReader("x"), "voice.wav", "audio/wav")
			if err != nil || diarCalls != 1 {
				t.Fatal("microphone invoked diarization", err)
			}
		})
	}
}
