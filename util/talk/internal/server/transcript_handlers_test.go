package server

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sparktalk/internal/asr"
	"sparktalk/internal/config"
	"sparktalk/internal/media"
	"strings"
	"testing"
)

func TestSpeakerTranscriptCacheAndDisplay(t *testing.T) {
	store, _ := media.New(t.TempDir() + "/chat.db")
	item, err := store.SaveReader(strings.NewReader("ID3audio"), "voice.mp3", "audio/mpeg", media.MaxAttachmentBytes)
	if err != nil {
		t.Fatal(err)
	}
	cfg := config.ASRConfig{Diarization: true}
	cached := media.TranscriptCache{Text: "hello", Fingerprint: transcriptFingerprint(cfg), DiarizationStatus: "completed", Turns: []asr.Turn{{Start: 0, End: 1, Speakers: []int{1}, Text: "hello"}}}
	if err = store.SaveTranscript(item.ID, cached); err != nil {
		t.Fatal(err)
	}
	s := &Server{media: store}
	actual, err := s.transcribeAttachment(context.Background(), item, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(transcriptBlock(item, actual), "화자 1: hello") {
		t.Fatal("speaker not passed to model")
	}
	rec := httptest.NewRecorder()
	s.attachmentTranscript(rec, httptest.NewRequest("GET", "/api/media/transcript/"+item.ID, nil))
	if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"speakers":[1]`) {
		t.Fatalf("view %d %s", rec.Code, rec.Body.String())
	}
	cfg.Diarization = false
	if _, ok, _ := store.LoadTranscript(item.ID, transcriptFingerprint(cfg)); ok {
		t.Fatal("diarization toggle reused stale cache")
	}
}

func TestNoRecognizedSpeechIsCachedWithoutRetryingOrHidingServerErrors(t *testing.T) {
	for _, tc := range []struct {
		name, body string
		status     int
		empty      bool
	}{
		{"no recognized words", `{"text":" ","language":"ko"}`, 200, true},
		{"missing text is malformed", `{}`, 200, false},
		{"backend failure stays an error", `{"error":"failed"}`, 500, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path == "/v1/audio/extract" {
					w.Header().Set("Content-Type", "audio/wav")
					io.WriteString(w, "audio")
					return
				}
				if err := r.ParseMultipartForm(1 << 20); err != nil {
					t.Error(err)
				}
				defer r.MultipartForm.RemoveAll()
				calls++
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tc.status)
				io.WriteString(w, tc.body)
			}))
			defer backend.Close()
			store, _ := media.New(t.TempDir() + "/chat.db")
			item, err := store.SaveReader(bytes.NewReader(importTestVideo), "MiniMax-H3.mp4", "video/mp4", media.MaxAttachmentBytes)
			if err != nil {
				t.Fatal(err)
			}
			cfg := config.ASRConfig{Enabled: true, Endpoint: backend.URL, FFmpegEndpoint: backend.URL, Timeout: "5s"}
			s := &Server{media: store, asr: asr.New(cfg)}
			for i := 0; i < 2; i++ {
				result, err := s.transcribeAttachment(context.Background(), item, cfg)
				if tc.empty {
					if err != nil || result.Status != "no_text" || result.Text != "" || result.Warning == "" {
						t.Fatalf("empty result: %+v %v", result, err)
					}
				} else if err == nil {
					t.Fatal("real failure suppressed")
				}
			}
			if tc.empty {
				if calls != 1 {
					t.Fatal("empty transcript was retried", calls)
				}
				rec := httptest.NewRecorder()
				s.attachmentTranscript(rec, httptest.NewRequest("GET", "/api/media/transcript/"+item.ID, nil))
				if rec.Code != 200 || !strings.Contains(rec.Body.String(), `"status":"no_text"`) {
					t.Fatal(rec.Code, rec.Body.String())
				}
			} else if calls != 2 {
				t.Fatal("failure incorrectly cached", calls)
			}
		})
	}
}
