package server

import (
	"bytes"
	"context"
	"encoding/json"
	"image"
	"image/jpeg"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"net/textproto"
	"os"
	"strings"
	"testing"
	"time"

	"sparktalk/internal/asr"
	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/knowledge"
	"sparktalk/internal/llm"
	"sparktalk/internal/media"
)

func TestAssistantImageHistoryUsesLabelledVisualContext(t *testing.T) {
	s, photo := testImageServer(t)
	const question = "fish에는 영상생성에 사용가능한 TTS 음성이 몇명 그리고 뭐뭐 있냐?"
	items := []db.Message{{Role: "user", Content: "원본", Attachments: []db.Attachment{photo}}, {Role: "assistant", Content: "미리보기입니다", Attachments: []db.Attachment{photo, photo}}, {Role: "user", Content: question}}
	messages, err := s.llmMessages(context.Background(), items, config.Config{})
	if err != nil {
		t.Fatal(err)
	}
	if len(messages) != 4 || messages[1].Role != "assistant" || messages[1].Content != "미리보기입니다" || messages[3].Content != question {
		t.Fatalf("history changed: %+v", messages)
	}
	images := 0
	for _, m := range messages {
		parts, _ := m.Content.([]map[string]any)
		for _, p := range parts {
			if p["type"] == "image_url" {
				images++
				if m.Role != "user" && m.Role != "tool" {
					t.Fatalf("invalid image role %s", m.Role)
				}
			}
		}
	}
	if images != 3 {
		t.Fatalf("lost images: %d", images)
	}
	raw, _ := json.Marshal(messages[2].Content)
	if !strings.Contains(string(raw), "Historical assistant image attachments") || !strings.Contains(string(raw), `origin=\"assistant\"`) || !strings.Contains(string(raw), photo.ID) {
		t.Fatalf("lost provenance: %s", raw)
	}
}

// Read-only reproduction using the original conversation and installed model.
func TestLiveAssistantImageHistoryFishQuestion(t *testing.T) {
	if os.Getenv("TALK_LIVE_ASSISTANT_IMAGE_HISTORY") != "1" {
		t.Skip("opt-in local model regression")
	}
	cfg, _, err := config.Load("../../dist/sparktalk.yaml")
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.Get("http://127.0.0.1:8585/api/sessions/chat-1792689381708313299/messages")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatal(resp.Status)
	}
	var items []db.Message
	if err = json.NewDecoder(resp.Body).Decode(&items); err != nil {
		t.Fatal(err)
	}
	var history []db.Message
	for _, item := range items {
		if item.ID <= 3091 {
			history = append(history, item)
		}
	}
	store, err := media.New("../../dist/sparktalk.db")
	if err != nil {
		t.Fatal(err)
	}
	s := &Server{media: store, knowledgeIndex: &knowledge.Extractor{}}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	messages, err := s.llmMessages(ctx, modelHistory(history, 3091), cfg)
	if err != nil {
		t.Fatal(err)
	}
	client := llm.New(cfg.Model.Endpoint, cfg.Model.DefaultModel, cfg.Model.APIKey, cfg.Model.ModelType)
	result, err := client.Stream(llm.WithOutputLimit(ctx, 2048), messages, cfg.Model.DefaultModel, "none", nil, func(string, string) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(result.Content) == "" {
		t.Fatal("empty response")
	}
	t.Logf("Original history accepted; response: %s", result.Content)
}

func TestLLMMessagesExtractsDocumentAttachment(t *testing.T) {
	store, err := media.New(t.TempDir() + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	document, err := store.SaveAttachment(testFileHeader(t, "notes.md", "text/markdown", []byte("# 회의록\n\n저항은 10킬로옴입니다.")))
	if err != nil {
		t.Fatal(err)
	}
	server := &Server{media: store, knowledgeIndex: &knowledge.Extractor{}}
	messages, err := server.llmMessages(context.Background(), []db.Message{{Role: "user", Content: "요약해줘", Attachments: []db.Attachment{document}}}, config.Config{})
	if err != nil {
		t.Fatal(err)
	}
	data, _ := json.Marshal(messages[0].Content)
	if !strings.Contains(string(data), "document_attachment") || !strings.Contains(string(data), "10킬로옴") || strings.Contains(string(data), "data:text") {
		t.Fatalf("unexpected document prompt: %s", data)
	}
}

func TestGeneratedVideoHistoryIsAReferenceNotAssistantVideoInput(t *testing.T) {
	store, err := media.New(t.TempDir() + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	video, err := store.SaveAttachment(testFileHeader(t, "short.mp4", "video/mp4", append([]byte{0, 0, 0, 12}, []byte("ftypisomvideo")...)))
	if err != nil {
		t.Fatal(err)
	}
	s := &Server{media: store}
	for _, withUserVideo := range []bool{false, true} {
		items := []db.Message{}
		if withUserVideo {
			items = append(items, db.Message{Role: "user", Content: "내 영상", Attachments: []db.Attachment{video}})
		}
		items = append(items, db.Message{Role: "assistant", Content: "완성 영상", Attachments: []db.Attachment{video}}, db.Message{Role: "user", Content: "60대 마음가짐으로 새 쇼츠 만들어라"})
		messages, err := s.llmMessages(context.Background(), items, config.Config{ASR: config.ASRConfig{Enabled: !withUserVideo, Endpoint: "http://127.0.0.1:1", FFmpegEndpoint: "http://127.0.0.1:1"}})
		if err != nil {
			t.Fatal("generated video must not trigger media decoding/transcription", err)
		}
		videos := 0
		for _, m := range messages {
			raw, _ := json.Marshal(m.Content)
			if strings.Contains(string(raw), `"type":"video_url"`) {
				videos++
				if m.Role != "user" {
					t.Fatal("assistant video_url")
				}
			}
			if m.Role == "assistant" && (!strings.Contains(string(raw), video.ID) || !strings.Contains(string(raw), "media_reference")) {
				t.Fatal("generated attachment reference lost", string(raw))
			}
		}
		want := 0
		if withUserVideo {
			want = 1
		}
		if videos != want {
			t.Fatal("generated video displaced user video", videos, want)
		}
	}
}

func TestLLMMessagesTranscribesAudioAndKeepsVideoVisuals(t *testing.T) {
	store, err := media.New(t.TempDir() + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	audio, err := store.SaveAttachment(testFileHeader(t, "voice.mp3", "audio/mpeg", []byte("ID3sample audio")))
	if err != nil {
		t.Fatal(err)
	}
	video, err := store.SaveAttachment(testFileHeader(t, "clip.mp4", "video/mp4", append([]byte{0, 0, 0, 12}, []byte("ftypisomvideo")...)))
	if err != nil {
		t.Fatal(err)
	}
	cfg := config.Config{ASR: config.ASRConfig{Enabled: true, Endpoint: "http://asr", FFmpegEndpoint: "http://ffmpeg", Model: "nemotron", MediaLanguage: "auto"}}
	fingerprint := transcriptFingerprint(cfg.ASR)
	for _, item := range []db.Attachment{audio, video} {
		if err := store.SaveTranscript(item.ID, media.TranscriptCache{Fingerprint: fingerprint, Text: item.Name + " 전사", Language: "Korean"}); err != nil {
			t.Fatal(err)
		}
	}
	s := &Server{media: store}
	messages, err := s.llmMessages(context.Background(), []db.Message{{Role: "user", Content: "분석해줘", Attachments: []db.Attachment{audio, video}}}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	payload, _ := json.Marshal(messages[0].Content)
	text := string(payload)
	for _, expected := range []string{"video_url", "data:video/mp4;base64,", "voice.mp3 전사", "clip.mp4 전사"} {
		if !strings.Contains(text, expected) {
			t.Fatalf("multimodal payload does not contain %q: %s", expected, text)
		}
	}
	if strings.Contains(text, "audio_url") || strings.Contains(text, "data:audio/mpeg;base64,") {
		t.Fatalf("raw audio must not be sent to the visual language model: %s", text)
	}
}

func TestLLMMessagesSendsOnlyLatestVideoAndKeepsCurrentInstruction(t *testing.T) {
	store, err := media.New(t.TempDir() + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	oldVideo, err := store.SaveAttachment(testFileHeader(t, "old.mp4", "video/mp4", append([]byte{0, 0, 0, 12}, []byte("ftypisom-old")...)))
	if err != nil {
		t.Fatal(err)
	}
	newVideo, err := store.SaveAttachment(testFileHeader(t, "new.mp4", "video/mp4", append([]byte{0, 0, 0, 12}, []byte("ftypisom-new")...)))
	if err != nil {
		t.Fatal(err)
	}
	cfg := config.Config{ASR: config.ASRConfig{Enabled: false, Model: "nemotron"}}
	s := &Server{media: store}
	messages, err := s.llmMessages(context.Background(), []db.Message{
		{Role: "user", Content: "이전 영상", Attachments: []db.Attachment{oldVideo}},
		{Role: "assistant", Content: "이전 답변"},
		{Role: "user", Content: "https://example.com/new 영상 분석해라", Attachments: []db.Attachment{newVideo}},
	}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	payload, _ := json.Marshal(messages)
	text := string(payload)
	if strings.Count(text, `"type":"video_url"`) != 1 {
		t.Fatalf("expected exactly one raw video input: %s", text)
	}
	for _, expected := range []string{"old.mp4", "Historical video", "https://example.com/new 영상 분석해라"} {
		if !strings.Contains(text, expected) {
			t.Fatalf("model history does not contain %q: %s", expected, text)
		}
	}
}

func TestLLMMessagesKeepsVideoWhenASRIsOffline(t *testing.T) {
	store, err := media.New(t.TempDir() + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	video, err := store.SaveAttachment(testFileHeader(t, "clip.mp4", "video/mp4", append([]byte{0, 0, 0, 12}, []byte("ftypisomvideo")...)))
	if err != nil {
		t.Fatal(err)
	}
	ffmpeg := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "audio/wav")
		_, _ = w.Write([]byte("extracted audio"))
	}))
	defer ffmpeg.Close()
	cfg := config.Config{ASR: config.ASRConfig{
		Enabled: true, Endpoint: "http://127.0.0.1:1", FFmpegEndpoint: ffmpeg.URL,
		Model: "nemotron", MediaLanguage: "auto", Timeout: "1s",
	}}
	s := &Server{media: store, asr: asr.New(cfg.ASR)}
	messages, err := s.llmMessages(context.Background(), []db.Message{{
		Role: "user", Content: "영상 화면을 분석해줘", Attachments: []db.Attachment{video},
	}}, cfg)
	if err != nil {
		t.Fatalf("video visuals must survive an ASR outage: %v", err)
	}
	payload, _ := json.Marshal(messages[0].Content)
	text := string(payload)
	for _, expected := range []string{"video_url", "data:video/mp4;base64,", `status=\"unavailable\"`, "Inspect the video frames directly"} {
		if !strings.Contains(text, expected) {
			t.Fatalf("video fallback does not contain %q: %s", expected, text)
		}
	}
}

func TestLLMMessagesStillRequiresASRForAudio(t *testing.T) {
	store, err := media.New(t.TempDir() + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	audio, err := store.SaveAttachment(testFileHeader(t, "voice.mp3", "audio/mpeg", []byte("ID3sample audio")))
	if err != nil {
		t.Fatal(err)
	}
	ffmpeg := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "audio/wav")
		_, _ = w.Write([]byte("extracted audio"))
	}))
	defer ffmpeg.Close()
	cfg := config.Config{ASR: config.ASRConfig{
		Enabled: true, Endpoint: "http://127.0.0.1:1", FFmpegEndpoint: ffmpeg.URL,
		Model: "nemotron", MediaLanguage: "auto", Timeout: "1s",
	}}
	s := &Server{media: store, asr: asr.New(cfg.ASR)}
	_, err = s.llmMessages(context.Background(), []db.Message{{
		Role: "user", Content: "음성을 분석해줘", Attachments: []db.Attachment{audio},
	}}, cfg)
	if err == nil || !strings.Contains(err.Error(), "ASR API") {
		t.Fatalf("audio-only input must still report the ASR outage, got %v", err)
	}
}

func TestMediaFileHandlerSupportsRangeRequests(t *testing.T) {
	store, err := media.New(t.TempDir() + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	item, err := store.SaveAttachment(testFileHeader(t, "clip.mp4", "video/mp4", append([]byte{0, 0, 0, 12}, []byte("ftypisomvideo")...)))
	if err != nil {
		t.Fatal(err)
	}
	s := &Server{media: store}
	req := httptest.NewRequest(http.MethodGet, item.URL, nil)
	req.Header.Set("Range", "bytes=0-3")
	recorder := httptest.NewRecorder()
	s.file(recorder, req)
	if recorder.Code != http.StatusPartialContent || recorder.Header().Get("Content-Type") != "video/mp4" || recorder.Body.Len() != 4 {
		t.Fatalf("unexpected range response: status=%d type=%q bytes=%d", recorder.Code, recorder.Header().Get("Content-Type"), recorder.Body.Len())
	}
}

func TestUploadSourceStoresMediaResponse(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/source/download" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "video/mp4")
		w.Header().Set("Content-Disposition", `attachment; filename="source.mp4"`)
		w.Header().Set("X-Media-Title", "Remote+clip")
		_, _ = w.Write(append([]byte{0, 0, 0, 12}, []byte("ftypisomvideo")...))
	}))
	defer upstream.Close()
	store, err := media.New(t.TempDir() + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	s := &Server{media: store, cfg: config.Config{ASR: config.ASRConfig{FFmpegEndpoint: upstream.URL, Timeout: "5s"}}}
	req := httptest.NewRequest(http.MethodPost, "/api/media/source", strings.NewReader(`{"url":"https://example.com/video"}`))
	recorder := httptest.NewRecorder()
	s.uploadSource(recorder, req)
	if recorder.Code != http.StatusCreated {
		t.Fatalf("unexpected status %d: %s", recorder.Code, recorder.Body.String())
	}
	var item db.Attachment
	if err := json.Unmarshal(recorder.Body.Bytes(), &item); err != nil {
		t.Fatal(err)
	}
	if item.Name != "Remote clip.mp4" || item.MIME != "video/mp4" {
		t.Fatalf("unexpected attachment: %+v", item)
	}
}

func testFileHeader(t *testing.T, name, mimeType string, data []byte) *multipart.FileHeader {
	t.Helper()
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	header := make(textproto.MIMEHeader)
	header.Set("Content-Disposition", `form-data; name="file"; filename="`+name+`"`)
	header.Set("Content-Type", mimeType)
	part, err := writer.CreatePart(header)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := part.Write(data); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	req := httptest.NewRequest(http.MethodPost, "/api/files", &body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	if err := req.ParseMultipartForm(1 << 20); err != nil {
		t.Fatal(err)
	}
	file, fileHeader, err := req.FormFile("file")
	if err != nil {
		t.Fatal(err)
	}
	_ = file.Close()
	return fileHeader
}

func TestDeepSeekVideoUsesFrameImageAndDisclosesMissingAudio(t *testing.T) {
	store, err := media.New(t.TempDir() + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	video, err := store.SaveAttachment(testFileHeader(t, "clip.mp4", "video/mp4", append([]byte{0, 0, 0, 12}, []byte("ftypisomvideo")...)))
	if err != nil {
		t.Fatal(err)
	}
	frames := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/video/frames" || r.Method != "POST" {
			t.Errorf("wrong request %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("X-Video-Duration", "8.000")
		_ = jpeg.Encode(w, image.NewRGBA(image.Rect(0, 0, 16, 8)), nil)
	}))
	defer frames.Close()
	cfg := config.Config{Model: config.ModelConfig{ModelType: "deepseek-v4"}, ASR: config.ASRConfig{FFmpegEndpoint: frames.URL}}
	s := &Server{media: store}
	messages, err := s.llmMessages(context.Background(), []db.Message{{Role: "user", Content: "영상 요약", Attachments: []db.Attachment{video}}}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	data, _ := json.Marshal(messages)
	text := string(data)
	if strings.Contains(text, "video_url") || !strings.Contains(text, "data:image/jpeg;base64,") || !strings.Contains(text, "Audio transcription is disabled") {
		t.Fatalf("incorrect DeepSeek payload: %s", text)
	}
}

func TestOrnithImageOnlyDeploymentUsesFrames(t *testing.T) {
	store, err := media.New(t.TempDir() + "/chat.db")
	if err != nil {
		t.Fatal(err)
	}
	video, err := store.SaveAttachment(testFileHeader(t, "clip.mp4", "video/mp4", append([]byte{0, 0, 0, 12}, []byte("ftypisomvideo")...)))
	if err != nil {
		t.Fatal(err)
	}
	frames := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/video/frames" || r.Method != "POST" {
			t.Errorf("wrong request %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("X-Video-Duration", "8.000")
		_ = jpeg.Encode(w, image.NewRGBA(image.Rect(0, 0, 16, 8)), nil)
	}))
	defer frames.Close()
	cfg := config.Config{Model: config.ModelConfig{ModelType: "qwen3.5", VideoInputs: map[string]string{"\n": "frames"}}, ASR: config.ASRConfig{FFmpegEndpoint: frames.URL}}
	s := &Server{media: store}
	messages, err := s.llmMessages(context.Background(), []db.Message{{Role: "user", Content: "영상 요약", Attachments: []db.Attachment{video}}}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	data, _ := json.Marshal(messages)
	text := string(data)
	if strings.Contains(text, "video_url") || !strings.Contains(text, "data:image/jpeg;base64,") || !strings.Contains(text, "Audio transcription is disabled") {
		t.Fatalf("incorrect image-only Ornith payload: %s", text)
	}
}

func TestLargeNativeVideoUsesFrames(t *testing.T) {
	store, _ := media.New(t.TempDir() + "/chat.db")
	f, err := os.CreateTemp(t.TempDir(), "video")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	f.Write(append([]byte{0, 0, 0, 24}, []byte("ftypisom0000")...))
	f.Truncate(65 << 20)
	f.Seek(0, 0)
	video, err := store.SaveReader(f, "long.mp4", "video/mp4", media.MaxRemoteVideoBytes)
	if err != nil {
		t.Fatal(err)
	}
	called := false
	frames := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.Header().Set("X-Video-Duration", "4795")
		jpeg.Encode(w, image.NewRGBA(image.Rect(0, 0, 16, 8)), nil)
	}))
	defer frames.Close()
	cfg := config.Config{Model: config.ModelConfig{ModelType: "qwen3.5"}, ASR: config.ASRConfig{FFmpegEndpoint: frames.URL}}
	s := &Server{media: store}
	messages, err := s.llmMessages(context.Background(), []db.Message{{Role: "user", Content: "요약", Attachments: []db.Attachment{video}}}, cfg)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(messages)
	if !called || strings.Contains(string(b), "video_url") || !strings.Contains(string(b), "Audio transcription is disabled") {
		t.Fatalf("unexpected large-video payload %s", b)
	}
}
