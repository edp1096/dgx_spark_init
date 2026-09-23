package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/media"
	"strings"
	"testing"
)

func TestLiveStoredVideoRetry(t *testing.T) {
	if os.Getenv("TALK_LIVE_VIDEO_RETRY") != "1" {
		t.Skip("explicit live model integration")
	}
	cfg, _, err := config.Load("../../dist/sparktalk.yaml")
	if err != nil {
		t.Fatal(err)
	}
	cfg.ASR.Enabled = false
	cfg.Context.Enabled = false
	cfg.Context.WindowTokens = 65536
	cfg.Context.OutputReserve = 512
	cfg.Model.SystemPrompt = "Describe only visible frames briefly in Korean. Do not claim to hear audio."
	cfg.Tools.Enabled = false
	s, _ := testImageServer(t)
	s.cfg = cfg
	s.llm = llm.New(cfg.Model.Endpoint, cfg.Model.DefaultModel, cfg.Model.APIKey, cfg.Model.ModelType)
	videoPath := os.Getenv("TALK_LIVE_VIDEO_FILE")
	if videoPath == "" {
		t.Fatal("TALK_LIVE_VIDEO_FILE must point to a local video fixture")
	}
	file, err := os.Open(videoPath)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	videoName := os.Getenv("TALK_LIVE_VIDEO_NAME")
	if videoName == "" {
		videoName = filepath.Base(videoPath)
	}
	a, err := s.media.SaveReader(file, videoName, "", media.MaxRemoteVideoBytes)
	if err != nil {
		t.Fatal(err)
	}
	u, err := s.db.AddPendingMessage("session", "첨부 영상의 대표 프레임에 보이는 내용을 두 문장으로 설명해.", []db.Attachment{a})
	if err != nil {
		t.Fatal(err)
	}
	if err = s.db.FailPendingTurn(u.ID, db.MessageFailed, "At most 0 video(s) may be provided", "", "old reasoning", nil); err != nil {
		t.Fatal(err)
	}
	history, _ := s.db.Messages("session")
	target := history[len(history)-1]
	body, _ := json.Marshal(map[string]any{"model": cfg.Model.DefaultModel, "reasoning_effort": "none", "tools_enabled": false})
	w := httptest.NewRecorder()
	s.messageAction(w, httptest.NewRequest("POST", fmt.Sprintf("/api/messages/%d/retry", target.ID), bytes.NewReader(body)))
	if strings.Contains(w.Body.String(), "event: error") || !strings.Contains(w.Body.String(), "event: done") {
		t.Fatalf("live retry failed: %s", w.Body.String())
	}
	history, _ = s.db.Messages("session")
	last := history[len(history)-1]
	t.Logf("Live retry completed: status=%s answer_chars=%d tools=%d", last.Status, len([]rune(last.Content)), len(last.ToolTrace))
	if len(last.Content) == 0 || len(last.ToolTrace) != 0 {
		t.Fatal("retry did not directly analyze media")
	}
}
