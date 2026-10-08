package server

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sparktalk/internal/asr"
	"sparktalk/internal/config"
	"sparktalk/internal/media"
	"sparktalk/internal/orchestrator"
	"strings"
	"testing"
	"time"
)

// Explicit integration of video extraction, GPU ASR and speaker separation.
func TestLiveRadixArkVideoASR(t *testing.T) {
	if os.Getenv("TALK_LIVE_RADIXARK_ASR") != "1" {
		t.Skip("explicit NVFP4/video transcription integration")
	}
	report := os.Getenv("TALK_LIVE_RADIXARK_ASR_REPORT")
	input := os.Getenv("TALK_LIVE_RADIXARK_ASR_VIDEO")
	if report == "" || input == "" {
		t.Fatal("video and report required")
	}
	cfg, _, err := config.Load("../../dist/sparktalk.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Runtime.ActiveBundle != "flash-next-radixark" || !cfg.ASR.Enabled || cfg.TTS.Enabled || cfg.Image.Enabled || !cfg.ASR.Diarization {
		t.Fatal("NVFP4 + on-demand ASR profile required")
	}
	s, _ := testImageServer(t)
	s.cfg = cfg
	s.asr = asr.New(cfg.ASR)
	s.runtime, err = orchestrator.NewControllerWithCatalog(*cfg.Runtime.Catalog)
	if err != nil {
		t.Fatal(err)
	}
	s.runtime.ConfigurePaths(cfg.Runtime.DataDir, cfg.Runtime.ModelCache)
	t.Cleanup(s.runtime.Close)
	identity := func() string {
		b, e := exec.Command("docker", "inspect", "-f", "{{.Id}} {{.State.Pid}}", "sglang-qwen38-fn-radixark").Output()
		if e != nil {
			t.Fatal(e)
		}
		return strings.TrimSpace(string(b))
	}
	before := identity()
	f, err := os.Open(input)
	if err != nil {
		t.Fatal(err)
	}
	item, err := s.media.SaveReader(f, filepath.Base(input), "video/mp4", media.MaxRemoteVideoBytes)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Minute)
	defer cancel()
	started := time.Now()
	result, err := s.transcribeAttachment(ctx, item, cfg.ASR)
	if err != nil {
		t.Fatal(err)
	}
	if result.Text == "" || result.DiarizationStatus != "completed" {
		t.Fatalf("incomplete transcription: %+v", result)
	}
	elapsed := time.Since(started)
	cached, err := s.transcribeAttachment(ctx, item, cfg.ASR)
	if err != nil || cached.Text != result.Text {
		t.Fatal("transcript cache failed", err)
	}
	if identity() != before {
		t.Fatal("LLM restarted during video transcription")
	}
	resp, err := http.Get(strings.TrimRight(cfg.Model.Endpoint, "/") + "/server_info")
	if err != nil {
		t.Fatal(err)
	}
	var info struct {
		Context  int `json:"context_length"`
		Capacity int `json:"max_total_num_tokens"`
	}
	err = json.NewDecoder(resp.Body).Decode(&info)
	resp.Body.Close()
	if err != nil || info.Context != 1048576 || info.Capacity != 1048576 {
		t.Fatalf("KV changed: %+v %v", info, err)
	}
	if err := os.MkdirAll(report, 0700); err != nil {
		t.Fatal(err)
	}
	data, _ := json.MarshalIndent(map[string]any{"video": filepath.Base(input), "transcript": result, "elapsed_seconds": elapsed.Seconds(), "same_llm_id_and_pid": true, "kv_tokens": info.Capacity, "context_tokens": info.Context, "cached_transcript_reused": true}, "", "  ")
	if err := os.WriteFile(filepath.Join(report, "results.json"), data, 0600); err != nil {
		t.Fatal(err)
	}
	t.Log(string(data))
}
