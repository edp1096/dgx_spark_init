package server

import (
	"context"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"sparktalk/internal/asr"
	"sparktalk/internal/config"
	"sparktalk/internal/llm"
	"sparktalk/internal/media"
	"sparktalk/internal/orchestrator"
	"sparktalk/internal/tts"
)

// Explicit opt-in: requires exclusive ownership of the local runtime during test.
func TestWorkloadSwapLive(t *testing.T) {
	path := os.Getenv("SPARKTALK_WORKLOAD_LIVE_CONFIG")
	if path == "" {
		t.Skip("live runtime not requested")
	}
	cfg, _, err := config.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	cfg.Runtime.ActiveBundle = os.Getenv("SPARKTALK_WORKLOAD_LIVE_BUNDLE")
	if cfg.Runtime.ActiveBundle == "" {
		cfg.Runtime.ActiveBundle = "flash-next"
	}
	cfg.ASR.Enabled = true
	cfg.Image.Enabled = true
	withTTS := os.Getenv("SPARKTALK_WORKLOAD_LIVE_TTS") == "1"
	if withTTS {
		cfg.TTS.Enabled = true
	}
	cfg.Normalize()
	s, _ := testImageServer(t)
	s.cfg = cfg
	s.asr = asr.New(cfg.ASR)
	s.tts = tts.New(cfg.TTS)
	s.runtime, err = orchestrator.NewControllerWithCatalog(*cfg.Runtime.Catalog)
	if err != nil {
		t.Fatal(err)
	}
	s.runtime.ConfigurePaths(cfg.Runtime.DataDir, cfg.Runtime.ModelCache)
	t.Cleanup(s.runtime.Close)
	audio, err := os.Open(os.Getenv("SPARKTALK_WORKLOAD_LIVE_AUDIO"))
	if err != nil {
		t.Fatal(err)
	}
	defer audio.Close()
	filename := filepath.Base(audio.Name())
	item, err := s.media.SaveReader(audio, filename, mime.TypeByExtension(filepath.Ext(filename)), media.MaxRemoteVideoBytes)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()
	rounds := 2
	if os.Getenv("SPARKTALK_WORKLOAD_LIVE_ROUNDS") == "1" {
		rounds = 1
	}
	var priorImagePID string
	speak := func(name string) {
		t.Helper()
		started := time.Now()
		request := httptest.NewRequest(http.MethodPost, "/api/tts/speech", strings.NewReader(`{"text":"안녕하세요. 이미지 생성과 음성 인식, 음성 합성을 같은 컴퓨터에서 순서대로 처리합니다. 요청이 끝난 뒤에는 잠시 모델을 유지해서 다음 문장을 빠르게 읽습니다."}`)).WithContext(ctx)
		response := httptest.NewRecorder()
		s.synthesizeSpeech(response, request)
		if response.Code != http.StatusOK || response.Body.Len() < 1000 {
			t.Errorf("TTS %s: status=%d body=%s", name, response.Code, response.Body.String())
			return
		}
		if directory := os.Getenv("SPARKTALK_WORKLOAD_LIVE_TTS_OUTPUT"); directory != "" {
			if e := os.WriteFile(filepath.Join(directory, name+".pcm"), response.Body.Bytes(), 0600); e != nil {
				t.Error(e)
			}
			if e := os.WriteFile(filepath.Join(directory, name+".rate"), []byte(response.Header().Get("X-Audio-Sample-Rate")), 0600); e != nil {
				t.Error(e)
			}
		}
		t.Logf("TTS %s complete %s, bytes=%d", name, time.Since(started), response.Body.Len())
	}
	for round := 0; round < rounds; round++ {
		swapBefore := liveSwapCounters(t)
		var wg sync.WaitGroup
		wg.Add(2)
		if withTTS {
			wg.Add(1)
			go func() {
				defer wg.Done()
				speak(fmt.Sprintf("combined-tts-round%d", round+1))
			}()
		}
		go func() {
			defer wg.Done()
			if os.Getenv("SPARKTALK_WORKLOAD_LIVE_ASR_ONLY") == "1" {
				return
			}
			started := time.Now()
			call := llm.ToolCall{ID: "live-image", Function: llm.FunctionCall{Name: "image_generate", Arguments: fmt.Sprintf(`{"operation":"generate","prompt":"A small garden turtle on a mossy log in a sunlit forest, nature photography","size":"1024x1024","seed":%d}`, 42+round)}}
			result, e := s.executeImageGenerateTool(ctx, "session", cfg.Image, call, func(string, any) error { return nil })
			if e != nil {
				t.Errorf("image: %v", e)
			} else if len(result.Attachments) != 1 {
				t.Error("missing image")
			} else {
				t.Logf("image complete %s", time.Since(started))
				if path := os.Getenv("SPARKTALK_WORKLOAD_LIVE_IMAGE_OUTPUT"); path != "" {
					f, e := s.media.Open(result.Attachments[0])
					if e != nil {
						t.Error(e)
					} else {
						data, e := io.ReadAll(f)
						f.Close()
						if e != nil {
							t.Error(e)
						} else if e = os.WriteFile(strings.TrimSuffix(path, filepath.Ext(path))+fmt.Sprintf("-round%d", round+1)+filepath.Ext(path), data, 0600); e != nil {
							t.Error(e)
						}
					}
				}
			}
		}()
		go func() {
			defer wg.Done()
			started := time.Now()
			if os.Getenv("SPARKTALK_WORKLOAD_LIVE_IMAGE_ONLY") == "1" {
				return
			}
			asrCfg := cfg.ASR
			asrCfg.Prompt = time.Now().String() // separate transcript-cache key for each round
			result, e := s.transcribeAttachment(ctx, item, asrCfg)
			if e != nil {
				t.Errorf("ASR: %v", e)
			} else if result.Text == "" {
				t.Error("empty transcript")
			} else {
				t.Logf("ASR complete %s, chars=%d, diarization=%s", time.Since(started), len(result.Text), result.DiarizationStatus)
				if cfg.ASR.Diarization && result.DiarizationStatus != "completed" {
					t.Errorf("diarization: %s %s", result.DiarizationStatus, result.Warning)
				}
			}
		}()
		wg.Wait()
		swapAfter := liveSwapCounters(t)
		t.Logf("round %d swap read=%d bytes write=%d bytes", round+1, swapAfter[0]-swapBefore[0], swapAfter[1]-swapBefore[1])
		if os.Getenv("SPARKTALK_WORKLOAD_LIVE_IMAGE_ONLY") == "1" {
			imageComponent, _ := s.runtime.Catalog().ResolveComponent("flash-next", "flux2")
			data, err := exec.Command("docker", "inspect", "-f", "{{.State.Pid}}", imageComponent.Container).Output()
			pid := strings.TrimSpace(string(data))
			if err != nil || pid == "" || pid == "0" {
				t.Fatalf("image service not retained: %q %v", pid, err)
			}
			if priorImagePID != "" && pid != priorImagePID {
				t.Fatalf("image service restarted: %s -> %s", priorImagePID, pid)
			}
			priorImagePID = pid
			t.Logf("reusable image PID %s", pid)
		}
		if s.runtime.Operation().State != "complete" {
			t.Fatalf("cleanup: %+v", s.runtime.Operation())
		}
		if t.Failed() {
			return
		}
	}
	if withTTS && !t.Failed() {
		var priorPID string
		for i := 1; i <= 2; i++ {
			speak(fmt.Sprintf("combined-tts-warm%d", i))
			component, _ := s.runtime.Catalog().ResolveComponent("flash-next", "qwen3-tts")
			data, err := exec.Command("docker", "inspect", "-f", "{{.State.Pid}}", component.Container).Output()
			pid := strings.TrimSpace(string(data))
			if err != nil || pid == "" || pid == "0" || (priorPID != "" && priorPID != pid) {
				t.Fatalf("TTS warm reuse: prior=%s pid=%s err=%v", priorPID, pid, err)
			}
			priorPID = pid
			t.Logf("reusable TTS PID %s", pid)
		}
	}
}

// Global counters include unrelated host activity; they establish actual I/O,
// not attribution to a particular model or outstanding swap occupancy.
func liveSwapCounters(t *testing.T) [2]uint64 {
	t.Helper()
	data, err := os.ReadFile("/proc/vmstat")
	if err != nil {
		t.Fatal(err)
	}
	var out [2]uint64
	for _, line := range strings.Split(string(data), "\n") {
		f := strings.Fields(line)
		if len(f) != 2 {
			continue
		}
		for i, key := range []string{"pswpin", "pswpout"} {
			if f[0] == key {
				n, e := strconv.ParseUint(f[1], 10, 64)
				if e != nil {
					t.Fatal(e)
				}
				out[i] = n * uint64(os.Getpagesize())
			}
		}
	}
	return out
}
