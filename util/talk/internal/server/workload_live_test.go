package server

import (
	"context"
	"fmt"
	"io"
	"mime"
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
	cfg.Runtime.ActiveBundle = "flash-next"
	cfg.ASR.Enabled = true
	cfg.Image.Enabled = true
	cfg.Normalize()
	s, _ := testImageServer(t)
	s.cfg = cfg
	s.asr = asr.New(cfg.ASR)
	s.runtime, err = orchestrator.NewControllerWithCatalog(*cfg.Runtime.Catalog)
	if err != nil {
		t.Fatal(err)
	}
	s.runtime.ConfigurePaths(cfg.Runtime.DataDir, cfg.Runtime.ModelCache)
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
	for round := 0; round < rounds; round++ {
		swapBefore := liveSwapCounters(t)
		var wg sync.WaitGroup
		wg.Add(2)
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
						} else if e = os.WriteFile(path, data, 0600); e != nil {
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
			data, err := exec.Command("docker", "inspect", "-f", "{{.State.Pid}}", "flux2-klein-nvfp4-api").Output()
			pid := strings.TrimSpace(string(data))
			if err != nil || pid == "" || pid == "0" {
				t.Fatalf("image service not retained: %q %v", pid, err)
			}
			if priorImagePID != "" && pid != priorImagePID {
				t.Fatalf("image service restarted: %s -> %s", priorImagePID, pid)
			}
			priorImagePID = pid
			t.Logf("reusable Flux PID %s", pid)
		}
		if s.runtime.Operation().State != "complete" {
			t.Fatalf("cleanup: %+v", s.runtime.Operation())
		}
		if t.Failed() {
			return
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
