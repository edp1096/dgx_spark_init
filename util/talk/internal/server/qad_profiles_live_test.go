package server

import (
	"context"
	"encoding/json"
	"fmt"
	"image"
	_ "image/png"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"sparktalk/internal/asr"
	"sparktalk/internal/config"
	"sparktalk/internal/knowledge"
	"sparktalk/internal/llm"
	"sparktalk/internal/media"
	"sparktalk/internal/orchestrator"
	"sparktalk/internal/tts"
	"strings"
	"testing"
	"time"
)

func TestLiveQADSmallKVWorkloads(t *testing.T) {
	if os.Getenv("TALK_LIVE_QAD_PROFILES") != "1" {
		t.Skip("explicit QAD 512K/384K workloads")
	}
	report := os.Getenv("TALK_LIVE_QAD_REPORT")
	mode := os.Getenv("TALK_LIVE_QAD_MODE")
	video := os.Getenv("TALK_LIVE_QAD_VIDEO")
	if report == "" || video == "" {
		t.Fatal("report/video required")
	}
	if err := os.MkdirAll(report, 0700); err != nil {
		t.Fatal(err)
	}
	cfg, _, err := config.Load("../../dist/sparktalk.yaml")
	if err != nil {
		t.Fatal(err)
	}
	b, ok := cfg.Runtime.Catalog.Bundle("flash-next")
	if !ok || cfg.Runtime.ActiveBundle != "flash-next" || (b.ContextTokens != 524288 && b.ContextTokens != 393216) {
		t.Fatal("live small-KV QAD profile required")
	}
	cfg.ASR.Enabled = true
	cfg.TTS.Enabled = true
	cfg.Image.Enabled = true
	if mode == "production-resident" {
		if b.ContextTokens != 524288 || cfg.Image.Endpoint != "http://127.0.0.1:8691" {
			t.Fatal("production QAD512K/normal image endpoint required")
		}
		for _, id := range []string{"flux2", "nemotron-asr", "qwen3-tts"} {
			x, found := cfg.Runtime.Catalog.ResolveComponent(b.ID, id)
			if !found || !x.KeepResident || !x.StartAfterLLM {
				t.Fatal("production resident binding missing: " + id)
			}
			if id == "flux2" && x.RuntimeOptions["IMAGE_RESIDENCY"] != "dit" {
				t.Fatal("production DiT binding missing")
			}
		}
	}
	if mode == "resident" || mode == "dit-resident" {
		keep, budget := true, 24.0
		if mode == "dit-resident" {
			budget = 14.0 // Single DiT plus sequential auxiliaries; not the whole-model cache.
			cfg.Image.Endpoint = "http://127.0.0.1:18731"
			for i := range cfg.Runtime.Catalog.Components {
				x := &cfg.Runtime.Catalog.Components[i]
				if x.ID == "flux2" {
					x.Container = "sparktalk-qwen-image21-dit-trial"
					x.Endpoint = cfg.Image.Endpoint
					x.HealthURL = cfg.Image.Endpoint + "/health"
					x.Port = 18731
					x.StartupMemoryGiB, x.MemoryGiB, x.WorkspaceMemoryGiB = 7, 14, 6
				}
			}
		}
		for i := range cfg.Runtime.Catalog.Bundles {
			x := &cfg.Runtime.Catalog.Bundles[i]
			if x.ID != "flash-next" {
				continue
			}
			for _, id := range []string{"flux2", "nemotron-asr", "qwen3-tts"} {
				binding := x.Bindings[id]
				binding.KeepResident = &keep
				binding.StartAfterLLM = &keep
				if id == "flux2" {
					binding.MemoryGiB = &budget
					if mode == "dit-resident" {
						startup, port := 7.0, 18731
						endpoint, health := cfg.Image.Endpoint, cfg.Image.Endpoint+"/health"
						binding.StartupMemoryGiB = &startup
						binding.Endpoint, binding.HealthURL, binding.Port = &endpoint, &health, &port
					}
				}
				x.Bindings[id] = binding
			}
		}
	}
	s, _ := testImageServer(t)
	s.cfg = cfg
	s.collector = knowledge.NewCollectorClient(cfg.Extra.CollectorEndpoint)
	s.asr = asr.New(cfg.ASR)
	s.tts = tts.New(cfg.TTS)
	s.runtime, err = orchestrator.NewControllerWithCatalog(*cfg.Runtime.Catalog)
	if err != nil {
		t.Fatal(err)
	}
	s.runtime.ConfigurePaths(cfg.Runtime.DataDir, cfg.Runtime.ModelCache)
	t.Cleanup(s.runtime.Close)
	identity := func(container string) string {
		out, e := exec.Command("docker", "inspect", "-f", "{{.Id}} {{.State.Pid}}", container).Output()
		if e != nil {
			return ""
		}
		return strings.TrimSpace(string(out))
	}
	before := identity("sglang-qwen38-fn")
	if before == "" {
		t.Fatal("QAD absent")
	}
	info := func() map[string]any {
		resp, e := http.Get(strings.TrimRight(cfg.Model.Endpoint, "/") + "/server_info")
		if e != nil {
			t.Fatal(e)
		}
		defer resp.Body.Close()
		var d map[string]any
		if e = json.NewDecoder(resp.Body).Decode(&d); e != nil {
			t.Fatal(e)
		}
		return d
	}
	startInfo := info()
	if int(startInfo["max_total_num_tokens"].(float64)) != b.ContextTokens || int(startInfo["context_length"].(float64)) != b.ContextTokens {
		t.Fatal("actual KV/context mismatch")
	}
	checks := []map[string]any{}
	snapshots := []map[string]any{}
	failure := ""
	residentBefore := map[string]string{}
	residentAfter := map[string]string{}
	stage := func(name string) {
		if p := os.Getenv("TALK_LIVE_QAD_STAGE"); p != "" {
			_ = os.WriteFile(p, []byte(fmt.Sprintf("%d/%s/%s", b.ContextTokens, mode, name)), 0600)
		}
	}
	snapshot := func(name string) {
		status := s.runtime.Snapshot(context.Background(), "flash-next")
		snapshots = append(snapshots, map[string]any{"stage": name, "snapshot": status})
		data, _ := json.Marshal(snapshots)
		_ = os.WriteFile(filepath.Join(report, "snapshots.json"), data, 0600)
	}
	run := func(name string, fn func(context.Context) error) bool {
		stage(name)
		ctx, cancel := context.WithTimeout(context.Background(), 8*time.Minute)
		defer cancel()
		started := time.Now()
		e := fn(ctx)
		row := map[string]any{"name": name, "elapsed_seconds": time.Since(started).Seconds(), "ok": e == nil}
		if e != nil {
			row["error"] = e.Error()
			failure = e.Error()
		}
		checks = append(checks, row)
		snapshot(name)
		t.Logf("%s: %v", name, e)
		return e == nil
	}
	imageJob := func(n int) func(context.Context) error {
		return func(ctx context.Context) error {
			seed := int64(1096 + n)
			prompt := "An adorable baby penguin on Antarctic snow, fluffy grey feathers, tiny orange feet, soft pastel sky, charming storybook illustration, no text."
			if (mode == "dit-resident" || mode == "production-resident") && n == 3 {
				prompt += " The penguin is wearing a small blue scarf."
			}
			body, _ := json.Marshal(imageGenerationArgs{Operation: "generate", Size: "1024x1024", Seed: &seed, Prompt: prompt})
			result, e := s.executeImageGenerateTool(ctx, "session", cfg.Image, llm.ToolCall{Function: llm.FunctionCall{Name: "image_generate", Arguments: string(body)}}, func(string, any) error { return nil })
			if e != nil {
				return e
			}
			if len(result.Attachments) != 1 {
				return fmt.Errorf("missing image")
			}
			f, e := s.media.Open(result.Attachments[0])
			if e != nil {
				return e
			}
			data, e := io.ReadAll(f)
			f.Close()
			if e != nil {
				return e
			}
			path := filepath.Join(report, fmt.Sprintf("image-%d.png", n))
			if e = os.WriteFile(path, data, 0600); e != nil {
				return e
			}
			f, e = os.Open(path)
			if e != nil {
				return e
			}
			dim, _, e := image.DecodeConfig(f)
			f.Close()
			if e != nil || dim.Width != 1024 || dim.Height != 1024 {
				return fmt.Errorf("invalid generated image")
			}
			return nil
		}
	}
	asrJob := func(ctx context.Context) error {
		f, e := os.Open(video)
		if e != nil {
			return e
		}
		item, e := s.media.SaveReader(f, "sample-video.mp4", "video/mp4", media.MaxRemoteVideoBytes)
		f.Close()
		if e != nil {
			return e
		}
		result, e := s.transcribeAttachment(ctx, item, cfg.ASR)
		if e != nil {
			return e
		}
		data, _ := json.MarshalIndent(result, "", "  ")
		_ = os.WriteFile(filepath.Join(report, "transcript.json"), data, 0600)
		if result.Text == "" || result.DiarizationStatus != "completed" {
			return fmt.Errorf("incomplete ASR")
		}
		return nil
	}
	ttsJob := func(ctx context.Context) error {
		req := httptest.NewRequest(http.MethodPost, "/api/tts/speech", strings.NewReader(`{"text":"안녕하세요. GPU에서 음성을 합성합니다. 영상은 팔육사 사팔공 초당 이십사 에프피에스입니다."}`)).WithContext(ctx)
		response := httptest.NewRecorder()
		s.synthesizeSpeech(response, req)
		if response.Code != 200 || response.Body.Len() < 4800 {
			return fmt.Errorf("TTS HTTP %d: %s", response.Code, response.Body.String())
		}
		if e := os.WriteFile(filepath.Join(report, "speech.pcm"), response.Body.Bytes(), 0600); e != nil {
			return e
		}
		return nil
	}
	snapshot("initial")
	if mode == "dit-resident" || mode == "production-resident" {
		extraJob := func(ctx context.Context) error {
			collected, e := s.collectSource(ctx, "https://example.com/", "browser", false)
			if e != nil {
				return e
			}
			if !strings.Contains(collected.Text, "Example Domain") {
				return fmt.Errorf("incomplete browser collection")
			}
			document, e := s.executeDocumentGenerate(ctx, llm.ToolCall{Function: llm.FunctionCall{Name: "document_generate", Arguments: `{"format":"pdf","filename":"qad-dit-extra-check","title":"QAD DiT 상주 검사","sections":[{"paragraphs":["QAD와 이미지 DiT, ASR, TTS를 유지하며 Extra를 실행합니다."]}]}`}})
			if e != nil {
				return e
			}
			if len(document.Attachments) != 1 {
				return fmt.Errorf("missing PDF")
			}
			file, e := s.media.Open(document.Attachments[0])
			if e != nil {
				return e
			}
			data, e := io.ReadAll(file)
			file.Close()
			if e != nil {
				return e
			}
			if !strings.HasPrefix(string(data), "%PDF") {
				return fmt.Errorf("invalid PDF")
			}
			if e = os.WriteFile(filepath.Join(report, "extra-document.pdf"), data, 0600); e != nil {
				return e
			}
			for _, id := range []string{"extra-collector", "extra-documents", "extra-media"} {
				if e = s.runtime.ComponentAction(id, "stop", "flash-next"); e != nil {
					return e
				}
				for {
					state := s.runtime.Snapshot(ctx, "flash-next").Operation
					if state.State != "running" {
						if state.State == "failed" {
							return fmt.Errorf("Extra stop: %s", state.Error)
						}
						break
					}
					select {
					case <-ctx.Done():
						return ctx.Err()
					case <-time.After(time.Second):
					}
				}
			}
			response, e := http.Get(strings.TrimRight(cfg.Extra.SSHEndpoint, "/") + "/health")
			if e != nil {
				return e
			}
			response.Body.Close()
			if response.StatusCode != 200 {
				return fmt.Errorf("SSH API unavailable")
			}
			return nil
		}
		llmJob := func(ctx context.Context) error {
			body, _ := json.Marshal(map[string]any{"model": cfg.Model.DefaultModel, "messages": []map[string]string{{"role": "user", "content": "한국어로 짧게 인사해 주세요."}}, "max_tokens": 64, "temperature": 0, "stream": false, "chat_template_kwargs": map[string]bool{"enable_thinking": false}})
			req, e := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(cfg.Model.Endpoint, "/")+"/v1/chat/completions", strings.NewReader(string(body)))
			if e != nil {
				return e
			}
			req.Header.Set("Content-Type", "application/json")
			response, e := http.DefaultClient.Do(req)
			if e != nil {
				return e
			}
			defer response.Body.Close()
			data, e := io.ReadAll(response.Body)
			if e != nil {
				return e
			}
			_ = os.WriteFile(filepath.Join(report, "llm-response.json"), data, 0600)
			if response.StatusCode != 200 {
				return fmt.Errorf("LLM HTTP %d: %s", response.StatusCode, data)
			}
			var d struct {
				Choices []struct {
					Message struct {
						Content string `json:"content"`
					} `json:"message"`
				} `json:"choices"`
			}
			if e = json.Unmarshal(data, &d); e != nil {
				return e
			}
			if len(d.Choices) == 0 || d.Choices[0].Message.Content == "" {
				return fmt.Errorf("empty LLM response")
			}
			return nil
		}
		clearUnusedCUDA := func(ctx context.Context) error {
			control := func(path, body string) error {
				req, e := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(cfg.Model.Endpoint, "/")+path, strings.NewReader(body))
				if e != nil {
					return e
				}
				req.Header.Set("Content-Type", "application/json")
				response, e := (&http.Client{Timeout: 15 * time.Second}).Do(req)
				if e != nil {
					return e
				}
				defer response.Body.Close()
				if response.StatusCode != 200 {
					data, _ := io.ReadAll(response.Body)
					return fmt.Errorf("%s: %s", path, data)
				}
				return nil
			}
			initial := info()
			if e := control("/pause_generation", `{"mode":"in_place"}`); e != nil {
				_ = control("/continue_generation", `{}`)
				return e
			}
			if e := control("/continue_generation", `{"torch_empty_cache":true}`); e != nil {
				_ = control("/continue_generation", `{}`)
				return e
			}
			final := info()
			data, _ := json.MarshalIndent(map[string]any{"before": initial, "after": final}, "", "  ")
			_ = os.WriteFile(filepath.Join(report, "unused-cuda-cache.json"), data, 0600)
			if initial["max_total_num_tokens"] != final["max_total_num_tokens"] {
				return fmt.Errorf("KV changed during unused allocator cleanup")
			}
			return nil
		}
		imageStartup := func(ctx context.Context) error {
			for {
				if _, e := os.Stat(filepath.Join(filepath.Dir(report), "docker-launch.json")); e == nil {
					state, _ := exec.Command("docker", "inspect", "-f", "{{.State.Running}}", "sparktalk-qwen-image21-dit-trial").Output()
					if strings.TrimSpace(string(state)) != "true" {
						logs, _ := exec.Command("docker", "logs", "--tail", "15", "sparktalk-qwen-image21-dit-trial").CombinedOutput()
						return fmt.Errorf("DiT worker startup stopped: %s", logs)
					}
				}
				req, e := http.NewRequestWithContext(ctx, http.MethodGet, cfg.Image.Endpoint+"/health", nil)
				if e != nil {
					return e
				}
				response, e := http.DefaultClient.Do(req)
				if e == nil {
					response.Body.Close()
					if response.StatusCode == 200 {
						return nil
					}
				}
				select {
				case <-ctx.Done():
					return ctx.Err()
				case <-time.After(time.Second):
				}
			}
		}
		if run("asr-initial", asrJob) && run("tts-initial", ttsJob) && run("unused-cuda-cache", clearUnusedCUDA) && run("image-startup", imageStartup) {
			containers := []string{"sglang-qwen38-fn", "sparktalk-nemotron-asr", "sparktalk-qwen3-tts", "sparktalk-qwen-image21-dit-trial"}
			if mode == "production-resident" {
				containers[3] = "sparktalk-qwen-image21"
			}
			ids := map[string]string{}
			for _, name := range containers {
				ids[name] = identity(name)
				residentBefore[name] = ids[name]
			}
			if run("image-first", imageJob(1)) && run("asr-after-image", asrJob) && run("tts-after-image", ttsJob) && run("llm-after-image", llmJob) && run("image-repeat", imageJob(2)) && run("extra-work-and-stop", extraJob) {
				run("idle-over-grace", func(ctx context.Context) error {
					select {
					case <-ctx.Done():
						return ctx.Err()
					case <-time.After(130 * time.Second):
					}
					for _, name := range containers {
						if ids[name] == "" || ids[name] != identity(name) {
							return fmt.Errorf("resident process changed: %s", name)
						}
					}
					return nil
				})
				if failure == "" {
					run("image-after-idle", imageJob(3))
				}
			}
			for _, name := range containers {
				residentAfter[name] = identity(name)
				if ids[name] == "" || ids[name] != identity(name) {
					failure = "resident process changed: " + name
				}
			}
		}
	} else if mode == "resident" {
		if run("image-resident-initial", imageJob(1)) && run("asr-resident", asrJob) && run("tts-resident", ttsJob) {
			run("image-resident-repeat", imageJob(2))
		}
	} else {
		if run("asr", asrJob) && run("tts", ttsJob) && run("image-initial", imageJob(1)) {
			pid := identity("sparktalk-qwen-image21")
			if run("image-repeat", imageJob(2)) && pid != identity("sparktalk-qwen-image21") {
				failure = "image service restarted between repeated jobs"
			}
		}
	}
	finalInfo := info()
	same := identity("sglang-qwen38-fn") == before
	output := map[string]any{"mode": mode, "context_tokens": b.ContextTokens, "capacity_tokens": finalInfo["max_total_num_tokens"], "same_llm_id_and_pid": same, "checks": checks, "failure": failure, "initial_native_memory": startInfo["internal_states"].([]any)[0].(map[string]any)["qad_memory"], "final_native_memory": finalInfo["internal_states"].([]any)[0].(map[string]any)["qad_memory"]}
	if mode == "dit-resident" || mode == "production-resident" {
		output["resident_processes_before"], output["resident_processes_after"] = residentBefore, residentAfter
		container := "sparktalk-qwen-image21-dit-trial"
		if mode == "production-resident" {
			container = "sparktalk-qwen-image21"
		}
		logs, _ := exec.Command("docker", "logs", container).CombinedOutput()
		_ = os.WriteFile(filepath.Join(report, "worker.log"), logs, 0600)
	}
	data, _ := json.MarshalIndent(output, "", "  ")
	_ = os.WriteFile(filepath.Join(report, "results.json"), data, 0600)
	stage("done")
	if !same || int(finalInfo["max_total_num_tokens"].(float64)) != b.ContextTokens {
		t.Fatal("LLM/KV not retained")
	}
	if failure != "" && mode != "resident" {
		t.Fatal(failure)
	}
}
