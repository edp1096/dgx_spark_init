package server

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sparktalk/internal/config"
	"sparktalk/internal/knowledge"
	"sparktalk/internal/llm"
	"sparktalk/internal/orchestrator"
	"strings"
	"testing"
	"time"
)

// Explicit CUDA/runtime integration. Artifacts and messages use a temporary DB.
func TestLiveRadixArkExtras(t *testing.T) {
	if os.Getenv("TALK_LIVE_RADIXARK_EXTRAS") != "1" {
		t.Skip("explicit NVFP4/Extra integration")
	}
	report := os.Getenv("TALK_LIVE_RADIXARK_EXTRAS_REPORT")
	if report == "" {
		t.Fatal("report directory required")
	}
	cfg, _, err := config.Load("../../dist/sparktalk.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Runtime.ActiveBundle != "flash-next-radixark" || cfg.TTS.Enabled || cfg.Image.Enabled {
		t.Fatal("NVFP4 + Extra profile required")
	}
	s, _ := testImageServer(t)
	s.cfg = cfg
	s.collector = knowledge.NewCollectorClient(cfg.Extra.CollectorEndpoint)
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
	embeddingBefore := liveRadixArkEmbeddingIdentity(t, cfg)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	if err := os.MkdirAll(report, 0700); err != nil {
		t.Fatal(err)
	}
	var results []string
	release, err := s.acquireWorkload(ctx, "extra-media")
	if err != nil {
		t.Fatal(err)
	}
	// One second of PCM WAV; verify actual FFmpeg conversion through the lease.
	var wav bytes.Buffer
	wav.WriteString("RIFF")
	binary.Write(&wav, binary.LittleEndian, uint32(36+48000))
	wav.WriteString("WAVEfmt ")
	binary.Write(&wav, binary.LittleEndian, uint32(16))
	binary.Write(&wav, binary.LittleEndian, uint16(1))
	binary.Write(&wav, binary.LittleEndian, uint16(1))
	binary.Write(&wav, binary.LittleEndian, uint32(24000))
	binary.Write(&wav, binary.LittleEndian, uint32(48000))
	binary.Write(&wav, binary.LittleEndian, uint16(2))
	binary.Write(&wav, binary.LittleEndian, uint16(16))
	wav.WriteString("data")
	binary.Write(&wav, binary.LittleEndian, uint32(48000))
	wav.Write(make([]byte, 48000))
	req, _ := http.NewRequestWithContext(ctx, "POST", strings.TrimRight(cfg.Extra.MediaEndpoint, "/")+"/v1/audio/extract?sample_rate=16000", &wav)
	req.Header.Set("Content-Type", "audio/wav")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		_ = release(err)
		t.Fatal(err)
	}
	data, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil || resp.StatusCode != 200 || len(data) < 32000 {
		_ = release(fmt.Errorf("bad conversion"))
		t.Fatalf("media %d: %s", resp.StatusCode, data)
	}
	if err := release(); err != nil {
		t.Fatal(err)
	}
	results = append(results, "Media: PCM WAV 24000 → 16000 Hz")
	collected, err := s.collectSource(ctx, "https://example.com/", "browser", false)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(collected.Text, "Example Domain") {
		t.Fatalf("collector returned %q", collected.Text)
	}
	results = append(results, "Collector: actual Chromium collection")
	document, err := s.executeDocumentGenerate(ctx, llm.ToolCall{Function: llm.FunctionCall{Name: "document_generate", Arguments: `{"format":"pdf","filename":"nvfp4-extra-check","title":"NVFP4 Extra 검사","sections":[{"paragraphs":["Qwen 본체와 1M KV를 유지하며 Extra 문서를 생성합니다."]}]}`}})
	if err != nil {
		t.Fatal(err)
	}
	if len(document.Attachments) != 1 {
		t.Fatal("PDF missing")
	}
	file, err := s.media.Open(document.Attachments[0])
	if err != nil {
		t.Fatal(err)
	}
	data, err = io.ReadAll(file)
	file.Close()
	if err != nil || !bytes.HasPrefix(data, []byte("%PDF")) {
		t.Fatal("invalid PDF")
	}
	if err := os.WriteFile(filepath.Join(report, "nvfp4-extra-check.pdf"), data, 0600); err != nil {
		t.Fatal(err)
	}
	results = append(results, "Documents: actual PDF saved in temporary store")
	resp, err = http.Get(strings.TrimRight(cfg.Extra.SSHEndpoint, "/") + "/health")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatal("SSH API unavailable")
	}
	results = append(results, "SSH: existing API healthy")
	if before != identity() {
		t.Fatal("LLM restarted during Extra work")
	}
	if embeddingBefore != liveRadixArkEmbeddingIdentity(t, cfg) {
		t.Fatal("resident embedding restarted during Extra work")
	}
	resp, err = http.Get(strings.TrimRight(cfg.Model.Endpoint, "/") + "/server_info")
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
	b, _ := s.runtime.Catalog().Bundle("flash-next-radixark")
	for _, id := range b.Components {
		x, _ := s.runtime.Catalog().ResolveComponent(b.ID, id)
		if !x.IsSupport() && id != "flash-next-radixark" && id != "nemotron-asr" {
			t.Fatal("excluded model included", id)
		}
	}
	out, _ := json.MarshalIndent(map[string]any{"checks": results, "same_llm_id_and_pid": true, "context_tokens": info.Context, "kv_tokens": info.Capacity}, "", "  ")
	if err := os.WriteFile(filepath.Join(report, "results.json"), out, 0600); err != nil {
		t.Fatal(err)
	}
	t.Log(string(out))
}

func liveRadixArkEmbeddingIdentity(t *testing.T, cfg config.Config) string {
	t.Helper()
	if !cfg.SemanticSearchEnabled() {
		t.Fatal("NVFP4 resident embedding must be enabled for this audit")
	}
	resp, err := http.Get(strings.TrimRight(cfg.Embedding.Endpoint, "/") + "/health")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var state struct {
		Ready  bool   `json:"ready"`
		Device string `json:"device"`
		Dtype  string `json:"dtype"`
	}
	if resp.StatusCode != 200 || json.NewDecoder(resp.Body).Decode(&state) != nil || !state.Ready || state.Device != "cuda" || state.Dtype != "bf16" {
		t.Fatal("resident CUDA BF16 embedding unavailable", state)
	}
	b, err := exec.Command("docker", "inspect", "-f", "{{.Id}} {{.State.Pid}}", "sparktalk-embedding").Output()
	if err != nil {
		t.Fatal(err)
	}
	return strings.TrimSpace(string(b))
}
