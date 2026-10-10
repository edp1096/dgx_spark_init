package orchestrator

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

// Explicit real Docker lifecycle: no external-controller or capacity-check overlay.
func TestLiveEXL3Q4ManagedBundle(t *testing.T) {
	if os.Getenv("TALK_LIVE_EXL3_Q4") != "1" {
		t.Skip("explicit 1M Q4 CUDA lifecycle")
	}
	report := os.Getenv("TALK_LIVE_EXL3_Q4_REPORT")
	if report == "" {
		t.Fatal("report directory required")
	}
	c, err := NewController()
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	c.ConfigurePaths("/home/edp1096/.local/share/sparktalk", "/home/edp1096/.cache/huggingface")
	ctx, cancel := context.WithTimeout(context.Background(), 35*time.Minute)
	defer cancel()
	if err := c.StartBundle(ctx, "qwen38fn_exl3_q4", 1.5); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(30 * time.Minute)
	for c.Operation().State == "running" && time.Now().Before(deadline) {
		time.Sleep(time.Second)
	}
	op := c.Operation()
	b, _ := json.MarshalIndent(op, "", "  ")
	os.WriteFile(filepath.Join(report, "managed-start.json"), b, 0600)
	if op.State != "complete" {
		t.Fatalf("managed start: %+v", op)
	}
	snapshot := c.Snapshot(ctx, "qwen38fn_exl3_q4")
	b, _ = json.MarshalIndent(snapshot, "", "  ")
	os.WriteFile(filepath.Join(report, "managed-snapshot.json"), b, 0600)
	for _, id := range []string{"qwen38fn_exl3_q4", "nemotron-asr", "extra-embedding"} {
		x, _ := c.Catalog().ResolveComponent("qwen38fn_exl3_q4", id)
		if !c.isHealthy(ctx, x) {
			t.Fatal("resident core unhealthy", id)
		}
	}
	identities := func() string {
		data, err := exec.Command("docker", "inspect", "--format", "{{.Id}} {{.State.Pid}}", "sparktalk-qwen38fn-exl3-q4", "sparktalk-nemotron-asr", "sparktalk-embedding").Output()
		if err != nil {
			t.Fatal(err)
		}
		return string(data)
	}
	before := identities()
	for _, id := range []string{"extra-media", "extra-ssh", "extra-collector", "extra-documents"} {
		release, err := c.AcquireWorkload(ctx, "qwen38fn_exl3_q4", id, 1.5)
		if err != nil {
			t.Fatal(id, err)
		}
		x, _ := c.Catalog().ResolveSupport("qwen38fn_exl3_q4", id)
		if !c.isHealthy(ctx, x) {
			t.Fatal("extra unhealthy", id)
		}
		if err := release(); err != nil {
			t.Fatal(id, err)
		}
		core, _ := c.Catalog().ResolveComponent("qwen38fn_exl3_q4", "qwen38fn_exl3_q4")
		if !c.isHealthy(ctx, core) {
			t.Fatal("core interrupted by extra", id)
		}
	}
	if identities() != before {
		t.Fatal("Extra restarted a resident core process")
	}
	// Leave the core running for API/vision/ASR checks in the integration runner.
}

func TestLiveEXL3Q4ManagedStop(t *testing.T) {
	if os.Getenv("TALK_LIVE_EXL3_Q4_STOP") != "1" {
		t.Skip("explicit managed stop")
	}
	c, err := NewController()
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	c.ConfigurePaths("/home/edp1096/.local/share/sparktalk", "/home/edp1096/.cache/huggingface")
	if err = c.StopBundle("qwen38fn_exl3_q4"); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(time.Minute)
	for c.Operation().State == "running" && time.Now().Before(deadline) {
		time.Sleep(time.Second)
	}
	op := c.Operation()
	data, _ := json.MarshalIndent(op, "", "  ")
	if report := os.Getenv("TALK_LIVE_EXL3_Q4_REPORT"); report != "" {
		os.WriteFile(filepath.Join(report, "managed-stop.json"), data, 0600)
	}
	if op.State != "complete" {
		t.Fatal(op)
	}
	for _, id := range []string{"qwen38fn_exl3_q4", "nemotron-asr", "extra-embedding"} {
		x, _ := c.Catalog().ResolveComponent("qwen38fn_exl3_q4", id)
		if c.componentRunning(context.Background(), x) {
			t.Fatal("core remained running", id)
		}
	}
}
