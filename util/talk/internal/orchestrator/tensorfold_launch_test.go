package orchestrator

import (
	"encoding/binary"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestTensorFoldPreservesQualifiedMemoryFloor(t *testing.T) {
	cat, _ := LoadCatalog()
	b, _ := cat.Bundle("ds41")
	if cat.minimumBundleReserve(b, 1.5) != 4 || cat.minimumBundleReserve(b, 8) != 8 {
		t.Fatal("TensorFold reserve must preserve the 4 GiB floor and larger user reserves")
	}
	b, _ = cat.Bundle("flash-next-radixark")
	if cat.minimumBundleReserve(b, 1.5) != 1.5 {
		t.Fatal("another runtime's reserve changed")
	}
}

func TestTensorFoldStartupProgressUsesEngineStages(t *testing.T) {
	x := Component{Controller: "ds41-cluster", ProgressKind: "service"}
	logs := ""
	for _, stage := range []struct{ line, key string }{
		{"images: native", "container"},
		{"info: tp: rank 0 of 2 up (roce)", "weights"},
		{"info: dsv41 load: weights 95000 ms", "pool"},
		{"info: graph cache row graphs: captured", "calibration"},
		{"[tensorfold] serving DeepSeek-V4.1-Flash-TF at http://127.0.0.1:8010/v1", "api"},
	} {
		logs += stage.line + "\n"
		if p := inferProgress(x, logs); p.Key != stage.key || strings.Contains(p.Phase, "보조 서비스") {
			t.Fatalf("wrong TensorFold startup stage: %+v", p)
		}
	}
}

// Execute the embedded shell entrypoint with fake Docker. This catches profile
// sourcing that overwrites rank-local paths or accidentally shares variant caches.
func TestTensorFoldLaunchUsesWorkerCacheAndSeparateVariantState(t *testing.T) {
	recipe := t.TempDir()
	for _, name := range []string{"env.sample", "manage.sh", "runtime.sh", "models.sh", "launch.sh", "profile.env", "pfdense-table.json"} {
		data, err := os.ReadFile(filepath.Join("recipe_sources", "ds41", name))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(recipe, name), data, 0700); err != nil {
			t.Fatal(err)
		}
	}
	cache := t.TempDir()
	root := filepath.Join(cache, "tensorfold", "deepseek-v41-flash-1017")
	for _, file := range []string{"models/abliterated/config.json", "models/abliterated/model.safetensors.index.json", "models/abliterated/ABLIT_META.json", "engram/rank1/engram-l1-r1of2.bin", "engram/rank1/engram-l14-r1of2.bin", "assets/rank1/engram-host.bin", "assets/rank1/aot/aot.json", "verification/pack.verified.json"} {
		path := filepath.Join(root, file)
		if err := os.MkdirAll(filepath.Dir(path), 0700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte("fixture"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	rope := make([]byte, 12)
	copy(rope, "DSV41RP1")
	binary.LittleEndian.PutUint32(rope[8:], 1050624)
	if err := os.WriteFile(filepath.Join(root, "assets/rank1/rope.bin"), rope, 0600); err != nil {
		t.Fatal(err)
	}
	mock := t.TempDir()
	docker := "#!/bin/bash\nif [[ $1 == image ]]; then echo \"$DSV41_IMAGE_ID\"; exit; fi\nprintf '%s\\n' \"$@\"\necho \"PROFILE=$TF_DSV41_POOL_TOKENS/$TF_DSV41_LOOKUP/$TF_DSV41_ENGRAM_PREFETCH\"\necho \"DEPTH=$TF_DSV41_DEPTH/$TF_DSV41_DRAFT_DEPTH\"\n"
	if err := os.WriteFile(filepath.Join(mock, "docker"), []byte(docker), 0700); err != nil {
		t.Fatal(err)
	}
	env := "MODEL_VARIANT=abliterated\nWORKER_HF_CACHE=" + shellQuote(cache) + "\nHF_CACHE=/missing-head-cache\nHEAD_RAIL_IP=10.200.0.1\nWORKER_NCCL_IF=enp1s0f1np1\nWORKER_NCCL_HCA=rocep1s0f1\n"
	if err := os.WriteFile(filepath.Join(recipe, ".env"), []byte(env), 0600); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("bash", "manage.sh", "rank", "1")
	cmd.Dir = recipe
	cmd.Env = append(os.Environ(), "PATH="+mock+":"+os.Getenv("PATH"))
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("worker launch: %v %s", err, out)
	}
	for _, expected := range []string{filepath.Join(root, "models/abliterated") + ":/model:ro", filepath.Join(root, "prepared/abliterated/rank1") + ":/prepared", filepath.Join(root, "state/abliterated/rank1") + ":/state", filepath.Join(root, "cache/rank1") + ":/cache", "sparktalk-ds41-tf-r1", "TF_TP_RANK=1", "follow", "PROFILE=2100000/1/1"} {
		if !strings.Contains(string(out), expected) {
			t.Fatalf("missing %q in %s", expected, out)
		}
	}
	if strings.Contains(string(out), "/missing-head-cache") || strings.Contains(string(out), "prepared/official") {
		t.Fatal("rank/variant cache isolation lost", string(out))
	}
	for _, value := range []string{"auto", "1", "2", "3", "4", "5"} {
		if err := os.WriteFile(filepath.Join(recipe, ".env"), []byte(env+"MTP_TOKENS="+value+"\n"), 0600); err != nil {
			t.Fatal(err)
		}
		cmd = exec.Command("bash", "manage.sh", "rank", "1")
		cmd.Dir = recipe
		cmd.Env = append(os.Environ(), "PATH="+mock+":"+os.Getenv("PATH"))
		output, err := cmd.CombinedOutput()
		want := "DEPTH=static/" + value
		if value == "auto" {
			want = "DEPTH=cost/5"
		}
		if err != nil || !strings.Contains(string(output), want) {
			t.Fatalf("MTP policy did not reach worker: %s: %v %s", value, err, output)
		}
	}
	if err := os.Remove(filepath.Join(root, "engram/rank1/engram-l14-r1of2.bin")); err != nil {
		t.Fatal(err)
	}
	cmd = exec.Command("bash", "manage.sh", "rank", "1")
	cmd.Dir = recipe
	cmd.Env = append(os.Environ(), "PATH="+mock+":"+os.Getenv("PATH"))
	out, err = cmd.CombinedOutput()
	if err == nil || !strings.Contains(string(out), "prerequisite missing") || strings.Contains(string(out), "--gpus") {
		t.Fatalf("incomplete rank started: %v %s", err, out)
	}
}
