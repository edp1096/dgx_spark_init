package orchestrator

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestQADConfiguredContextChangesActualKVPoolTogether(t *testing.T) {
	for _, length := range []string{"655360", "1048576", "65536", "640k", "0", "655361", "2097152"} {
		data, err := composeAsset("compose.flash-next.yaml")
		if err != nil {
			t.Fatal(err)
		}
		var recipe map[string]any
		if err := yaml.Unmarshal(data, &recipe); err != nil {
			t.Fatal(err)
		}
		service := recipe["services"].(map[string]any)["runtime"].(map[string]any)
		err = applyQwenQADCheckpoint(service, Component{RuntimeOptions: map[string]string{"MODEL_VARIANT": "huihui_lil", "MTP_TOKENS": "3", "MAX_MODEL_LEN": length}})
		valid := length == "655360" || length == "1048576" || length == "65536"
		if !valid {
			if err == nil {
				t.Fatalf("invalid context accepted: %s", length)
			}
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		command := service["command"].([]any)
		flags := map[string]string{}
		for i, value := range command {
			if key, ok := value.(string); ok && strings.HasPrefix(key, "--") && i+1 < len(command) {
				flags[key], _ = command[i+1].(string)
			}
		}
		if flags["--context-length"] != length || flags["--max-total-tokens"] != length || flags["--speculative-num-steps"] != "3" || flags["--kv-cache-dtype"] != "fp8_e4m3" || flags["--mem-fraction-static"] != "0.86" {
			t.Fatalf("context/KV or existing engine options not preserved: %+v", flags)
		}
	}
}

func TestQADVariantRendersMatchingCheckpointAndAPIIdentity(t *testing.T) {
	for _, variant := range []string{"", "official", "abliterated", "huihui_lil", "invalid"} {
		t.Run(variant, func(t *testing.T) {
			dir := t.TempDir()
			os.WriteFile(filepath.Join(dir, "docker"), []byte("#!/bin/sh\ncase \"$*\" in *config) cat;; esac\n"), 0700)
			t.Setenv("PATH", dir+":"+os.Getenv("PATH"))
			c, _ := NewController()
			c.ConfigurePaths(dir, filepath.Join(dir, "models"))
			component, _ := c.Catalog().Component("flash-next")
			component.RuntimeOptions = map[string]string{"MODEL_VARIANT": variant}
			err := c.prepareOrStartComponent(context.Background(), component, true)
			if variant == "invalid" {
				if err == nil {
					t.Fatal("accepted invalid variant")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			data, err := os.ReadFile(filepath.Join(dir, "runtime", component.ID, "compose.yaml"))
			if err != nil {
				t.Fatal(err)
			}
			var recipe map[string]any
			if err = yaml.Unmarshal(data, &recipe); err != nil {
				t.Fatal(err)
			}
			command := recipe["services"].(map[string]any)["runtime"].(map[string]any)["command"].([]any)
			repo, revision, _ := QwenQADCheckpoint(variant)
			want := map[string]string{"--model-path": "/hf/hub/models--" + strings.ReplaceAll(repo, "/", "--") + "/snapshots/" + revision, "--served-model-name": repo, "--context-length": "1048576", "--kv-cache-dtype": "fp8_e4m3"}
			if variant == "huihui_lil" {
				want["--model-path"] = "/hf/" + QwenQADHuihuiLIL
			}
			for flag, value := range want {
				found := false
				for i, arg := range command {
					if arg == flag && i+1 < len(command) {
						found = command[i+1] == value
					}
				}
				if !found {
					t.Fatalf("%s mismatch: %v", flag, command)
				}
			}
		})
	}
}

func TestQADVariantBindingRoundTripKeepsTP2(t *testing.T) {
	catalog, _ := LoadCatalog()
	before, _ := catalog.Bundle("flash-next-tp2")
	for _, variant := range []string{"abliterated", "official", "huihui_lil"} {
		for i := range catalog.Bundles {
			b := &catalog.Bundles[i]
			if b.ID != "flash-next" {
				continue
			}
			opts := map[string]string{"MODEL_VARIANT": variant}
			b.Bindings["flash-next"] = Deployment{RuntimeOptions: &opts}
		}
		var err error
		catalog, err = ValidateCatalog(catalog)
		if err != nil {
			t.Fatal(err)
		}
		component, _ := catalog.ResolveComponent("flash-next", "flash-next")
		bundle, _ := catalog.Bundle("flash-next")
		repo, _, _ := QwenQADCheckpoint(variant)
		if component.Model != repo || bundle.ModelID != repo {
			t.Fatalf("inconsistent model identity: %s %s", component.Model, bundle.ModelID)
		}
		after, _ := catalog.Bundle("flash-next-tp2")
		if !reflect.DeepEqual(before, after) {
			t.Fatal("TP2 changed")
		}
	}
}

func TestQADPreparationUsesPinnedVariantWithoutStartingGPU(t *testing.T) {
	dir := t.TempDir()
	script := "#!/bin/sh\ncase \"$*\" in *config) cat;; run*) cat > \"$QAD_TEST_PAYLOAD\"; printf '%s\\n' \"$@\" > \"$QAD_TEST_ARGS\";; esac\n"
	if err := os.WriteFile(filepath.Join(dir, "docker"), []byte(script), 0700); err != nil {
		t.Fatal(err)
	}
	argsPath := filepath.Join(dir, "args")
	t.Setenv("QAD_TEST_ARGS", argsPath)
	t.Setenv("QAD_TEST_PAYLOAD", filepath.Join(dir, "payload"))
	t.Setenv("PATH", dir+":"+os.Getenv("PATH"))
	c, _ := NewController()
	c.ConfigurePaths(dir, filepath.Join(dir, "cache"))
	component, _ := c.Catalog().Component("flash-next")
	if err := c.PrepareModel(context.Background(), component, "abliterated", "model", "test-secret-token"); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(argsPath)
	if err != nil {
		t.Fatal(err)
	}
	args := string(raw)
	payload, err := os.ReadFile(filepath.Join(dir, "payload"))
	if err != nil {
		t.Fatal(err)
	}
	var request struct {
		Items []modelAsset `json:"items"`
		Token string       `json:"token"`
	}
	if err = json.Unmarshal(payload, &request); err != nil {
		t.Fatal(err)
	}
	if len(request.Items) != 1 || request.Items[0].Repo != QwenQADAbliterated || request.Items[0].Revision != "93a1b466ce773185f21a49d1649b7933ce0fc910" || request.Token != "test-secret-token" {
		t.Fatalf("invalid preparation request")
	}
	for _, want := range []string{filepath.Join(dir, "cache") + ":/hf"} {
		if !strings.Contains(args, want) {
			t.Fatalf("missing %s", want)
		}
	}
	if strings.Contains(args, "test-secret-token") || strings.Contains(args, "--gpus") {
		t.Fatal("credential exposed or GPU requested")
	}
}

func TestQADNoMTPKeepsFullContextAndDisablesDraftShortlist(t *testing.T) {
	for _, mode := range []string{"0", "3", "bad"} {
		data, err := composeAsset("compose.flash-next.yaml")
		if err != nil {
			t.Fatal(err)
		}
		var recipe map[string]any
		if err = yaml.Unmarshal(data, &recipe); err != nil {
			t.Fatal(err)
		}
		service := recipe["services"].(map[string]any)["runtime"].(map[string]any)
		component := Component{RuntimeOptions: map[string]string{"MTP_TOKENS": mode, "MODEL_VARIANT": "abliterated"}}
		err = applyQwenQADCheckpoint(service, component)
		if mode == "bad" {
			if err == nil {
				t.Fatal("invalid MTP option accepted")
			}
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		cmd := service["command"].([]any)
		spec := false
		full := false
		for i, arg := range cmd {
			if arg == "--speculative-algorithm" {
				spec = true
			}
			if arg == "--context-length" {
				full = cmd[i+1] == "1048576"
			}
		}
		if spec != (mode == "3") || !full {
			t.Fatal("MTP mode changed context or failed to remove draft")
		}
		if mode == "0" && service["environment"].(map[string]any)["SPARKTALK_FLASH_NEXT_DRAFT_VOCAB"] != "off" {
			t.Fatal("draft shortlist still enabled")
		}
		if mode == "0" && service["environment"].(map[string]any)["SGLANG_QAD_B12X_GDN"] != "0" {
			t.Fatal("non-speculative GDN must use native implementation")
		}

	}
}

func TestQwenQADLegacyLocalNameMigration(t *testing.T) {
	c := Component{ComposeAsset: "compose.flash-next.yaml", Model: "edp1096/Huihui-LIL-Qwen3.8-Flash-Next-abliterated-NVFP4"}
	if got := c.qwenQADModel().Model; got != QwenQADHuihuiLIL {
		t.Fatalf("legacy local name resolved to %q instead of %q", got, QwenQADHuihuiLIL)
	}
	c.ComposeAsset = "compose.flash-next-tp2.yaml"
	if got := c.qwenQADModel().Model; got != c.Model {
		t.Fatalf("TP2 model changed: %q", got)
	}
}

func TestRadixArkUsesNativeQuantizationAndOfflineCache(t *testing.T) {
	catalog, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	x, ok := catalog.ResolveComponent("flash-next-radixark", "flash-next-radixark")
	if !ok {
		t.Fatal("missing RadixArk")
	}
	assets := componentModelAssets(x)
	if len(assets) != 1 || !assets[0].LocalOnly || assets[0].Repo != QwenRadixArk {
		t.Fatalf("downloadable or wrong cache: %+v", assets)
	}
	data, _ := composeAsset(x.ComposeAsset)
	var recipe map[string]any
	if err := yaml.Unmarshal(data, &recipe); err != nil {
		t.Fatal(err)
	}
	service := recipe["services"].(map[string]any)["runtime"].(map[string]any)
	if err := applyQwenQADCheckpoint(service, x); err != nil {
		t.Fatal(err)
	}
	flags := map[string]string{}
	command := service["command"].([]any)
	for i, arg := range command {
		if key, ok := arg.(string); ok && strings.HasPrefix(key, "--") && i+1 < len(command) {
			flags[key], _ = command[i+1].(string)
		}
	}
	for key, want := range map[string]string{"--model-path": "/hf/" + QwenRadixArk, "--served-model-name": QwenRadixArk, "--mem-fraction-static": "0.92", "--quantization": "modelopt_fp4", "--speculative-draft-model-quantization": "modelopt_fp4", "--context-length": "1048576", "--max-total-tokens": "1048576", "--speculative-num-steps": "3"} {
		if flags[key] != want {
			t.Fatalf("%s: %q != %q", key, flags[key], want)
		}
	}
	if service["environment"].(map[string]any)["SGLANG_QAD_B12X_GDN"] != "0" {
		t.Fatal("QAD-only GDN used for RadixArk")
	}
	if !strings.Contains(fmt.Sprint(service["volumes"]), "radixark_launch.py") {
		t.Fatal("missing indexed draft launch")
	}
}
