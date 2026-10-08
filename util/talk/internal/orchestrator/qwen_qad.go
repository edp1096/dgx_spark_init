package orchestrator

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

const QwenQADOfficial = "local-inference-lab/Qwen3.8-Flash-Next-NVFP4"
const QwenQADAbliterated = "huginnfork/Qwen3.8-Flash-Next-NVFP4-Abliterated"
const QwenQADHuihuiLIL = "edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD"
const QwenRadixArk = "edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4"

// QwenQADCheckpoint is scoped to the TP1 QAD recipe, never the TP2 model.
func QwenQADCheckpoint(variant string) (repo, revision string, err error) {
	switch variant {
	case "", "official":
		return QwenQADOfficial, "7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd", nil
	case "huihui_lil":
		return QwenQADHuihuiLIL, qwenQADHuihuiRelease.Revision, nil
	case "abliterated":
		return QwenQADAbliterated, "93a1b466ce773185f21a49d1649b7933ce0fc910", nil
	case "radixark":
		return QwenRadixArk, "40f09f531da577a4fcfbd6b368e7b6ebacde403e", nil
	default:
		return "", "", fmt.Errorf("invalid QAD MODEL_VARIANT: %s", variant)
	}
}

func (c Component) qwenQADVariant() string {
	if v := c.RuntimeOptions["MODEL_VARIANT"]; v != "" {
		return v
	}
	if c.Model == QwenQADHuihuiLIL || c.Model == "edp1096/Huihui-LIL-Qwen3.8-Flash-Next-abliterated-NVFP4" {
		return "huihui_lil"
	}
	if c.Model == QwenQADAbliterated {
		return "abliterated"
	}
	if c.Model == QwenRadixArk {
		return "radixark"
	}
	return "official"
}

func (c Component) qwenQADModel() Component {
	if c.ComposeAsset == "compose.flash-next.yaml" {
		if repo, _, err := QwenQADCheckpoint(c.qwenQADVariant()); err == nil {
			c.Model = repo
		}
	}
	return c
}

func applyQwenQADCheckpoint(service map[string]any, component Component) error {
	repo, revision, err := QwenQADCheckpoint(component.qwenQADVariant())
	if err != nil {
		return err
	}
	mtp := component.RuntimeOptions["MTP_TOKENS"]
	if mtp != "" && mtp != "0" && mtp != "3" {
		return fmt.Errorf("QAD MTP_TOKENS must be 0 or 3")
	}
	command, ok := service["command"].([]any)
	if !ok {
		return fmt.Errorf("QAD command is missing")
	}
	values := map[string]string{
		"--model-path":        "/hf/hub/models--" + strings.ReplaceAll(repo, "/", "--") + "/snapshots/" + revision,
		"--served-model-name": repo,
	}
	// A saved TP1 profile may choose a smaller real context/KV pool without
	// changing its checkpoint, MTP, or the separate TP2/EXL3 profiles.
	if value := component.RuntimeOptions["MAX_MODEL_LEN"]; value != "" {
		length, err := strconv.Atoi(value)
		if err != nil || length < 65536 || length > 1048576 || length%64 != 0 {
			return fmt.Errorf("QAD MAX_MODEL_LEN must be a multiple of 64 between 65536 and 1048576")
		}
		values["--context-length"] = value
		values["--max-total-tokens"] = value
	}
	if component.qwenQADVariant() == "huihui_lil" || component.qwenQADVariant() == "radixark" {
		values["--model-path"] = "/hf/" + repo
	}
	if component.qwenQADVariant() == "radixark" {
		values["--quantization"] = "modelopt_fp4"
		// LLM-only profile: retain about 9.4 GiB of runtime slack instead
		// of the auxiliary-sharing recipe's 16.5 GiB at 0.86.
		values["--mem-fraction-static"] = "0.92"
		values["--speculative-draft-model-quantization"] = "modelopt_fp4"
		service["environment"].(map[string]any)["SGLANG_QAD_B12X_GDN"] = "0"
		service["entrypoint"] = []any{"python3", "/opt/radixark/launch.py"}
		volumes := service["volumes"].([]any)
		service["volumes"] = append(volumes, "${SPARKTALK_BUILD_DIR}/radixark_launch.py:/opt/radixark/launch.py:ro", "${SPARKTALK_BUILD_DIR}/radixark_launch.py:/opt/qad-tp1/qad_loader.py:ro")
	}
	for flag, value := range values {
		found := false
		for i, arg := range command {
			if arg == flag && i+1 < len(command) {
				command[i+1] = value
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("QAD command missing %s", flag)
		}
	}
	if mtp == "0" {
		filtered := make([]any, 0, len(command))
		for i := 0; i < len(command); i++ {
			arg, _ := command[i].(string)
			if strings.HasPrefix(arg, "--speculative-") {
				i++
				continue
			}
			filtered = append(filtered, command[i])
		}
		service["command"] = filtered
		service["environment"].(map[string]any)["SPARKTALK_FLASH_NEXT_DRAFT_VOCAB"] = "off"
		// Non-speculative long-context decoding uses the validated native GDN path.
		service["environment"].(map[string]any)["SGLANG_QAD_B12X_GDN"] = "0"
	}
	return nil
}

// Release data is embedded in Talk; preparation never reads compose_yaml.
var qwenQADHuihuiRelease = func() modelAsset {
	raw, err := assets.ReadFile("assets/model-prepare/qad-huihui-step5500.json")
	if err != nil {
		panic(err)
	}
	var release modelAsset
	if err = json.Unmarshal(raw, &release); err != nil {
		panic(err)
	}
	if release.Repo != QwenQADHuihuiLIL || len(release.Revision) != 40 || len(release.SHA256) == 0 {
		panic("invalid Huihui QAD release")
	}
	return release
}()

var qwenRadixArkRelease = func() modelAsset {
	raw, err := assets.ReadFile("assets/model-prepare/radixark-local.json")
	if err != nil {
		panic(err)
	}
	var release modelAsset
	if err := json.Unmarshal(raw, &release); err != nil {
		panic(err)
	}
	if release.Repo != QwenRadixArk || len(release.Revision) != 40 || !release.LocalOnly || len(release.SHA256) == 0 {
		panic("invalid local RadixArk identity")
	}
	return release
}()
