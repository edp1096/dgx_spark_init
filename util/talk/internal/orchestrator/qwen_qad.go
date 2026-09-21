package orchestrator

import (
	"fmt"
	"strings"
)

const QwenQADOfficial = "local-inference-lab/Qwen3.8-Flash-Next-NVFP4"
const QwenQADAbliterated = "huginnfork/Qwen3.8-Flash-Next-NVFP4-Abliterated"
const QwenQADHuihuiLIL = "edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD"

// QwenQADCheckpoint is scoped to the TP1 QAD recipe, never the TP2 model.
func QwenQADCheckpoint(variant string) (repo, revision string, err error) {
	switch variant {
	case "", "official":
		return QwenQADOfficial, "7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd", nil
	case "huihui_lil":
		return QwenQADHuihuiLIL, "local", nil
	case "abliterated":
		return QwenQADAbliterated, "93a1b466ce773185f21a49d1649b7933ce0fc910", nil
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
	if component.qwenQADVariant() == "huihui_lil" {
		values["--model-path"] = "/hf/" + QwenQADHuihuiLIL
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
