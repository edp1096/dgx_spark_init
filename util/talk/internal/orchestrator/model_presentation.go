package orchestrator

import "fmt"

// Human names are independent of API IDs, repositories and runtime filenames.
// This registry also recognizes shipped names without overwriting user names.
type runtimeName struct {
	name     string
	previous []string
}

var runtimeNames = map[string]runtimeName{
	"ornith35":            {"Ornith 1.5 35B", []string{"Huihui Ornith 1.5 35B"}},
	"qwen38fn_exl3":       {"Qwen 3.8 Flash-Next EXL3 3bit", []string{"Qwen 3.8 Flash-Next EXL3", "Qwen3.8 Flash-Next EXL3", "Huihui Qwen3.8 EXL3", "Huihui Qwen3.8 Native EXL3", "Qwen3.8 EXL3", "Qwen3.8 Native EXL3", "Flash-Next EXL3", "Flash-Next EXL3 세트"}},
	"flash-next":          {"Qwen 3.8 Flash-Next QAD", []string{"Flash-Next", "Flash-Next 세트", "Qwen3.8 Flash-Next", "Qwen3.8 Flash-Next QAD", "Huihui-RadixArk Qwen3.8 Flash-Next"}},
	"qwen38fn_exl3_q4":    {"Qwen 3.8 Flash-Next EXL3 4bit", nil},
	"flash-next-radixark": {"Qwen 3.8 Flash-Next NVFP4", nil},
	"flash-next-tp2":      {"Qwen 3.8 Flash-Next TP2", []string{"Qwen3.8 Flash-Next TP2", "Huihui-RadixArk Qwen3.8 Flash-Next TP2"}},
	"gemma26":             {"Gemma 4 26B", []string{"Huihui Gemma 4 26B"}},
	"gemma31":             {"Gemma 4 31B", []string{"Gemma", "Gemma 31B", "Gemma4 31B"}},
	"gemma":               {"Gemma 4 31B", []string{"Gemma", "Gemma 세트", "Gemma 31B", "Gemma4 31B"}},
	"glm53":               {"GLM 5.3 Flash", []string{"GLM 5.3 Flash NVFP4"}},
	"glm53-worker-extra":  {"GLM 5.3 Flash", []string{"GLM 5.3 Flash NVFP4"}},
	"ds4fve":              {"DeepSeek V4 Flash Vision Exp", []string{"DeepSeek V4 Flash VE", "DeepSeek V4 Flash Vision Exp + 워커 Extra·ASR"}},
	"ds41":                {"DeepSeek V4.1 Flash", nil},
	"flux2":               {"Qwen Image 2.1", nil},
	"dreamlite":           {"DreamLite Mobile (시험)", nil},
	"nemotron-asr":        {"Nemotron 3.5 ASR", []string{"Nemotron ASR", "Nemotron ASR Q5_K"}},
	"qwen3-tts":           {"Qwen3-TTS 0.6B", []string{"Qwen3-TTS 0.6B Q8"}},
	"extra-media":         {"Extra Media", nil}, "extra-ssh": {"Extra SSH", nil},
	"extra-collector": {"Extra Collector", nil}, "extra-documents": {"Extra Documents", nil},
}

func RuntimeDisplayName(id, name string) string {
	entry, ok := runtimeNames[id]
	if !ok {
		return name
	}
	if name == "" || name == entry.name {
		return entry.name
	}
	for _, previous := range entry.previous {
		if name == previous {
			return entry.name
		}
	}
	return name
}

type ModelWeightOption struct {
	ID           string   `json:"id"`
	Label        string   `json:"label"`
	Format       string   `json:"format"`
	ModelID      string   `json:"model_id,omitempty"`
	Repositories []string `json:"repositories,omitempty"`
}
type ModelPresentation struct {
	SelectedVariant string              `json:"selected_variant"`
	Weights         []ModelWeightOption `json:"weights"`
}

// Preparation and all UI weight controls consume the same supported choices.
// Fixed recipes expose their actual choice, never a fictitious original option.
func modelPresentation(c Component) *ModelPresentation {
	var weights []ModelWeightOption
	selected := "official"
	switch {
	case c.ID == "flash-next-radixark":
		selected = "radixark"
		weights = []ModelWeightOption{{ID: selected, Label: "Abliterated", Format: "NVFP4", ModelID: QwenRadixArk, Repositories: []string{QwenRadixArk}}}
	case c.ComposeAsset == "compose.flash-next.yaml":
		selected = c.qwenQADVariant()
		weights = []ModelWeightOption{
			{ID: "official", Label: "원본", Format: "NVFP4 QAD", ModelID: QwenQADOfficial, Repositories: []string{QwenQADOfficial}},
			{ID: "abliterated", Label: "Abliterated", Format: "NVFP4", ModelID: QwenQADAbliterated, Repositories: []string{QwenQADAbliterated}},
			{ID: "huihui_lil", Label: "Abliterated · QAD", Format: "NVFP4 QAD", ModelID: QwenQADHuihuiLIL, Repositories: []string{QwenQADHuihuiLIL}},
		}
	case c.Controller == "glm53-cluster":
		selected = c.RuntimeOptions["MODEL_VARIANT"]
		if selected == "" {
			selected = "official"
		}
		weights = []ModelWeightOption{{ID: "official", Label: "원본", Format: "NVFP4", Repositories: []string{"nvidia/GLM-5.3-Flash-NVFP4"}}, {ID: "abliterated", Label: "Abliterated", Format: "NVFP4", Repositories: []string{"edp1096/Huihui-GLM-5.3-Flash-abliterated-NVFP4"}}}
	case c.Controller == "dspark-cluster":
		selected = c.RuntimeOptions["MODEL_VARIANT"]
		if selected == "" {
			selected = "official"
		}
		weights = []ModelWeightOption{{ID: "official", Label: "원본", Repositories: []string{"deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"}}, {ID: "abliterated", Label: "Abliterated", Repositories: []string{"drowzeys/keys-DeepSeekV4Flash-Vision-EXP-ablit"}}}
	case c.Controller == "ds41-cluster":
		weights = []ModelWeightOption{{ID: "official", Label: "원본", Format: "SSD expert streaming", Repositories: []string{"deepseek-ai/DeepSeek-V4.1-Flash"}}}
	case c.Controller == "qwen38-cluster":
		selected = "abliterated"
		weights = []ModelWeightOption{{ID: selected, Label: "Abliterated", Format: "NVFP4", ModelID: c.Model, Repositories: []string{"edp1096/Huihui-RadixArk-Qwen3.8-Flash-Next-abliterated-NVFP4"}}}
	default:
		label, format := "원본", ""
		switch c.ComposeAsset {
		case "compose.qwen38fn_exl3_q4.yaml":
			selected, label, format = "abliterated", "Abliterated", "EXL3 4bit HQ h6 ng6 · VeloGB10"
		case "compose.qwen38fn_exl3.yaml":
			selected, label, format = "abliterated", "Abliterated", "EXL3 3bit HQ h6 ng6"
		case "compose.ornith35.yaml", "compose.gemma26.yaml", "compose.gemma31.yaml":
			selected, label, format = "abliterated", "Abliterated", "NVFP4"
		case "compose.qwen-image21.yaml":
			label, format = "Uncensored", "NVFP4 · W4A8 · BF16 VAE"
		case "compose.nemotron-asr.yaml":
			format = "Q5_K · 화자 구분 Q8_0"
		case "compose.qwen3-tts.yaml":
			format = "Q8_0 본체 · Q8_0 오디오 codec"
		case "compose.extra-embedding.yaml":
			format = "BF16 · 텍스트 270M · 768차원"
		case "compose.dreamlite.yaml":
			format = ""
		case "compose.flux2.yaml":
			format = "NVFP4"
		default:
			return nil
		}
		option := ModelWeightOption{ID: selected, Label: label, Format: format, ModelID: c.Model}
		for _, asset := range componentModelAssets(c) {
			duplicate := false
			for _, repo := range option.Repositories {
				duplicate = duplicate || repo == asset.Repo
			}
			if !duplicate {
				option.Repositories = append(option.Repositories, asset.Repo)
			}
		}
		weights = []ModelWeightOption{option}
	}
	return &ModelPresentation{SelectedVariant: selected, Weights: weights}
}

func ValidateModelPreparationVariant(c Component, variant string) error {
	info := modelPresentation(c)
	if info == nil {
		return fmt.Errorf("%s: model preparation is not supported", c.Name)
	}
	for _, option := range info.Weights {
		if option.ID == variant {
			return nil
		}
	}
	return fmt.Errorf("%s: unsupported weight variant %q", c.Name, variant)
}
