package orchestrator

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestModelPresentationSharesNamesAndPreservesAPIIdentity(t *testing.T) {
	cat, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"ornith35", "qwen38fn_exl3", "flash-next", "flash-next-tp2", "gemma26", "ds4fve", "ds41"} {
		component, _ := cat.Component(id)
		bundle, _ := cat.Bundle(id)
		if component.Name != bundle.Name {
			t.Fatalf("%s: component/set names differ", id)
		}
		if component.ModelPresentation == nil {
			t.Fatalf("%s: missing weight metadata", id)
		}
	}
	native, _ := cat.Component("qwen38fn_exl3")
	if native.Model != "qwen38fn_exl3" || native.Container != "sparktalk-qwen38fn_exl3" {
		t.Fatal("presentation changed runtime identity")
	}
	if native.ModelPresentation.SelectedVariant != "abliterated" || ValidateModelPreparationVariant(native, "official") == nil {
		t.Fatal("EXL3 offers unsupported original weights")
	}
	bytes, err := yaml.Marshal(native)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(bytes), "model_presentation") {
		t.Fatal("computed metadata persisted into runtime YAML")
	}
	if RuntimeDisplayName(native.ID, "내 EXL3") != "내 EXL3" || RuntimeDisplayName(native.ID, "Huihui Qwen3.8 Native EXL3") != native.Name {
		t.Fatal("display-name migration lost user names or shipped aliases")
	}
}

func TestModelPresentationUsesActualPreparationRepositoriesAndBoundVariant(t *testing.T) {
	cat, _ := LoadCatalog()
	for _, id := range []string{"qwen38fn_exl3", "flash-next", "gemma26", "gemma31", "ornith35", "flux2", "nemotron-asr", "qwen3-tts"} {
		component, _ := cat.Component(id)
		for _, option := range component.ModelPresentation.Weights {
			component.RuntimeOptions = map[string]string{"MODEL_VARIANT": option.ID}
			var repos []string
			for _, asset := range componentModelAssets(component) {
				if !strings.Contains(strings.Join(repos, "\n"), asset.Repo) {
					repos = append(repos, asset.Repo)
				}
			}
			if !reflect.DeepEqual(option.Repositories, repos) {
				t.Fatalf("%s/%s: display/preparation sources differ", id, option.ID)
			}
		}
	}
	for i := range cat.Bundles {
		if cat.Bundles[i].ID == "flash-next" {
			variant := "huihui_lil"
			options := map[string]string{"MODEL_VARIANT": variant}
			binding := cat.Bundles[i].Bindings["flash-next"]
			binding.RuntimeOptions = &options
			cat.Bundles[i].Bindings["flash-next"] = binding
		}
	}
	cat, err := ValidateCatalog(cat)
	if err != nil {
		t.Fatal(err)
	}
	component, _ := cat.ResolveComponent("flash-next", "flash-next")
	if component.ModelPresentation.SelectedVariant != "huihui_lil" {
		t.Fatal("resolved presentation ignored bundle weights")
	}
}
