package orchestrator

import "testing"

func TestResidentEmbeddingBootstrapsBeforeTTSOnSameHost(t *testing.T) {
	c, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	b, _ := c.Bundle("qwen38fn_exl3")
	order := c.startupOrder(c.StartBundleMembers(b))
	pos := map[string]int{}
	for i, id := range order {
		pos[id] = i
	}
	if !(pos["qwim-mmh3"] < pos["qwen38fn_exl3"] &&
		pos["qwen38fn_exl3"] < pos["nemotron-asr"] &&
		pos["nemotron-asr"] < pos["extra-embedding"] &&
		pos["extra-embedding"] < pos["qwen3-tts"]) {
		t.Fatal("resident CUDA startup order", order)
	}
	// A service on another host cannot consume this TTS process's headroom.
	binding := b.Bindings["extra-embedding"]
	worker := "worker"
	binding.Host = &worker
	b.Bindings["extra-embedding"] = binding
	c.byBundle[b.ID] = b
	order = c.startupOrder(c.StartBundleMembers(b))
	for i, id := range order {
		pos[id] = i
	}
	if pos["qwen3-tts"] > pos["extra-embedding"] {
		t.Fatal("unrelated host changed normal startup order", order)
	}
}
