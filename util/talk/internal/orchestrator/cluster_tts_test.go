package orchestrator

import "testing"

func TestClusterMagpiePlacementAndStartup(t *testing.T) {
	c, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"glm53-worker-extra", "ds4fve", "ds41"} {
		b, _ := c.Bundle(id)
		tts, ok := c.ResolveComponent(id, "magpie-tts")
		if !ok || tts.Host != "worker" || tts.Endpoint != "http://192.168.100.60:8692" || tts.MemoryGiB != 1.6 {
			t.Fatalf("%s: invalid worker TTS binding: %+v", id, tts)
		}
		order := c.startupOrder(c.StartBundleMembers(b))
		llmIndex, ttsIndex := -1, -1
		for i, member := range order {
			x, _ := c.ResolveComponent(id, member)
			if x.Role == "llm" {
				llmIndex = i
			}
			if member == "magpie-tts" {
				ttsIndex = i
			}
		}
		if llmIndex < 0 || ttsIndex < 0 || (ttsIndex > llmIndex) != (id == "ds41") {
			t.Fatalf("%s: wrong startup order: %v", id, order)
		}
	}
	// QAD TP1 starts auxiliary models after the LLM to reserve its KV cache first.
	bundle, _ := c.Bundle("flash-next")
	order := c.startupOrder(c.StartBundleMembers(bundle))
	llmIndex := -1
	for i, id := range order {
		if id == "flash-next" {
			llmIndex = i
		}
	}
	if llmIndex < 0 {
		t.Fatal("Qwen missing from startup order")
	}
	for _, id := range []string{"magpie-tts", "nemotron-asr", "flux2"} {
		local, ok := c.ResolveComponent("flash-next", id)
		if !ok || local.Host == "worker" || !local.StartAfterLLM {
			t.Fatalf("invalid QAD auxiliary binding: %s %+v", id, local)
		}
		position := -1
		for i, member := range order {
			if member == id {
				position = i
			}
		}
		if position <= llmIndex {
			t.Fatalf("%s must start after Qwen: %v", id, order)
		}
	}
}
