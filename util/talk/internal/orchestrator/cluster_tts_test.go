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
	local, _ := c.ResolveComponent("flash-next", "magpie-tts")
	if local.Host == "worker" || local.StartAfterLLM {
		t.Fatal("cluster binding leaked into Qwen")
	}
}
