package config

import "testing"

func TestClusterSwitchRoutesMagpieWithoutChangingVoice(t *testing.T) {
	cfg := Config{}
	cfg.Normalize()
	cfg.TTS.Enabled = true
	for _, id := range []string{"glm53-worker-extra", "ds4fve", "ds41"} {
		voice := cfg.TTS.Voice
		cfg.ApplyManagedBundle(id)
		if cfg.TTS.Endpoint != "http://192.168.100.60:8692" || cfg.TTS.Model != "magpietts" || !cfg.TTS.Enabled || cfg.TTS.Voice != voice {
			t.Fatalf("%s: wrong TTS profile: %+v", id, cfg.TTS)
		}
	}
	cfg.ApplyManagedBundle("flash-next")
	if cfg.TTS.Enabled || cfg.ASR.Enabled {
		t.Fatal("QAD must disable speech services absent from its default set")
	}
}
