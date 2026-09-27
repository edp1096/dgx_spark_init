package config

import "testing"

func TestClusterSwitchRoutesMagpieWithoutChangingVoice(t *testing.T) {
	cfg := Config{}
	cfg.Normalize()
	cfg.TTS.Enabled = true
	for _, id := range []string{"ds4fve", "ds41"} {
		voice := cfg.TTS.Voice
		cfg.ApplyManagedBundle(id)
		if cfg.TTS.Endpoint != "http://192.168.100.60:8692" || cfg.TTS.Model != "magpietts" || !cfg.TTS.Enabled || cfg.TTS.Voice != voice {
			t.Fatalf("%s: wrong TTS profile: %+v", id, cfg.TTS)
		}
	}
	cfg.ApplyManagedBundle("glm53-worker-extra")
	if cfg.TTS.Enabled {
		t.Fatal("GLM must not enable TTS")
	}
	cfg.ASR.Enabled = true
	cfg.ApplyManagedBundle("flash-next")
	if cfg.TTS.Enabled || !cfg.ASR.Enabled {
		t.Fatal("QAD must enable ASR and keep TTS disabled")
	}
}
