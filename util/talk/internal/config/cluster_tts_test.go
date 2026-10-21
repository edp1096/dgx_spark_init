package config

import "testing"

func TestClusterSwitchRoutesQwenTTSWithoutChangingVoice(t *testing.T) {
	cfg := Config{}
	cfg.Normalize()
	cfg.TTS.Enabled = true
	for _, id := range []string{"ds41", "glm53-worker-extra"} {
		voice := cfg.TTS.Voice
		cfg.ApplyManagedBundle(id)
		want := "http://192.168.100.60:8692"
		if id == "ds41" {
			want = "http://127.0.0.1:8692"
		}
		if cfg.TTS.Endpoint != want || cfg.TTS.Model != "qwen3-tts-0.6b-q8" || !cfg.TTS.Enabled || cfg.TTS.Voice != voice {
			t.Fatalf("%s: wrong TTS profile: %+v", id, cfg.TTS)
		}
	}
	cfg.TTS.Enabled = false
	cfg.ASR.Enabled = true
	cfg.ApplyManagedBundle("flash-next")
	if cfg.TTS.Enabled || !cfg.ASR.Enabled {
		t.Fatal("set switch must preserve user TTS toggle")
	}
}
