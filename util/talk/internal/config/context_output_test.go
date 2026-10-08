package config

import "testing"

func TestAutomaticOutputFollowsContextSize(t *testing.T) {
	c := ContextConfig{OutputAuto: true, OutputReserve: 16384, SafetyMargin: 1024}
	// Include switching back from a large model and detection failure.
	for _, tc := range []struct{ window, want int }{
		{32768, 8192}, {65536, 16384}, {131072, 32768},
		{262144, 65536}, {524288, 65536}, {1048576, 65536},
		{32768, 8192}, {0, 16384}, {8192, 1792},
	} {
		if got := c.EffectiveOutputReserve(tc.window); got != tc.want {
			t.Errorf("window %d: got %d, want %d", tc.window, got, tc.want)
		}
	}
	c.OutputAuto = false
	c.OutputReserve = 12288
	if got := c.EffectiveOutputReserve(1048576); got != 12288 {
		t.Fatalf("manual budget changed: %d", got)
	}
}
