package orchestrator

import (
	"context"
	"testing"
)

func TestExcludedAudioCannotStartThroughSharedServices(t *testing.T) {
	cat, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	c, err := NewControllerWithCatalog(cat)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	for _, bundle := range []string{"flash-next-fp4-local", "flash-next-tp2", "glm53-worker-extra", "ds41"} {
		b, _ := cat.Bundle(bundle)
		b.StartSupport = true
		b.Components = append(append([]string(nil), b.Components...), "extra-fish", "extra-music-ace")
		for _, id := range cat.StartBundleMembers(b).Components {
			if id == "extra-fish" || id == "extra-music-ace" {
				t.Fatal("disabled service included at startup", bundle, id)
			}
		}
		for _, id := range []string{"extra-fish", "extra-music-ace"} {
			if _, ok := cat.ResolveSupport(bundle, id); ok {
				t.Fatal(bundle, id, "still available")
			}
			for _, x := range cat.SupportComponents(bundle) {
				if x.ID == id {
					t.Fatal(bundle, id, "still advertised")
				}
			}
			if release, err := c.AcquireWorkload(context.Background(), bundle, id, 4); err == nil || release != nil {
				t.Fatal("disabled service admitted", bundle, id)
			}
			if err := c.ComponentActionWithReserve(id, "start", 4, bundle); err == nil {
				t.Fatal("manual start admitted", bundle, id)
			}
		}
	}
}
