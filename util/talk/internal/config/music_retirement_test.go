package config

import (
	"reflect"
	"sparktalk/internal/orchestrator"
	"testing"
)

func TestRetiredMusicRemovedWithoutChangingModelSet(t *testing.T) {
	cat, _ := orchestrator.LoadCatalog()
	ace, _ := cat.Component("extra-music-ace")
	ace.Endpoint = "http://custom-worker:8727"
	cat.Components = append(cat.Components, orchestrator.Component{ID: "extra-music-heart", ComposeAsset: "compose.extra-music-heart.yaml"})
	cat.Bundles[0].Components = append(cat.Bundles[0].Components, "extra-music-heart")
	if cat.Bundles[0].Bindings == nil {
		cat.Bundles[0].Bindings = map[string]orchestrator.Deployment{}
	}
	cat.Bundles[0].Bindings["extra-music-heart"] = orchestrator.Deployment{}
	for i, x := range cat.Components {
		if x.ID == ace.ID {
			cat.Components[i] = ace
		}
	}
	cfg := Config{Runtime: RuntimeConfig{Catalog: &cat, Bundle: cat.Bundles[0].ID, ActiveBundle: cat.Bundles[0].ID, AutoStart: true}}
	count := len(cat.Bundles)
	active := cfg.Runtime.ActiveBundle
	cfg.removeRetiredMusicSupport()
	cfg.ensureShortsSupport()
	cfg.removeRetiredMusicSupport()
	if len(cat.Bundles) != count || cfg.Runtime.ActiveBundle != active || cfg.Runtime.Bundle != active || !cfg.Runtime.AutoStart {
		t.Fatal("changed model set")
	}
	found := false
	for _, x := range cat.Components {
		if x.ID == "extra-music-heart" {
			t.Fatal("retired component reappeared")
		}
		if x.ID == ace.ID {
			found = true
			if !reflect.DeepEqual(x, ace) {
				t.Fatal("changed ACE deployment")
			}
		}
	}
	if !found {
		t.Fatal("removed ACE")
	}
	for _, b := range cat.Bundles {
		for _, id := range b.Components {
			if id == "extra-music-heart" {
				t.Fatal("stale membership")
			}
		}
		if _, ok := b.Bindings["extra-music-heart"]; ok {
			t.Fatal("stale binding")
		}
	}
	if cfg.SupportEnabled("music-heart") || cfg.SupportEndpoint("music-heart") != "" {
		t.Fatal("retired service still enabled")
	}
}
