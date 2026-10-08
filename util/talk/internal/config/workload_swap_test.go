package config

import (
	"sparktalk/internal/orchestrator"
	"testing"
)

func TestQADWorkloadSwapMigrationPreservesMembershipAndOtherSets(t *testing.T) {
	cat, _ := orchestrator.LoadCatalog()
	for i := range cat.Bundles {
		cat.Bundles[i].WorkloadSwap = false
	}
	cfg := Config{Runtime: RuntimeConfig{BuiltinRevision: 19, Catalog: &cat}}
	cfg.Normalize()
	for _, b := range cfg.Runtime.Catalog.Bundles {
		if b.WorkloadSwap != (b.ID == "flash-next" || b.ID == "flash-next-radixark") {
			t.Fatalf("unexpected policy for %s", b.ID)
		}
		if b.ID == "flash-next" {
			found := false
			for _, id := range b.Components {
				found = found || id == "nemotron-asr"
			}
			if !found {
				t.Fatal("ASR removed instead of scheduled")
			}
		}
	}
	for i := range cfg.Runtime.Catalog.Bundles {
		cfg.Runtime.Catalog.Bundles[i].WorkloadSwap = false
	}
	cfg.Normalize()
	for _, b := range cfg.Runtime.Catalog.Bundles {
		if b.WorkloadSwap {
			t.Fatal("migration overwrote subsequent choice")
		}
	}
}

func TestPhasedFluxBudgetMigrationPreservesCustomValues(t *testing.T) {
	for _, value := range []float64{12, 13, 15} {
		cat, _ := orchestrator.LoadCatalog()
		for i := range cat.Components {
			if cat.Components[i].ID == "flux2" {
				cat.Components[i].ComposeAsset = "compose.flux2.yaml"
				cat.Components[i].Model = "custom-legacy-checkpoint"
				cat.Components[i].MemoryGiB = value
			}
		}
		cfg := Config{Runtime: RuntimeConfig{BuiltinRevision: 20, Catalog: &cat, MemoryReserveGiB: 1.5}}
		cfg.Normalize()
		want := value
		if value == 12 || value == 13 {
			want = 5.25
		}
		for _, x := range cfg.Runtime.Catalog.Components {
			if x.ID == "flux2" && x.MemoryGiB != want {
				t.Fatalf("budget=%v want=%v", x.MemoryGiB, want)
			}
		}
		if cfg.Runtime.MemoryReserveGiB != 1.5 {
			t.Fatal("reserve changed")
		}
	}
}
