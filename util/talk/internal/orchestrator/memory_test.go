package orchestrator

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestMemoryHeadroomAccountsForReturnedModelMemory(t *testing.T) {
	memory := SystemMemory{AvailableGiB: 18, FreeGiB: 2}
	plan := memoryPlan{NeededGiB: 60, FreedGiB: 55, RequiresCUDAStart: true}
	if err := validateMemoryHeadroom(memory, plan, 8); err != nil {
		t.Fatalf("returned model memory should make the switch safe: %v", err)
	}
}

func TestMemoryHeadroomRejectsInsufficientProjectedReserve(t *testing.T) {
	memory := SystemMemory{AvailableGiB: 20, FreeGiB: 10}
	plan := memoryPlan{NeededGiB: 15}
	err := validateMemoryHeadroom(memory, plan, 8)
	if err == nil || !strings.Contains(err.Error(), "통합메모리 부족 예상") {
		t.Fatalf("expected projected-memory failure, got %v", err)
	}
}

func TestMemoryHeadroomRejectsLowImmediateCUDAFreeMemory(t *testing.T) {
	memory := SystemMemory{AvailableGiB: 20, FreeGiB: 1.5}
	plan := memoryPlan{NeededGiB: 6.7, RequiresCUDAStart: true}
	err := validateMemoryHeadroom(memory, plan, 8)
	if err == nil || !strings.Contains(err.Error(), "CUDA 기동용 즉시 여유") {
		t.Fatalf("expected immediate-free failure, got %v", err)
	}
}

func TestHealthyLLMMemoryIsNotCountedTwice(t *testing.T) {
	component := Component{Role: "llm", MemoryGiB: 96}
	if remaining := healthyComponentRemainingMemory(component, 87.2); remaining != 0 {
		t.Fatalf("healthy LLM host allocations are already reflected in MemAvailable, got %.1f GiB", remaining)
	}
}

func TestHealthyLazyImageKeepsRemainingPeak(t *testing.T) {
	component := Component{Role: "image", MemoryGiB: 6.7}
	if remaining := healthyComponentRemainingMemory(component, .2); remaining != 6.5 {
		t.Fatalf("unexpected lazy image reserve: %.1f GiB", remaining)
	}
}

func TestCacheReclaimDoesNotMaskInsufficientCapacity(t *testing.T) {
	plan := memoryPlan{NeededGiB: 110, RequiresCUDAStart: true}
	var cold *cudaStartMemoryError
	if err := validateMemoryHeadroom(SystemMemory{AvailableGiB: 120, FreeGiB: 2}, plan, 8); !errors.As(err, &cold) {
		t.Fatalf("expected reclaimable free-memory error: %v", err)
	}
	if err := validateMemoryHeadroom(SystemMemory{AvailableGiB: 100, FreeGiB: 2}, plan, 8); err == nil || errors.As(err, &cold) {
		t.Fatalf("real capacity shortage must not trigger reclaim retry: %v", err)
	}
	catalog, _ := LoadCatalog()
	controller := newController(catalog)
	bundle, _ := catalog.Bundle("ds41")
	if attempted, err := controller.reclaimGLMStartupCache(context.Background(), bundle); attempted || err != nil {
		t.Fatal("non-GLM start must not reclaim", attempted, err)
	}
}

func TestComponentStartCannotBorrowOtherLLMMemory(t *testing.T) {
	plan := componentStartMemoryPlan(Component{Role: "image", MemoryGiB: 13}, 0)
	if plan.FreedGiB != 0 {
		t.Fatal("single start credited another model")
	}
	if err := validateMemoryHeadroom(SystemMemory{AvailableGiB: 11, FreeGiB: 5}, plan, 4); err == nil {
		t.Fatal("unsafe FLUX start accepted")
	}
}

func TestQADVariantsUseSameMemoryGuard(t *testing.T) {
	for _, variant := range []string{"official", "abliterated"} {
		component := Component{ComposeAsset: "compose.flash-next.yaml", Role: "llm", MemoryGiB: 100, RuntimeOptions: map[string]string{"MODEL_VARIANT": variant}}
		plan := componentStartMemoryPlan(component.qwenQADModel(), 0)
		plan.NeededGiB += 13 + 1.3 + 1.2
		if err := validateMemoryHeadroom(SystemMemory{AvailableGiB: 116.9, FreeGiB: 15}, plan, 4); err == nil {
			t.Fatalf("%s must use the same capacity check", variant)
		}
	}
}

func TestQADMemoryReservationTracksActualMTPProfile(t *testing.T) {
	for _, model := range []string{QwenQADOfficial, QwenQADAbliterated} {
		c := Component{ComposeAsset: "compose.flash-next.yaml", Model: model, MemoryGiB: 97, RuntimeOptions: map[string]string{"MTP_TOKENS": "0"}}
		if c.runtimeMemoryEstimate().MemoryGiB != 97 {
			t.Fatal("native no-MTP budget")
		}
		c.RuntimeOptions["MTP_TOKENS"] = "3"
		if c.runtimeMemoryEstimate().MemoryGiB != 100 {
			t.Fatal("MTP must reserve its extra allocations")
		}
		c.MemoryGiB = 108
		if c.runtimeMemoryEstimate().MemoryGiB != 108 {
			t.Fatal("larger user reservation lost")
		}
	}
	plan := memoryPlan{NeededGiB: 97 + 13 + 1.3 + 1.2, RequiresCUDAStart: true}
	if err := validateMemoryHeadroom(SystemMemory{AvailableGiB: 116.9, FreeGiB: 15}, plan, 4); err != nil {
		t.Fatal(err)
	}
}

func TestResidentMemoryDoesNotCreditReclaimableCheckpointCache(t *testing.T) {
	stat := []byte("anon 1073741824\nfile 1099511627776\ninactive_file 549755813888\n")
	if hostResidentMemoryGiB(stat) != 1 {
		t.Fatal("file cache must not be counted as memory released by stopping a service")
	}
}

func TestASRStartupBudgetDoesNotReserveLongTranscriptionWorkspace(t *testing.T) {
	component := Component{Role: "asr", MemoryGiB: 6, StartupMemoryGiB: 3.5}
	plan := componentStartMemoryPlan(component, 0)
	plan.NeededGiB += 100 + 13
	if err := validateMemoryHeadroom(SystemMemory{AvailableGiB: 118.4, FreeGiB: 114}, plan, 1.5); err != nil {
		t.Fatal(err)
	}
	if component.MemoryGiB != 6 {
		t.Fatal("processing peak was reduced")
	}
	if err := validateMemoryHeadroom(SystemMemory{AvailableGiB: 116, FreeGiB: 114}, plan, 1.5); err == nil {
		t.Fatal("actual startup shortage must still fail")
	}
}

func TestStartupStagingPeakIsNotClampedToServingBudget(t *testing.T) {
	component := Component{Role: "llm", MemoryGiB: 10, StartupMemoryGiB: 14}
	plan := componentStartMemoryPlan(component, 0)
	// The steady allocation fits, but the temporary loading peak does not.
	if err := validateMemoryHeadroom(SystemMemory{AvailableGiB: 13, FreeGiB: 13}, plan, 1.5); err == nil {
		t.Fatal("startup staging peak must not be clipped to the serving budget")
	}
	if err := validateMemoryHeadroom(SystemMemory{AvailableGiB: 16, FreeGiB: 16}, plan, 1.5); err != nil {
		t.Fatal(err)
	}
}

func TestSequentialStartupPeaksDoNotAccumulate(t *testing.T) {
	var phases startupMemoryPhases
	phases.add(Component{MemoryGiB: 10, StartupMemoryGiB: 14}, 0)
	phases.add(Component{MemoryGiB: 6, StartupMemoryGiB: 9}, 0)
	// Both retained models (16) plus the larger temporary peak (4).
	// 23 would count two non-overlapping loading buffers simultaneously.
	if phases.needed() != 20 {
		t.Fatalf("unexpected phase budget: %v", phases.needed())
	}
	var restart startupMemoryPhases
	restart.add(Component{MemoryGiB: 10, StartupMemoryGiB: 14}, 12)
	if restart.needed() != 2 {
		t.Fatal("restart double-counted already resident allocation")
	}
	var lazy startupMemoryPhases
	lazy.add(Component{MemoryGiB: 6, StartupMemoryGiB: 3.5}, 0)
	if lazy.needed() != 3.5 {
		t.Fatal("startup reserved an unrequested workload")
	}
}

func TestTiledFluxAdmissionAtReportedHeadroom(t *testing.T) {
	memory := SystemMemory{AvailableGiB: 14.4, FreeGiB: 6.9}
	if err := validateMemoryHeadroom(memory, memoryPlan{NeededGiB: 13, RequiresCUDAStart: true}, 1.5); err == nil {
		t.Fatal("old budget should reproduce rejection")
	}
	if err := validateMemoryHeadroom(memory, memoryPlan{NeededGiB: 12, RequiresCUDAStart: true}, 1.5); err != nil {
		t.Fatal(err)
	}
	if err := validateMemoryHeadroom(SystemMemory{AvailableGiB: 13, FreeGiB: 6}, memoryPlan{NeededGiB: 12, RequiresCUDAStart: true}, 1.5); err == nil {
		t.Fatal("reserve must remain enforced")
	}
}

func TestPhasedFluxAtWarmQwenHeadroom(t *testing.T) {
	m := SystemMemory{AvailableGiB: 7, FreeGiB: 5.2}
	if err := validateMemoryHeadroom(m, memoryPlan{NeededGiB: 5.25, RequiresCUDAStart: true}, 1.5); err != nil {
		t.Fatal(err)
	}
	if err := validateMemoryHeadroom(m, memoryPlan{NeededGiB: 12, RequiresCUDAStart: true}, 1.5); err == nil {
		t.Fatal("old budget must fail")
	}
	if err := validateMemoryHeadroom(SystemMemory{AvailableGiB: 6.5, FreeGiB: 5}, memoryPlan{NeededGiB: 5.25, RequiresCUDAStart: true}, 1.5); err == nil {
		t.Fatal("low reserve must still fail")
	}
}

func TestHostResidentIncludesSharedAndKernelWithoutFileCache(t *testing.T) {
	stat := []byte("anon 1073741824\nshmem 536870912\nfile 1099511627776\nkernel 268435456\nslab_reclaimable 134217728\n")
	if got := hostResidentMemoryGiB(stat); got != 1.625 {
		t.Fatalf("host resident = %v", got)
	}
}

func TestCurrentHuihuiQADBudgetIncludesHostMemory(t *testing.T) {
	c := Component{ComposeAsset: "compose.flash-next.yaml", Model: QwenQADHuihuiLIL, MemoryGiB: 100, RuntimeOptions: map[string]string{"MTP_TOKENS": "3"}}
	if got := c.runtimeMemoryEstimate().MemoryGiB; got != 108 {
		t.Fatalf("budget=%v", got)
	}
	c.MemoryGiB = 112
	if got := c.runtimeMemoryEstimate().MemoryGiB; got != 112 {
		t.Fatalf("user budget lost: %v", got)
	}
}

func TestObservedBudgetIncludesCoreHostAndRetainedAuxiliaries(t *testing.T) {
	cat, _ := LoadCatalog()
	b, _ := cat.Bundle("flash-next")
	core, _ := cat.ResolveComponent(b.ID, "flash-next")
	image, _ := cat.ResolveComponent(b.ID, "flux2")
	speech, _ := cat.ResolveComponent(b.ID, "nemotron-asr")
	states := []ComponentStatus{{Component: core, ResidentMemoryGiB: core.MemoryGiB + 8}}
	if got := observedBundleBudget(cat, b, states); got != b.MemoryGiB+8 {
		t.Fatalf("core host omitted: %v", got)
	}
	states = append(states, ComponentStatus{Component: image, ResidentMemoryGiB: 5}, ComponentStatus{Component: speech, ResidentMemoryGiB: 5})
	if got := observedBundleBudget(cat, b, states); got < core.MemoryGiB+8+10 {
		t.Fatalf("retained auxiliaries omitted: %v", got)
	}
}

func TestResidentWeightsDoNotHideRequestWorkspace(t *testing.T) {
	c := Component{MemoryGiB: 5.25, WorkspaceMemoryGiB: 2.5}
	for _, tc := range []struct{ resident, want float64 }{{0, 5.25}, {2, 3.25}, {7, 2.5}} {
		if got := workloadAdditionalMemory(c, tc.resident); got != tc.want {
			t.Fatalf("resident %.2f: got %.2f want %.2f", tc.resident, got, tc.want)
		}
	}
}

func TestPersistedKleinProfileMigratesWithoutChangingCustomBudgets(t *testing.T) {
	old := Component{ComposeAsset: "compose.flux2.yaml", Model: "flux2-klein-4b-nvfp4", MemoryGiB: 5.25, WorkspaceMemoryGiB: 4.5}
	updated := componentDefaults(old)
	if updated.MemoryGiB != 10 || updated.StartupMemoryGiB != 3.75 || updated.WorkspaceMemoryGiB != 6.5 {
		t.Fatal(updated)
	}
	old.MemoryGiB = 14
	if got := componentDefaults(old); got.MemoryGiB != 14 || got.StartupMemoryGiB != 0 {
		t.Fatal("custom reservation changed", got)
	}
}
