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
	if anonymousMemoryGiB(stat) != 1 {
		t.Fatal("file cache must not be counted as memory released by stopping a service")
	}
}
