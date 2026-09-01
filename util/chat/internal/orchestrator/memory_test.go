package orchestrator

import (
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
