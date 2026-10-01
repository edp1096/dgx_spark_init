package orchestrator

import (
	"math"
	"testing"
)

func TestEngineMemoryReceiptRejectsStaleOrInvalidGeometry(t *testing.T) {
	m := EngineMemory{Schema: 1, Unit: "GiB", Context: 1048576, Capacity: 1048576, KVDType: "fp8", MTPLayers: 1,
		TargetLoad: 75, DraftLoad: 4, KV: 13.8, Mamba: 1.8, Allocated: 97, Reserved: 99, PeakAllocated: 98, PeakReserved: 100}
	if !m.valid(1048576, 1048576) {
		t.Fatal("valid receipt rejected")
	}
	if m.valid(262144, 262144) {
		t.Fatal("different allocation accepted")
	}
	for _, change := range []func(*EngineMemory){
		func(m *EngineMemory) { m.Reserved = 96 },
		func(m *EngineMemory) { m.KV = math.NaN() },
		func(m *EngineMemory) { m.DraftLoad = -1 },
		func(m *EngineMemory) { m.Unit = "GB" },
	} {
		bad := m
		change(&bad)
		if bad.valid(1048576, 1048576) {
			t.Fatal("invalid receipt accepted")
		}
	}
	c := &Controller{}
	x := Component{ID: "qad", Model: QwenQADHuihuiLIL, HealthURL: "http://localhost:8000/health"}
	c.recordEngineMemory(x, &m, 1048576, 1048576)
	if c.observedEngineMemory(x) == nil {
		t.Fatal("receipt not retained")
	}
	other := x
	other.Model = QwenQADOfficial
	if c.observedEngineMemory(other) != nil {
		t.Fatal("receipt reused for another checkpoint")
	}
	c.recordEngineMemory(x, nil, 1048576, 1048576)
	if c.observedEngineMemory(x) != nil {
		t.Fatal("old receipt retained after missing measurement")
	}
}

func TestLoadedCoreBudgetDoesNotAddColdProfileOrKVAgain(t *testing.T) {
	cat, _ := LoadCatalog()
	b, _ := cat.Bundle("flash-next")
	core, _ := cat.ResolveComponent(b.ID, "flash-next")
	m := &EngineMemory{Schema: 1, Unit: "GiB", Context: 1048576, Capacity: 1048576, KVDType: "fp8", KV: 13.8, Allocated: 94, Reserved: 96, PeakAllocated: 95, PeakReserved: 98}
	s := ComponentStatus{Component: core, Health: "online", MemoryMeasured: true, ResidentMemoryGiB: 100, EngineMemory: m}
	want := b.MemoryGiB - core.MemoryGiB + 102 // resident 100 plus unretained peak 2
	if got := observedBundleBudget(cat, b, []ComponentStatus{s}); got != want {
		t.Fatalf("got %v want %v", got, want)
	}
	s.EngineMemory = nil
	if got := observedBundleBudget(cat, b, []ComponentStatus{s}); got < b.MemoryGiB {
		t.Fatal("missing telemetry reduced the cold fallback")
	}
}
