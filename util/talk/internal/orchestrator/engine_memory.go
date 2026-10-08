package orchestrator

import (
	"encoding/json"
	"math"
	"time"
)

// Allocator totals already contain KV, recurrent states and graph allocations.
// Load deltas are phase observations, not extra allocations to add to the total.
type EngineMemory struct {
	Schema        int     `json:"schema"`
	Unit          string  `json:"unit"`
	Context       int     `json:"context_tokens"`
	Capacity      int     `json:"capacity_tokens"`
	KVDType       string  `json:"kv_dtype"`
	MTPLayers     int     `json:"mtp_layers"`
	TargetLoad    float64 `json:"target_load_delta_gib"`
	DraftLoad     float64 `json:"draft_load_delta_gib"`
	KV            float64 `json:"kv_and_qsa_gib"`
	Mamba         float64 `json:"mamba_cache_gib"`
	Allocated     float64 `json:"cuda_allocated_gib"`
	Reserved      float64 `json:"cuda_reserved_gib"`
	PeakAllocated float64 `json:"cuda_peak_allocated_gib"`
	PeakReserved  float64 `json:"cuda_peak_reserved_gib"`
}

func (m *EngineMemory) valid(context, capacity int) bool {
	if m == nil || m.Schema != 1 || m.Unit != "GiB" || context <= 0 || capacity < context || m.Context != context || m.Capacity != capacity || m.KVDType == "" || m.MTPLayers < 0 {
		return false
	}
	for _, n := range []float64{m.TargetLoad, m.DraftLoad, m.KV, m.Mamba, m.Allocated, m.Reserved, m.PeakAllocated, m.PeakReserved} {
		if n < 0 || math.IsNaN(n) || math.IsInf(n, 0) {
			return false
		}
	}
	return m.Allocated > 0 && m.Reserved >= m.Allocated && m.PeakReserved >= m.Reserved && m.PeakAllocated >= m.Allocated && m.KV <= m.Allocated
}

type engineMemoryObservation struct {
	value EngineMemory
	at    time.Time
}

func engineMemoryKey(c Component) string {
	options, _ := json.Marshal(c.RuntimeOptions)
	return c.DeploymentKey() + "|" + c.Model + "|" + c.HealthURL + "|" + string(options)
}

func (c *Controller) recordEngineMemory(component Component, value *EngineMemory, context, capacity int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.engineMemory == nil {
		c.engineMemory = map[string]engineMemoryObservation{}
	}
	key := engineMemoryKey(component)
	delete(c.engineMemory, key)
	if value.valid(context, capacity) {
		c.engineMemory[key] = engineMemoryObservation{*value, time.Now()}
	}
}

func (c *Controller) observedEngineMemory(component Component) *EngineMemory {
	c.mu.RLock()
	defer c.mu.RUnlock()
	value, ok := c.engineMemory[engineMemoryKey(component)]
	if !ok || time.Since(value.at) > 30*time.Second {
		return nil
	}
	m := value.value
	return &m
}
