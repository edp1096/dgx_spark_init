package server

import "testing"

func TestRuntimePhaseRequiresMatchingOperationAndKnownPhase(t *testing.T) {
	phase := runtimePhase{OperationID: "job", Phase: "model_loading"}
	if !phase.validFor("job") {
		t.Fatal("matching model loading phase should be accepted")
	}
	if phase.validFor("other") {
		t.Fatal("foreign engine operation must be ignored")
	}
	phase.Phase = "invented"
	if phase.validFor("job") {
		t.Fatal("unknown phase must be ignored")
	}
}

func TestMergeObservedRuntimeDoesNotOverwriteGenerationParams(t *testing.T) {
	target := map[string]any{"seed": 7, "stage": "running"}
	source := map[string]any{
		"seed":                  99,
		"runtime_phase":         map[string]any{"phase": "decoding"},
		"runtime_phase_history": []any{map[string]any{"phase": "sampling"}},
	}
	mergeObservedRuntime(&target, source)
	if target["seed"] != 7 || target["stage"] != "running" {
		t.Fatalf("generation params were overwritten: %#v", target)
	}
	if target["runtime_phase"] == nil || target["runtime_phase_history"] == nil {
		t.Fatalf("runtime telemetry was not merged: %#v", target)
	}
}
