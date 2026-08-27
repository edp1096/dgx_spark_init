package server

import (
	"testing"
	"time"
)

func TestImageJobParamsRoundTripKeepsPersistedContract(t *testing.T) {
	options := imageGenerationOptions{
		checkpoint: "official", identityPath: "/input/source.png",
		identityRefPaths: []string{"/input/reference.png"}, identityPreset: "tryon",
		identityAutoPrompt: true, identityUserPrompt: true, identityStrength: 1.2,
		refBoost: 5, sourceRefBoost: 1.5, groundingPX: 768, steps: 10,
		samplingPreset: "detail", sampler: "er_sde", scheduler: "simple",
		filterMode: "balanced", filterStrength: 1, promptEnhStrength: 1,
		promptTextScale: 1.75, vaeMode: "default", identityFitMode: "fit",
		identityModel: "convrot", identityEncoder: "heretic",
		visionMode: "descriptor", visionMegapixels: 1,
	}
	params := imageJobParamsFromOptions(1024, 768, 42, 0, "create", "krea-2", "", .65, options)
	params.IdentityPreserveItems = []string{"face", "hair"}
	params.EnhancedPrompt = "enhanced"
	params.Stage = "queued"
	params.QueuedAt = "2026-08-27T00:00:00Z"

	stored := params.toMap()
	decoded := decodeImageJobParams(stored)
	if decoded.Width != 1024 || decoded.Height != 768 || decoded.Seed != 42 || decoded.Mode != "create" {
		t.Fatalf("core params did not round-trip: %#v", decoded)
	}
	if !decoded.Identity || !decoded.IdentityReference || decoded.IdentityReferenceCount != 1 {
		t.Fatalf("identity contract did not round-trip: %#v", decoded)
	}
	if decoded.IdentityPreset != "tryon" || len(decoded.IdentityPreserveItems) != 2 || decoded.Sampler != "er_sde" {
		t.Fatalf("Krea options did not round-trip: %#v", decoded)
	}
}

func TestDecodeImageJobParamsAppliesLegacyDefaults(t *testing.T) {
	decoded := decodeImageJobParams(map[string]any{"width": float64(512), "height": float64(512)})
	if decoded.Mode != "create" || decoded.Seed != -1 || decoded.ControlType != "canny" {
		t.Fatalf("legacy defaults were not applied: %#v", decoded)
	}
	if decoded.IdentityModel != "convrot" || decoded.IdentityEncoder != "heretic" || decoded.Steps != 8 {
		t.Fatalf("Krea legacy defaults were not applied: %#v", decoded)
	}
}

func TestBuildImageJobPlanUsesTypedSequenceContract(t *testing.T) {
	form := imageCreateForm{
		OriginalPrompt: "scene one", Seed: 100,
		Sequence: imageSequenceForm{
			Prompts: []string{"scene one", "scene two"}, Regions: []string{"all", "left-arm"},
			IdentityStrength: .8,
		},
	}
	params := newImageJobParams()
	params.Width, params.Height, params.Mode, params.Stage = 1024, 1024, "create", "queued"
	params.QueuedAt = "2026-08-27T00:00:00Z"
	planned := buildImageJobPlan("root", form, params, time.Unix(0, 0), nil)
	if len(planned) != 2 {
		t.Fatalf("expected two planned jobs, got %d", len(planned))
	}
	child := decodeImageJobParams(planned[1].Params)
	if child.SequencePreviousJobID != "root" || child.SequenceIndex != 2 || child.SequenceRegion != "left-arm" {
		t.Fatalf("unexpected child sequence contract: %#v", child)
	}
	if child.Identity || !child.AnyPaint || !child.AnyPaintMask || child.Seed != 101 {
		t.Fatalf("regional sequence job was planned incorrectly: %#v", child)
	}
}
