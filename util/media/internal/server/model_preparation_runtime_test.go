package server

import (
	"mediaapp/internal/jobs"
	"testing"
	"time"
)

func TestGenerationModelPlansDescribeRuntimeSwaps(t *testing.T) {
	major := jobs.Job{Kind: "image", Params: map[string]any{
		"mode": "create", "checkpoint": "official", "sequence_id": "sequence",
		"sequence_strategy": "major", "sequence_previous_job_id": "previous",
	}}
	plan := generationModelPlan(major)
	if plan.Profile != "krea-create" || !plan.RequiresSwap || len(plan.Following) == 0 {
		t.Fatalf("unexpected major draft plan: %#v", plan)
	}
	major.Params["sequence_draft_ready"] = true
	plan = generationModelPlan(major)
	if plan.Profile != "krea-identity-convrot-heretic" || plan.Label != "Krea Identity Edit 탑재" {
		t.Fatalf("unexpected identity phase plan: %#v", plan)
	}

	a2v := jobs.Job{Kind: "video", Params: map[string]any{"mode": "a2v", "audio": true}}
	if plan := generationModelPlan(a2v); plan.Profile != "ltx-a2v" || !plan.RequiresSwap {
		t.Fatalf("unexpected A2V plan: %#v", plan)
	}
	upscale := jobs.Job{Kind: "image", Params: map[string]any{"mode": "upscale"}}
	if plan := generationModelPlan(upscale); plan.Profile != "seedvr2" {
		t.Fatalf("unexpected upscale plan: %#v", plan)
	}
}

func TestSequenceQueueOrdersAllDraftsBeforeIdentityPasses(t *testing.T) {
	draft := imageJobParams{SequenceID: "sequence", SequenceStrategy: "major", SequencePreviousJobID: "previous"}
	identity := draft
	identity.SequenceDraftReady = true
	if sequenceQueuePhase(draft) >= sequenceQueuePhase(identity) {
		t.Fatalf("draft phase must be scheduled before identity: draft=%d identity=%d", sequenceQueuePhase(draft), sequenceQueuePhase(identity))
	}
}

func TestSequenceBatchingDoesNotReorderUnrelatedJobs(t *testing.T) {
	base := time.Date(2026, 8, 27, 12, 0, 0, 0, time.UTC)
	older := jobs.Job{ID: "unrelated", Kind: "video", CreatedAt: base, Params: map[string]any{
		"queued_at": base.Format(time.RFC3339Nano),
	}}
	draft := jobs.Job{ID: "draft", Kind: "image", CreatedAt: base.Add(time.Second), Params: imageJobParams{
		SequenceID: "sequence", SequenceStrategy: "major", SequencePreviousJobID: "master",
		QueuedAt: base.Add(time.Second).Format(time.RFC3339Nano),
	}.toMap()}
	if generationComesBefore(draft, older) {
		t.Fatal("batch optimization must not jump ahead of an older unrelated job")
	}

	identity := draft
	identity.ID = "identity"
	identity.Params = imageJobParams{
		SequenceID: "sequence", SequenceStrategy: "major", SequencePreviousJobID: "master",
		SequenceDraftReady: true, QueuedAt: base.Format(time.RFC3339Nano),
	}.toMap()
	if !generationComesBefore(draft, identity) {
		t.Fatal("a later draft in the same sequence must run before an identity pass")
	}
}
