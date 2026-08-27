package server

import (
	"mediaapp/internal/jobs"
	"testing"
	"time"
)

func TestVideoJobParamsRoundTripGenerationContract(t *testing.T) {
	params := newVideoJobParams()
	params.Mode = "a2v"
	params.Width, params.Height, params.Frames = 1280, 768, 121
	params.FPS, params.Seed = 24, 77
	params.StartImage, params.EndImage, params.Keyframes = true, true, 1
	params.Conditions = []savedVideoCondition{{Role: "start", FrameIdx: 0, Strength: 1}, {Role: "keyframe", Index: 0, FrameIdx: 60, Strength: .8}}
	params.Audio = true
	params.AudioSourceJobIDs = []string{"speech-1"}
	params.AudioClips = []savedVideoAudioClip{{Index: 0, SourceJobID: "speech-1", Start: 1.5}}
	params.MotionLoRAEnabled, params.MotionLoRAStrength = true, .7
	params.AccelerationRequested = "sol"

	decoded := decodeVideoJobParams(params.toMap())
	if decoded.Mode != "a2v" || decoded.Width != 1280 || decoded.Frames != 121 || decoded.Seed != 77 {
		t.Fatalf("video generation contract did not round-trip: %#v", decoded)
	}
	if len(decoded.Conditions) != 2 || len(decoded.AudioClips) != 1 || decoded.AudioClips[0].Start != 1.5 {
		t.Fatalf("video inputs did not round-trip: %#v", decoded)
	}
	if !decoded.MotionLoRAEnabled || decoded.AccelerationRequested != "sol" {
		t.Fatalf("video runtime options did not round-trip: %#v", decoded)
	}
}

func TestVideoJobParamsRoundTripUpscaleContract(t *testing.T) {
	params := newVideoJobParams()
	params.Mode = "upscale"
	params.SourceJobID, params.SourceKind = "video-1", "video"
	params.Width, params.Height = 2560, 1440
	params.SourceWidth, params.SourceHeight = 1280, 720
	params.UpscaleScale, params.BatchSize, params.TemporalOverlap = 2, 9, 2
	params.SourceStartTime, params.SourceEndTime, params.Duration = 3, 13, 10
	params.FPS, params.Seed = 24, 123
	params.Stage, params.QueuedAt = "queued", "2026-08-27T00:00:00Z"

	decoded := decodeVideoJobParams(params.toMap())
	if decoded.Mode != "upscale" || decoded.SourceJobID != "video-1" || decoded.UpscaleScale != 2 {
		t.Fatalf("video upscale contract did not round-trip: %#v", decoded)
	}
	if decoded.SourceStartTime != 3 || decoded.SourceEndTime != 13 || decoded.Duration != 10 {
		t.Fatalf("video upscale range did not round-trip: %#v", decoded)
	}
}

func TestResetGenerationJobForRetryPreservesContract(t *testing.T) {
	job := jobs.Job{
		ID: "job-1", Kind: "video", Status: "failed", Error: "boom", OutputURL: "/old.mp4",
		Params: map[string]any{"mode": "generate", "started_at": "old", "retry_count": 2, "width": 1280},
	}
	now := time.Date(2026, 8, 27, 1, 2, 3, 0, time.UTC)
	retried := resetGenerationJobForRetry(job, now)
	if retried.Status != "queued" || retried.Error != "" || retried.OutputURL != "" {
		t.Fatalf("job state was not reset: %#v", retried)
	}
	if _, exists := retried.Params["started_at"]; exists {
		t.Fatal("started_at must be cleared when retrying")
	}
	if retried.Params["stage"] != "queued" || intParam(retried.Params, "retry_count", 0) != 3 {
		t.Fatalf("retry metadata was not updated: %#v", retried.Params)
	}
	if intParam(retried.Params, "width", 0) != 1280 {
		t.Fatal("media-specific params must survive a retry")
	}
}
