package server

import (
	"mediaapp/internal/jobs"
	"time"
)

func buildImageJobPlan(rootID string, form imageCreateForm, params imageJobParams, now time.Time, sequenceBase *jobs.Job) []jobs.Job {
	root := jobs.Job{
		ID: rootID, Kind: "image", Status: "queued",
		Prompt: form.OriginalPrompt, Params: params.toMap(), CreatedAt: now,
	}
	if len(form.Sequence.Prompts) == 0 {
		return []jobs.Job{root}
	}

	params.EnhancedPrompt = ""
	params.SequenceID = rootID
	params.SequenceIndex = 1
	params.SequenceTotal = len(form.Sequence.Prompts)
	params.SequenceIdentityStrength = form.Sequence.IdentityStrength
	root.Prompt = form.Sequence.Prompts[0]
	root.Params = params.toMap()

	jobsToCreate := make([]jobs.Job, 0, len(form.Sequence.Prompts))
	previousID := rootID
	if sequenceBase == nil {
		jobsToCreate = append(jobsToCreate, root)
	} else {
		previousID = sequenceBase.ID
	}
	for index := 1; index < len(form.Sequence.Prompts); index++ {
		childID := newID()
		childParams := params
		childParams.Identity = true
		childParams.IdentityReference = false
		childParams.IdentityReferenceCount = 0
		childParams.IdentityStrength = form.Sequence.IdentityStrength
		childParams.ReferenceBoost = 4
		childParams.GroundingPixels = 768
		childParams.Steps = max(childParams.Steps, 10)
		childParams.EnhancedPrompt = sequenceEditPrompt(form.Sequence.Prompts[index])
		childParams.ParentJobID = previousID
		childParams.SequencePreviousJobID = previousID
		childParams.SequenceID = rootID
		childParams.SequenceIndex = index + 1
		childParams.SequenceTotal = len(form.Sequence.Prompts)
		childParams.SequenceIdentityStrength = form.Sequence.IdentityStrength
		childParams.SequenceRegion = form.Sequence.Regions[index]
		if childParams.SequenceRegion != "all" {
			childParams.Identity = false
			childParams.AnyPaint = true
			childParams.AnyPaintMask = true
			childParams.AnyPaintStrength = 1
			childParams.AnyPaintBoundary = 32
			childParams.Styles = []styleSelection{}
			childParams.UserLoRAs = []userLoRASelection{}
			childParams.Style = ""
		}
		childTime := now.Add(time.Duration(index) * time.Nanosecond)
		childParams.QueuedAt = childTime.Format(time.RFC3339Nano)
		if form.Seed >= 0 {
			childParams.Seed = form.Seed + int64(index)
		}
		jobsToCreate = append(jobsToCreate, jobs.Job{
			ID: childID, Kind: "image", Status: "queued",
			Prompt: form.Sequence.Prompts[index], Params: childParams.toMap(), CreatedAt: childTime,
		})
		previousID = childID
	}
	return jobsToCreate
}
