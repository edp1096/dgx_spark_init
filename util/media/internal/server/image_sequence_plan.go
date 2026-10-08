package server

import (
	"mediaapp/internal/jobs"
	"time"
)

func buildImageJobPlan(rootID string, form imageCreateForm, params imageJobParams, now time.Time, _ *jobs.Job) []jobs.Job {
	root := jobs.Job{
		ID: rootID, Kind: "image", Status: "queued",
		Prompt: form.OriginalPrompt, Params: params.toMap(), CreatedAt: now,
	}
	if len(form.Sequence.Prompts) == 0 {
		return []jobs.Job{root}
	}
	planned := form.Sequence.Planned && len(form.Sequence.EnhancedPrompts) == len(form.Sequence.Prompts) && len(form.Sequence.Strategies) == len(form.Sequence.Prompts)
	enhancedPrompts := form.Sequence.EnhancedPrompts
	if !planned {
		enhancedPrompts = append([]string(nil), form.Sequence.Prompts...)
	}
	jobsToCreate := make([]jobs.Job, 0, len(form.Sequence.Prompts))
	for index := range form.Sequence.Prompts {
		childParams := params
		childParams.EnhancedPrompt = enhancedPrompts[index]
		childParams.SequenceID = rootID
		childParams.SequenceIndex = index + 1
		childParams.SequenceTotal = len(form.Sequence.Prompts)
		childParams.SequenceStrategy = "storyboard"
		childParams.SequenceMasterJobID = rootID
		childParams.SequenceSharedPrompt = form.Sequence.SharedPrompt
		childParams.SequenceCanonicalPrompt = form.Sequence.CanonicalPrompt
		childTime := now.Add(time.Duration(index) * time.Nanosecond)
		childParams.QueuedAt = childTime.Format(time.RFC3339Nano)
		if form.Seed >= 0 {
			childParams.Seed = form.Seed + int64(index)
		}
		childID := rootID
		if index > 0 {
			childID = newID()
		}
		jobsToCreate = append(jobsToCreate, jobs.Job{
			ID: childID, Kind: "image", Status: "queued",
			Prompt: form.Sequence.Prompts[index], Params: childParams.toMap(), CreatedAt: childTime,
		})
	}
	return jobsToCreate
}
