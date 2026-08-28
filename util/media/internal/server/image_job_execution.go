package server

import (
	"context"
	"fmt"
	"mediaapp/internal/jobs"
	"os"
	"path/filepath"
)

// generationExecution contains everything needed to replay a persisted image
// job. Uploaded inputs live below data/inputs/<job id>, not in the container's
// temporary upload directory, so queued work survives browser and app restarts.
type generationImageExecution struct {
	prompt            string
	references        []string
	width, height     int
	seed              int64
	mode, controlType string
	controlStrength   float64
	options           imageGenerationOptions
}

func (s *Server) executeQueuedImage(ctx context.Context, job jobs.Job) {
	mode := decodeImageJobParams(job.Params).Mode
	if mode == "garment_extract" {
		if err := s.prepareSimpleRuntime(ctx, &job, generationModelPlan(job)); err != nil {
			s.fail(job, err)
			return
		}
		s.runGarmentExtraction(ctx, job)
		return
	}
	if mode == "face_swap" {
		if err := s.prepareSimpleRuntime(ctx, &job, generationModelPlan(job)); err != nil {
			s.fail(job, err)
			return
		}
		s.runFaceSwap(ctx, job)
		return
	}
	if mode == "detail_enhance" || mode == "upscale" {
		sourceID := stringParam(job.Params, "source_job_id", "")
		source, ok := s.jobs.Get(sourceID)
		if !ok || source.Kind != "image" || source.Status != "completed" || source.OutputURL == "" {
			s.fail(job, fmt.Errorf("source image is no longer available"))
			return
		}
		data, err := os.ReadFile(s.jobs.OutputPath(filepath.Base(source.OutputURL)))
		if err != nil {
			s.fail(job, fmt.Errorf("source image is no longer available: %w", err))
			return
		}
		if mode == "detail_enhance" {
			s.runImageDetailEnhance(ctx, job, data,
				floatParam(job.Params, "detail_strength", 1),
				int64Param(job.Params, "seed", -1),
				stringParam(job.Params, "detail_vae", "wan"))
		} else {
			if err := s.prepareSimpleRuntime(ctx, &job, generationModelPlan(job)); err != nil {
				if s.requeueGenerationAfterEngineConflict(job, err) {
					return
				}
				s.fail(job, err)
				return
			}
			s.runImageUpscale(ctx, job, data,
				intParam(job.Params, "upscale_scale", 2),
				int64Param(job.Params, "seed", -1))
		}
		return
	}
	params := decodeImageJobParams(job.Params)
	majorSequence := params.SequenceStrategy == "major" && params.SequencePreviousJobID != ""
	if majorSequence && params.SequenceDraftReady {
		if err := s.materializeSequenceIdentity(job); err != nil {
			s.fail(job, err)
			return
		}
		execution, err := s.loadImageExecution(job)
		if err != nil {
			s.fail(job, err)
			return
		}
		s.runMajorSequenceIdentity(ctx, job, execution)
		return
	}
	if !majorSequence {
		if err := s.materializeSequenceIdentity(job); err != nil {
			s.fail(job, err)
			return
		}
	}
	execution, err := s.loadImageExecution(job)
	if err != nil {
		s.fail(job, err)
		return
	}
	plan := generationModelPlan(job)
	prepareExecution := execution
	if majorSequence {
		clearKreaEditInputs(&prepareExecution.options)
		prepareExecution.options.promptEnhancer = false
	}
	if err := s.prepareKreaRuntime(ctx, &job, prepareExecution, plan); err != nil {
		if s.requeueGenerationAfterEngineConflict(job, err) {
			return
		}
		s.fail(job, err)
		return
	}
	if majorSequence {
		s.runMajorSequenceDraft(ctx, job, execution)
		return
	}
	s.runImage(ctx, job, execution.prompt, execution.references, execution.width, execution.height,
		execution.seed, execution.mode, execution.controlType, execution.controlStrength, execution.options)
}

func (s *Server) loadImageExecution(job jobs.Job) (generationImageExecution, error) {
	params := decodeImageJobParams(job.Params)
	result := generationImageExecution{
		prompt:          params.EnhancedPrompt,
		width:           params.Width,
		height:          params.Height,
		seed:            params.Seed,
		mode:            params.Mode,
		controlType:     params.ControlType,
		controlStrength: params.ControlStrength,
	}
	if result.prompt == "" {
		result.prompt = job.Prompt
	}
	if result.width == 0 {
		result.width = s.config().Image.DefaultWidth
	}
	if result.height == 0 {
		result.height = s.config().Image.DefaultHeight
	}
	if _, ok := s.config().Image.Backends[result.mode]; !ok {
		return result, fmt.Errorf("the original image backend is no longer configured")
	}
	var err error
	result.references, err = s.imageInputFiles(job.ID, "reference")
	if err != nil {
		return result, err
	}
	paths := map[string][]string{}
	for _, role := range []string{"identity", "sequence_character", "identity_reference", "identity_mask", "strict_mask", "depth", "vision", "style_reference", "nk2e", "anypaint", "anypaint_mask"} {
		paths[role], err = s.imageInputFiles(job.ID, role)
		if err != nil {
			return result, err
		}
	}
	require := func(enabled bool, role string) error {
		if enabled && len(paths[role]) == 0 {
			return fmt.Errorf("saved %s input is missing", role)
		}
		return nil
	}
	if params.References > len(result.references) ||
		params.IdentityReferenceCount > len(paths["identity_reference"]) ||
		params.VisionCount > len(paths["vision"]) ||
		params.StyleReferenceCount > len(paths["style_reference"]) {
		return result, fmt.Errorf("one or more saved reference images are missing")
	}
	for _, check := range []struct {
		enabled bool
		role    string
	}{
		{params.Identity, "identity"},
		{params.SequenceReID, "sequence_character"},
		{params.IdentityReference, "identity_reference"},
		{params.IdentityMask, "identity_mask"},
		{params.StrictMask, "strict_mask"},
		{params.Depth, "depth"},
		{params.NK2E, "nk2e"},
		{params.AnyPaint, "anypaint"},
		{params.AnyPaintMask, "anypaint_mask"},
	} {
		if err := require(check.enabled, check.role); err != nil {
			return result, err
		}
	}
	result.options = params.generationOptions(paths)
	return result, nil
}
