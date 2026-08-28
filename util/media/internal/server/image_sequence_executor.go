package server

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"math"
	"mediaapp/internal/jobs"
	"os"
	"path/filepath"
	"time"
)

func (s *Server) runMajorSequenceDraft(ctx context.Context, job jobs.Job, execution generationImageExecution) {
	s.heavyMu.Lock()
	defer s.heavyMu.Unlock()

	backend := s.config().Image.Backends[execution.mode]
	setImageSequenceStage(&job, "pose-draft")
	_ = s.saveJobPreservingRuntime(job)
	draftOptions := execution.options
	clearKreaEditInputs(&draftOptions)
	draftOptions.promptEnhancer = false
	observer := s.startRuntimeObserver(ctx, job.ID, backend.Endpoint)
	draftResponse, err := s.generateKreaCreate(ctx, backend, job.ID, execution.prompt, execution.width, execution.height, execution.seed, draftOptions)
	observer.Stop()
	if err != nil {
		if s.requeueGenerationAfterEngineConflict(job, err) {
			return
		}
		s.fail(job, fmt.Errorf("sequence pose draft: %w", err))
		return
	}
	draft, err := decodeImage(draftResponse)
	if err != nil {
		s.fail(job, fmt.Errorf("sequence pose draft: %w", err))
		return
	}
	draftPath := filepath.Join(s.dataDir, "inputs", job.ID, "sequence-draft", "0.png")
	if err := os.MkdirAll(filepath.Dir(draftPath), 0o755); err != nil {
		s.fail(job, err)
		return
	}
	if err := os.WriteFile(draftPath, draft, 0o644); err != nil {
		s.fail(job, err)
		return
	}
	ensureJobParams(&job)
	job.Status = "queued"
	job.Error = ""
	job.Params["sequence_draft_ready"] = true
	if started, parseErr := time.Parse(time.RFC3339Nano, stringParam(job.Params, "generation_started_at", "")); parseErr == nil {
		job.Params["sequence_draft_seconds"] = max(0, time.Since(started).Seconds())
	}
	job.Params["stage"] = "draft-ready"
	job.Params["stage_started_at"] = time.Now().Format(time.RFC3339Nano)
	job.Params["model_plan"] = modelPlanMap(imageRuntimePlan(decodeImageJobParams(job.Params), true))
	delete(job.Params, "generation_started_at")
	if err := s.saveJobPreservingRuntime(job); err != nil {
		s.fail(job, err)
		return
	}
	s.wakeGenerationQueue()
}

func (s *Server) runMajorSequenceIdentity(ctx context.Context, job jobs.Job, execution generationImageExecution) {
	s.heavyMu.Lock()
	defer s.heavyMu.Unlock()

	params := decodeImageJobParams(job.Params)
	backend := s.config().Image.Backends[execution.mode]
	previousPaths, err := s.imageInputFiles(job.ID, "sequence-previous")
	if err != nil || len(previousPaths) == 0 {
		s.fail(job, fmt.Errorf("previous sequence image is missing"))
		return
	}
	masterPaths, err := s.imageInputFiles(job.ID, "sequence-master")
	if err != nil || len(masterPaths) == 0 {
		s.fail(job, fmt.Errorf("sequence master image is missing"))
		return
	}
	draftPath := filepath.Join(s.dataDir, "inputs", job.ID, "sequence-draft", "0.png")
	if info, statErr := os.Stat(draftPath); statErr != nil || info.Size() == 0 {
		s.fail(job, fmt.Errorf("sequence pose draft is missing"))
		return
	}

	finalPrompt := sequenceIdentityTransferPrompt(execution.prompt)
	finalOptions := execution.options
	clearKreaEditInputs(&finalOptions)
	finalOptions.promptEnhancer = false
	finalOptions.identityPath = draftPath
	finalOptions.identityRefPaths = []string{masterPaths[0]}
	finalOptions.identityStrength = math.Max(0.42, math.Min(params.SequenceIdentityStrength, 0.60))
	finalOptions.sourceRefBoost = 0.55
	finalOptions.refBoost = 3
	finalOptions.groundingPX = 768
	finalOptions.steps = max(finalOptions.steps, 10)
	identityPlan := modelRuntimePlan{
		Engine: "image", Profile: "krea-identity-" + valueOr(finalOptions.identityModel, "convrot") + "-" + valueOr(finalOptions.identityEncoder, "heretic"),
		Label:        "Krea Identity Edit 탑재",
		Components:   []string{"Krea Identity Edit", valueOr(finalOptions.identityEncoder, "heretic") + " text encoder", "Qwen Image VAE"},
		RuntimeOrder: []string{"Identity 워크플로우 준비", "Identity DiT·인코더·VAE 탑재", "참조 조건 인코딩", "Identity 추론", "VAE 디코딩", "ComfyUI 캐시 유지"},
		RequiresSwap: true, EstimateSeconds: 55,
	}
	identityExecution := execution
	identityExecution.prompt = finalPrompt
	identityExecution.options = finalOptions
	if err := s.prepareKreaRuntime(ctx, &job, identityExecution, identityPlan); err != nil {
		if s.requeueGenerationAfterEngineConflict(job, err) {
			return
		}
		s.fail(job, fmt.Errorf("sequence identity preparation: %w", err))
		return
	}
	setImageSequenceStage(&job, "identity-transfer")
	_ = s.saveJobPreservingRuntime(job)
	observer := s.startRuntimeObserver(ctx, job.ID, backend.Endpoint)
	response, err := s.generateKreaCreate(ctx, backend, job.ID, finalPrompt, execution.width, execution.height, execution.seed, finalOptions)
	observer.Stop()
	if err != nil {
		if s.requeueGenerationAfterEngineConflict(job, err) {
			return
		}
		s.fail(job, fmt.Errorf("sequence identity transfer: %w", err))
		return
	}
	data, err := decodeImage(response)
	if err != nil {
		s.fail(job, fmt.Errorf("sequence identity transfer: %w", err))
		return
	}

	setImageSequenceStage(&job, "quality-check")
	if similarity, ok := imageSimilarityFiles(previousPaths[0], data); ok {
		params.SequenceSimilarity = similarity
		job.Params["sequence_similarity"] = similarity
		// A major-action result that is almost identical to the preceding frame
		// normally means Identity Edit restored the old pose. Retry once with the
		// pose draft carrying more weight.
		if similarity > 0.965 {
			params.SequenceRetryCount = 1
			job.Params["sequence_retry_count"] = 1
			setImageSequenceStage(&job, "pose-retry")
			_ = s.saveJobPreservingRuntime(job)
			finalOptions.identityStrength = 0.42
			finalOptions.sourceRefBoost = 0.35
			finalOptions.refBoost = 2.4
			retrySeed := execution.seed
			if retrySeed >= 0 {
				retrySeed += 7919
			}
			retryObserver := s.startRuntimeObserver(ctx, job.ID, backend.Endpoint)
			response, err = s.generateKreaCreate(ctx, backend, job.ID, finalPrompt, execution.width, execution.height, retrySeed, finalOptions)
			retryObserver.Stop()
			if err != nil {
				if s.requeueGenerationAfterEngineConflict(job, err) {
					return
				}
				s.fail(job, fmt.Errorf("sequence pose retry: %w", err))
				return
			}
			data, err = decodeImage(response)
			if err != nil {
				s.fail(job, fmt.Errorf("sequence pose retry: %w", err))
				return
			}
			if retrySimilarity, valid := imageSimilarityFiles(previousPaths[0], data); valid {
				job.Params["sequence_similarity"] = retrySimilarity
			}
		}
	}
	if actualSeed, ok := decodeImageSeed(response); ok {
		job.Params["seed"] = actualSeed
	}
	setImageSequenceStage(&job, "completed")
	if err := s.writeImageResult(&job, data, finalPrompt); err != nil {
		s.fail(job, err)
	}
}

func sequenceIdentityTransferPrompt(scene string) string {
	return "Create the final frame described below. Image One is the pose and composition draft: preserve its exact body pose, limb placement, action, framing, and camera angle. Image Two is the continuity reference: transfer only the same subject identity, face, hair, body design, stable outfit details unless the final-frame description changes them, and stable visual style from it. Never restore Image Two's pose and never add limbs from Image Two. Render exactly one anatomically correct subject.\nFinal frame: " + scene
}

func clearKreaEditInputs(options *imageGenerationOptions) {
	options.identityPath = ""
	options.identityRefPaths = nil
	options.identityMaskPath = ""
	options.strictMaskPath = ""
	options.depthPath = ""
	options.visionPaths = nil
	options.styleRefPaths = nil
	options.nk2ePath = ""
	options.anypaintPath = ""
	options.anypaintMaskPath = ""
}

func setImageSequenceStage(job *jobs.Job, stage string) {
	if job.Params == nil {
		job.Params = map[string]any{}
	}
	job.Params["stage"] = stage
}

func imageSimilarityFiles(previousPath string, generated []byte) (float64, bool) {
	previousFile, err := os.Open(previousPath)
	if err != nil {
		return 0, false
	}
	defer previousFile.Close()
	previous, _, err := image.Decode(previousFile)
	if err != nil {
		return 0, false
	}
	current, _, err := image.Decode(bytes.NewReader(generated))
	if err != nil {
		return 0, false
	}
	const samples = 32
	var difference float64
	for y := 0; y < samples; y++ {
		for x := 0; x < samples; x++ {
			a := sampledLuma(previous, x, y, samples)
			b := sampledLuma(current, x, y, samples)
			difference += math.Abs(a - b)
		}
	}
	return math.Max(0, 1-difference/float64(samples*samples)/65535), true
}

func sampledLuma(source image.Image, x, y, grid int) float64 {
	bounds := source.Bounds()
	px := bounds.Min.X + min(bounds.Dx()-1, x*bounds.Dx()/grid)
	py := bounds.Min.Y + min(bounds.Dy()-1, y*bounds.Dy()/grid)
	r, g, b, _ := source.At(px, py).RGBA()
	return 0.2126*float64(r) + 0.7152*float64(g) + 0.0722*float64(b)
}
