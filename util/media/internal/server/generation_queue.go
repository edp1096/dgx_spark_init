package server

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"mediaapp/internal/jobs"
	"net/http"
	"strings"
	"time"
)

func (s *Server) wakeGenerationQueue() {
	s.generationQueueOnce.Do(func() { go s.generationQueueLoop() })
	select {
	case s.generationQueueWake <- struct{}{}:
	default:
	}
}

func (s *Server) generationQueueLoop() {
	for {
		job, ok := s.nextQueuedGeneration()
		if !ok {
			<-s.generationQueueWake
			continue
		}
		s.executeQueuedGeneration(job)
	}
}

func isGenerationKind(kind string) bool {
	return kind == "image" || kind == "video" || kind == "speech"
}

func (s *Server) nextQueuedGeneration() (jobs.Job, bool) {
	s.annotateQueuedModelPlans()
	var next jobs.Job
	found := false
	for _, job := range s.jobs.List() {
		if !isGenerationKind(job.Kind) || job.Status != "queued" {
			continue
		}
		if !s.generationDependencyReady(job) {
			continue
		}
		if !found || generationComesBefore(job, next) {
			next, found = job, true
		}
	}
	return next, found
}

func (s *Server) annotateQueuedModelPlans() {
	for _, job := range s.jobs.List() {
		if !isGenerationKind(job.Kind) || job.Status != "queued" || job.Params == nil || job.Params["model_plan"] != nil {
			continue
		}
		setInitialModelPlan(&job)
		_ = s.jobs.Save(job)
	}
}

func (s *Server) generationDependencyReady(job jobs.Job) bool {
	previousID := ""
	if job.Kind == "image" {
		params := decodeImageJobParams(job.Params)
		if params.SequenceStrategy == "major" && params.SequencePreviousJobID != "" && !params.SequenceDraftReady {
			previousID = params.SequenceMasterJobID
		} else {
			previousID = params.SequencePreviousJobID
		}
	}
	if previousID == "" {
		return true
	}
	previous, ok := s.jobs.Get(previousID)
	return !ok || (previous.Status != "queued" && previous.Status != "running")
}

func generationComesBefore(candidate, current jobs.Job) bool {
	left := decodeImageJobParams(candidate.Params)
	right := decodeImageJobParams(current.Params)
	if candidate.Kind == "image" && current.Kind == "image" && left.SequenceID != "" && left.SequenceID == right.SequenceID {
		leftRank := sequenceQueuePhase(left)
		rightRank := sequenceQueuePhase(right)
		if leftRank != rightRank {
			return leftRank < rightRank
		}
	}
	leftTime, rightTime := generationQueueTime(candidate), generationQueueTime(current)
	return leftTime.Before(rightTime) || (leftTime.Equal(rightTime) && candidate.ID < current.ID)
}

func sequenceQueuePhase(params imageJobParams) int {
	if params.SequenceStrategy == "major" && params.SequencePreviousJobID != "" {
		if params.SequenceDraftReady {
			return 2
		}
		return 0
	}
	return 1
}

func generationQueueTime(job jobs.Job) time.Time {
	if value, ok := job.Params["queued_at"].(string); ok {
		if parsed, err := time.Parse(time.RFC3339Nano, value); err == nil {
			return parsed
		}
	}
	return job.CreatedAt
}

func (s *Server) executeQueuedGeneration(job jobs.Job) {
	if s.generationEngineBusy(job) {
		s.generationStateMu.Lock()
		current, ok := s.jobs.Get(job.ID)
		if ok && current.Status == "queued" {
			if current.Params == nil {
				current.Params = map[string]any{}
			}
			current.Params["stage"] = s.generationWaitingStage(current)
			_ = s.jobs.Save(current)
		}
		s.generationStateMu.Unlock()
		// Avoid a tight queue loop while an orphaned request finishes in the
		// independently running SeedVR2 service.
		time.Sleep(2 * time.Second)
		return
	}
	s.generationStateMu.Lock()
	current, ok := s.jobs.Get(job.ID)
	if !ok || current.Status != "queued" || !isGenerationKind(current.Kind) {
		s.generationStateMu.Unlock()
		return
	}
	setInitialModelPlan(&current)
	transitionJobRunning(&current, "model-preparing", time.Now())
	if err := s.jobs.Save(current); err != nil {
		s.generationStateMu.Unlock()
		return
	}
	s.generationStateMu.Unlock()
	ctx, cancel := context.WithCancel(context.Background())
	s.generationCancelMu.Lock()
	s.generationCancels[current.ID] = cancel
	s.generationCancelMu.Unlock()
	defer func() {
		cancel()
		s.generationCancelMu.Lock()
		delete(s.generationCancels, current.ID)
		s.generationCancelMu.Unlock()
	}()
	// A cancellation can race with registration immediately after the job is
	// changed from queued to running. Observe the persisted state once more.
	if s.jobCancelled(current.ID) {
		return
	}

	switch current.Kind {
	case "speech":
		if err := s.markResidentRuntimeReady(&current, generationModelPlan(current)); err != nil {
			s.fail(current, err)
			return
		}
		s.runSpeech(ctx, current, decodeSpeechJobParams(current.Params, s.config().Speech))
	case "video":
		if decodeVideoJobParams(current.Params).Mode == "upscale" {
			if err := s.prepareSimpleRuntime(ctx, &current, generationModelPlan(current)); err != nil {
				if s.requeueGenerationAfterEngineConflict(current, err) {
					return
				}
				s.fail(current, err)
				return
			}
			s.runVideoUpscale(ctx, current)
			return
		}
		videoParams := decodeVideoJobParams(current.Params)
		if err := s.prepareVideoRuntime(ctx, &current, videoParams); err != nil {
			if s.requeueGenerationAfterEngineConflict(current, err) {
				return
			}
			s.fail(current, err)
			return
		}
		execution, err := s.loadVideoExecution(current)
		if err != nil {
			s.fail(current, err)
			return
		}
		s.runVideo(ctx, current, execution.prompt, execution.conditions, execution.audioPaths, execution.audioStarts, execution.width, execution.height, execution.frames, execution.fps, execution.seed)
	case "image":
		s.executeQueuedImage(ctx, current)
	}
}

type generationEngineTarget struct {
	name     string
	endpoint string
	waiting  string
}

func (s *Server) generationTarget(job jobs.Job) generationEngineTarget {
	cfg := s.config()
	switch job.Kind {
	case "video":
		if decodeVideoJobParams(job.Params).Mode == "upscale" {
			return generationEngineTarget{"upscale", cfg.Engines["upscale"].Endpoint, "waiting_upscale_engine"}
		}
		return generationEngineTarget{"video", cfg.Engines["video"].Endpoint, "waiting_video_engine"}
	case "image":
		mode := decodeImageJobParams(job.Params).Mode
		switch mode {
		case "upscale":
			return generationEngineTarget{"upscale", cfg.Engines["upscale"].Endpoint, "waiting_upscale_engine"}
		case "garment_extract":
			return generationEngineTarget{"garment", cfg.Engines["garment"].Endpoint, "waiting_garment_engine"}
		case "face_swap":
			return generationEngineTarget{"faceswap", cfg.Engines["faceswap"].Endpoint, "waiting_faceswap_engine"}
		case "detail_enhance":
			mode = "create"
		}
		backend, ok := cfg.Image.Backends[mode]
		if !ok {
			backend = cfg.Image.Backends[cfg.Image.DefaultMode]
		}
		return generationEngineTarget{"image", backend.Endpoint, "waiting_image_engine"}
	default:
		return generationEngineTarget{}
	}
}

func (s *Server) generationWaitingStage(job jobs.Job) string {
	if stage := s.generationTarget(job).waiting; stage != "" {
		return stage
	}
	return "queued"
}

// generationEngineBusy keeps queued GPU work from colliding with an orphaned
// request that survived an app restart. Engines without a busy health field
// safely fall through.
func (s *Server) generationEngineBusy(job jobs.Job) bool {
	target := s.generationTarget(job)
	endpoint := strings.TrimRight(target.endpoint, "/")
	if endpoint == "" {
		return false
	}
	response, err := s.health.Get(endpoint + "/health")
	if err != nil {
		return false
	}
	defer response.Body.Close()
	if response.StatusCode/100 != 2 {
		// ComfyUI-backed APIs can accept TCP connections before their internal
		// node server is ready. A 503 is a transient engine preparation state;
		// keep the durable job queued instead of starting it into a predictable
		// model-prepare failure. Other status codes remain ordinary failures.
		return response.StatusCode == http.StatusServiceUnavailable
	}
	var status struct {
		Busy    bool            `json:"busy"`
		Runtime json.RawMessage `json:"runtime"`
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 64<<10)).Decode(&status); err != nil {
		return false
	}
	s.runtimeCapabilityMu.Lock()
	s.runtimeCapabilities[endpoint] = len(status.Runtime) > 0 && string(status.Runtime) != "null"
	s.runtimeCapabilityMu.Unlock()
	if !status.Busy && target.name != "" {
		s.engineDrainMu.Lock()
		delete(s.engineDraining, target.name)
		s.engineDrainMu.Unlock()
	}
	return status.Busy
}

func isGenerationEngineBusyConflict(err error) bool {
	var responseErr *engineHTTPError
	if !errors.As(err, &responseErr) || responseErr.StatusCode != http.StatusConflict {
		return false
	}
	body := strings.ToLower(responseErr.Body)
	return (strings.Contains(body, "generation") && strings.Contains(body, "running")) ||
		(strings.Contains(body, "operation") && strings.Contains(body, "running")) ||
		strings.Contains(body, "engine is busy")
}

// requeueGenerationAfterEngineConflict closes the small race between the health
// probe and the generation request. A downstream 409 means that an older
// request still owns the GPU, so the durable job waits instead of becoming a
// user-visible failure.
func (s *Server) requeueGenerationAfterEngineConflict(job jobs.Job, cause error) bool {
	if !isGenerationEngineBusyConflict(cause) {
		return false
	}
	target := s.generationTarget(job)
	if target.name == "" {
		return false
	}
	s.generationStateMu.Lock()
	current, ok := s.jobs.Get(job.ID)
	if ok && current.Status == "running" {
		ensureJobParams(&current)
		current.Status = "queued"
		current.Error = ""
		current.Params["stage"] = target.waiting
		delete(current.Params, "started_at")
		_ = s.jobs.Save(current)
	}
	s.generationStateMu.Unlock()
	s.engineDrainMu.Lock()
	s.engineDraining[target.name] = true
	s.engineDrainMu.Unlock()
	s.wakeGenerationQueue()
	return true
}
