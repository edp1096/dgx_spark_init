package server

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"mediaapp/internal/jobs"
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
	var next jobs.Job
	found := false
	for _, job := range s.jobs.List() {
		if !isGenerationKind(job.Kind) || job.Status != "queued" {
			continue
		}
		if !s.generationDependencyReady(job) {
			continue
		}
		if !found || generationQueueTime(job).Before(generationQueueTime(next)) ||
			(generationQueueTime(job).Equal(generationQueueTime(next)) && job.ID < next.ID) {
			next, found = job, true
		}
	}
	return next, found
}

func (s *Server) generationDependencyReady(job jobs.Job) bool {
	previousID := imageStringParam(job.Params, "sequence_previous_job_id", "")
	if previousID == "" {
		return true
	}
	previous, ok := s.jobs.Get(previousID)
	return !ok || (previous.Status != "queued" && previous.Status != "running")
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
	s.generationStateMu.Lock()
	current, ok := s.jobs.Get(job.ID)
	if !ok || current.Status != "queued" || !isGenerationKind(current.Kind) {
		s.generationStateMu.Unlock()
		return
	}
	if current.Params == nil {
		current.Params = map[string]any{}
	}
	current.Status = "running"
	current.Params["stage"] = "running"
	current.Params["started_at"] = time.Now().Format(time.RFC3339Nano)
	if err := s.jobs.Save(current); err != nil {
		s.generationStateMu.Unlock()
		return
	}
	s.generationStateMu.Unlock()

	switch current.Kind {
	case "speech":
		s.runSpeech(current,
			imageStringParam(current.Params, "language", s.config().Speech.DefaultLanguage),
			imageStringParam(current.Params, "speaker", s.config().Speech.DefaultSpeaker),
			imageStringParam(current.Params, "instructions", ""),
			imageInt64Param(current.Params, "seed", -1),
		)
	case "video":
		execution, err := s.loadVideoExecution(current)
		if err != nil {
			s.fail(current, err)
			return
		}
		s.runVideo(current, execution.prompt, execution.conditions, execution.width, execution.height, execution.frames, execution.fps, execution.seed)
	case "image":
		s.executeQueuedImage(current)
	}
}

func (s *Server) executeQueuedImage(job jobs.Job) {
	mode := imageStringParam(job.Params, "mode", "create")
	if mode == "garment_extract" {
		s.runGarmentExtraction(job)
		return
	}
	if mode == "detail_enhance" || mode == "upscale" {
		sourceID := imageStringParam(job.Params, "source_job_id", "")
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
			s.runImageDetailEnhance(job, data,
				imageFloatParam(job.Params, "detail_strength", 1),
				imageInt64Param(job.Params, "seed", -1),
				imageStringParam(job.Params, "detail_vae", "wan"))
		} else {
			s.runImageUpscale(job, data,
				imageIntParam(job.Params, "upscale_scale", 2),
				imageInt64Param(job.Params, "seed", -1))
		}
		return
	}
	if err := s.materializeSequenceIdentity(job); err != nil {
		s.fail(job, err)
		return
	}

	execution, err := s.loadImageExecution(job)
	if err != nil {
		s.fail(job, err)
		return
	}
	s.runImage(job, execution.prompt, execution.references, execution.width, execution.height,
		execution.seed, execution.mode, execution.controlType, execution.controlStrength, execution.options)
}

// materializeSequenceIdentity links the previous scene result into this job's
// persisted inputs. Full-frame edits use Identity Edit; region edits use
// AnyPaint so the mask participates in generation instead of post-compositing.
func (s *Server) materializeSequenceIdentity(job jobs.Job) error {
	previousID := imageStringParam(job.Params, "sequence_previous_job_id", "")
	if previousID == "" {
		return nil
	}
	previous, ok := s.jobs.Get(previousID)
	if !ok || previous.Kind != "image" {
		return fmt.Errorf("previous sequence image no longer exists")
	}
	if previous.Status != "completed" || previous.OutputURL == "" {
		return fmt.Errorf("previous sequence image did not complete")
	}
	source := s.jobs.OutputPath(filepath.Base(previous.OutputURL))
	ext := strings.ToLower(filepath.Ext(source))
	if ext == "" {
		ext = ".png"
	}
	region := imageStringParam(job.Params, "sequence_region", "all")
	role := "identity"
	if region != "all" {
		role = "anypaint"
	}
	dir := filepath.Join(s.dataDir, "inputs", job.ID, role)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	destination := filepath.Join(dir, "0"+ext)
	if _, err := os.Stat(destination); err != nil && !os.IsNotExist(err) {
		return err
	} else if os.IsNotExist(err) {
		if err := linkOrCopyFile(source, destination); err != nil {
			return fmt.Errorf("prepare previous sequence image: %w", err)
		}
	}
	if region == "all" {
		return nil
	}
	return s.materializeSequenceAnyPaintMask(job, source)
}

// materializeSequenceAnyPaintMask creates a normalized generation mask after
// the preceding scene exists. White pixels are regenerated by AnyPaint while
// the known image outside the mask is preserved by the diffusion workflow.
func (s *Server) materializeSequenceAnyPaintMask(job jobs.Job, source string) error {
	region := imageStringParam(job.Params, "sequence_region", "all")
	if region == "all" {
		return nil
	}
	if region == "custom" {
		dir := filepath.Join(s.dataDir, "inputs", job.ID, "anypaint-mask")
		entries, err := os.ReadDir(dir)
		if err != nil || len(entries) == 0 {
			return fmt.Errorf("painted sequence mask is missing")
		}
		return nil
	}
	input, err := os.Open(source)
	if err != nil {
		return err
	}
	config, _, err := image.DecodeConfig(input)
	_ = input.Close()
	if err != nil || config.Width <= 0 || config.Height <= 0 {
		return fmt.Errorf("read sequence source dimensions: %w", err)
	}
	mask := image.NewGray(image.Rect(0, 0, config.Width, config.Height))
	paint := func(x0, y0, x1, y1 float64) {
		rect := image.Rect(int(x0*float64(config.Width)), int(y0*float64(config.Height)), int(x1*float64(config.Width)), int(y1*float64(config.Height))).Intersect(mask.Bounds())
		for y := rect.Min.Y; y < rect.Max.Y; y++ {
			for x := rect.Min.X; x < rect.Max.X; x++ {
				mask.SetGray(x, y, color.Gray{Y: 255})
			}
		}
	}
	switch region {
	case "left":
		paint(0, 0, .58, 1)
	case "right":
		paint(.42, 0, 1, 1)
	case "upper":
		paint(0, 0, 1, .58)
	case "lower":
		paint(0, .42, 1, 1)
	case "left-arm":
		paint(0, .08, .38, 1)
		paint(.38, .38, .48, 1)
	case "right-arm":
		paint(.62, .08, 1, 1)
		paint(.52, .38, .62, 1)
	default:
		return fmt.Errorf("unsupported sequence region %q", region)
	}
	dir := filepath.Join(s.dataDir, "inputs", job.ID, "anypaint-mask")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	destination := filepath.Join(dir, "0.png")
	if _, err := os.Stat(destination); err == nil {
		return nil
	} else if !os.IsNotExist(err) {
		return err
	}
	output, err := os.OpenFile(destination, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return err
	}
	encodeErr := png.Encode(output, mask)
	closeErr := output.Close()
	if encodeErr != nil {
		return encodeErr
	}
	return closeErr
}

type generationVideoExecution struct {
	prompt        string
	conditions    []videoConditioningInput
	width, height int
	frames        int
	fps           float64
	seed          int64
}

func (s *Server) loadVideoExecution(job jobs.Job) (generationVideoExecution, error) {
	result := generationVideoExecution{
		prompt: imageStringParam(job.Params, "enhanced_prompt", job.Prompt),
		width:  imageIntParam(job.Params, "width", s.config().Video.DefaultWidth),
		height: imageIntParam(job.Params, "height", s.config().Video.DefaultHeight),
		frames: imageIntParam(job.Params, "num_frames", s.config().Video.DefaultFrames),
		fps:    imageFloatParam(job.Params, "fps", s.config().Video.DefaultFPS),
		seed:   imageInt64Param(job.Params, "seed", -1),
	}
	var saved []savedVideoCondition
	decodeImageParam(job.Params, "video_conditions", &saved)
	if len(saved) == 0 {
		if imageBoolParam(job.Params, "start_image", false) {
			saved = append(saved, savedVideoCondition{Role: "start", FrameIdx: 0, Strength: imageFloatParam(job.Params, "image_strength", 1)})
		}
		count := imageIntParam(job.Params, "keyframes", 0)
		for index := 0; index < count; index++ {
			saved = append(saved, savedVideoCondition{Role: "keyframe", Index: index, FrameIdx: int(float64(result.frames-1) * float64(index+1) / float64(count+1)), Strength: 1})
		}
		if imageBoolParam(job.Params, "end_image", false) {
			saved = append(saved, savedVideoCondition{Role: "end", FrameIdx: result.frames - 1, Strength: 1})
		}
	}
	for _, condition := range saved {
		dir := condition.Role
		if condition.Role == "keyframe" {
			dir = "keyframe-" + strconv.Itoa(condition.Index)
		}
		path, err := s.savedVideoInput(job.ID, dir)
		if err != nil {
			return result, err
		}
		if path == "" {
			return result, fmt.Errorf("saved %s image is missing", condition.Role)
		}
		result.conditions = append(result.conditions, videoConditioningInput{Path: path, FrameIdx: condition.FrameIdx, Strength: condition.Strength, Role: condition.Role})
	}
	return result, nil
}

func (s *Server) loadImageExecution(job jobs.Job) (generationImageExecution, error) {
	result := generationImageExecution{
		prompt:          imageStringParam(job.Params, "enhanced_prompt", job.Prompt),
		width:           imageIntParam(job.Params, "width", s.config().Image.DefaultWidth),
		height:          imageIntParam(job.Params, "height", s.config().Image.DefaultHeight),
		seed:            imageInt64Param(job.Params, "seed", -1),
		mode:            imageStringParam(job.Params, "mode", "create"),
		controlType:     imageStringParam(job.Params, "control_type", "canny"),
		controlStrength: imageFloatParam(job.Params, "control_strength", .65),
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
	for _, role := range []string{"identity", "identity_reference", "identity_mask", "strict_mask", "depth", "vision", "style_reference", "nk2e", "anypaint", "anypaint_mask"} {
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
	if imageIntParam(job.Params, "references", 0) > len(result.references) ||
		imageIntParam(job.Params, "identity_reference_count", 0) > len(paths["identity_reference"]) ||
		imageIntParam(job.Params, "vision_count", 0) > len(paths["vision"]) ||
		imageIntParam(job.Params, "style_reference_count", 0) > len(paths["style_reference"]) {
		return result, fmt.Errorf("one or more saved reference images are missing")
	}
	for _, check := range []struct {
		enabled bool
		role    string
	}{
		{imageBoolParam(job.Params, "identity", false), "identity"},
		{imageBoolParam(job.Params, "identity_reference", false), "identity_reference"},
		{imageBoolParam(job.Params, "identity_mask", false), "identity_mask"},
		{imageBoolParam(job.Params, "strict_mask", false), "strict_mask"},
		{imageBoolParam(job.Params, "depth", false), "depth"},
		{imageBoolParam(job.Params, "nk2e", false), "nk2e"},
		{imageBoolParam(job.Params, "anypaint", false), "anypaint"},
		{imageBoolParam(job.Params, "anypaint_mask", false), "anypaint_mask"},
	} {
		if err := require(check.enabled, check.role); err != nil {
			return result, err
		}
	}
	o := imageGenerationOptions{
		checkpoint:         imageStringParam(job.Params, "checkpoint", "official"),
		identityPreset:     imageStringParam(job.Params, "identity_preset", ""),
		identityAutoPrompt: imageBoolParam(job.Params, "identity_auto_prompt", false),
		identityUserPrompt: imageBoolParam(job.Params, "identity_user_prompt", false),
		identityPath:       firstImagePath(paths["identity"]), identityRefPaths: paths["identity_reference"],
		identityMaskPath: firstImagePath(paths["identity_mask"]), strictMaskPath: firstImagePath(paths["strict_mask"]),
		depthPath: firstImagePath(paths["depth"]), visionPaths: paths["vision"], styleRefPaths: paths["style_reference"],
		nk2ePath: firstImagePath(paths["nk2e"]), anypaintPath: firstImagePath(paths["anypaint"]), anypaintMaskPath: firstImagePath(paths["anypaint_mask"]),
		identityStrength: imageFloatParam(job.Params, "identity_strength", 1), refBoost: imageFloatParam(job.Params, "ref_boost", 4), sourceRefBoost: imageFloatParam(job.Params, "source_ref_boost", 1), groundingPX: imageIntParam(job.Params, "grounding_px", 768),
		steps: imageIntParam(job.Params, "steps", 8), samplingPreset: imageStringParam(job.Params, "sampling_preset", "default"), sampler: imageStringParam(job.Params, "sampler", "euler"), scheduler: imageStringParam(job.Params, "scheduler", "simple"),
		depthStrength: imageFloatParam(job.Params, "depth_strength", .8), depthPrompt: imageStringParam(job.Params, "depth_pose_prompt", ""), preparePoseRef: imageBoolParam(job.Params, "prepare_pose_reference", false), visionMode: imageStringParam(job.Params, "vision_mode", "descriptor"), visionMegapixels: imageFloatParam(job.Params, "vision_megapixels", 1), styleRefStrength: imageFloatParam(job.Params, "style_reference_strength", 1),
		nk2eMode: imageStringParam(job.Params, "nk2e_mode", "edit"), nk2eStrength: imageFloatParam(job.Params, "nk2e_strength", .7), nk2ePreprocessed: imageBoolParam(job.Params, "nk2e_preprocessed", false),
		outpaintLeft: imageIntParam(job.Params, "outpaint_left", 0), outpaintTop: imageIntParam(job.Params, "outpaint_top", 0), outpaintRight: imageIntParam(job.Params, "outpaint_right", 0), outpaintBottom: imageIntParam(job.Params, "outpaint_bottom", 0),
		anypaintStrength: imageFloatParam(job.Params, "anypaint_strength", 1), anypaintBoundary: imageIntParam(job.Params, "anypaint_boundary_redraw_px", 32), strictMaskGrow: imageIntParam(job.Params, "strict_mask_grow", 0), strictMaskFeather: imageFloatParam(job.Params, "strict_mask_feather", 0),
		vaeMode: imageStringParam(job.Params, "vae_mode", "default"), identityFitMode: imageStringParam(job.Params, "identity_fit_mode", "fit"), identityModel: imageStringParam(job.Params, "identity_model", "convrot"), identityEncoder: imageStringParam(job.Params, "identity_encoder", "heretic"), filterMode: imageStringParam(job.Params, "filter_mode", "balanced"), filterStrength: imageFloatParam(job.Params, "filter_strength", 1),
		promptEnhancer: imageBoolParam(job.Params, "prompt_enhancer", false), promptEnhStrength: imageFloatParam(job.Params, "prompt_enhancer_strength", 1), promptTextScale: imageFloatParam(job.Params, "prompt_text_scale", 1.75),
	}
	decodeImageParam(job.Params, "styles", &o.styles)
	decodeImageParam(job.Params, "user_loras", &o.userLoras)
	if len(o.styles) > 0 {
		o.style, o.styleStrength = o.styles[0].Name, o.styles[0].Strength
	}
	result.options = o
	return result, nil
}
