package server

import "encoding/json"

// imageJobParams is the persisted contract between the HTTP creation layer and
// the generation queue. It deliberately keeps the existing flat JSON shape so
// older jobs and the web client remain compatible.
type imageJobParams struct {
	Width      int    `json:"width"`
	Height     int    `json:"height"`
	Seed       int64  `json:"seed"`
	References int    `json:"references"`
	Mode       string `json:"mode"`
	Model      string `json:"model"`

	ParentJobID     string  `json:"parent_job_id,omitempty"`
	ControlType     string  `json:"control_type,omitempty"`
	ControlStrength float64 `json:"control_strength,omitempty"`

	Checkpoint             string              `json:"checkpoint,omitempty"`
	Identity               bool                `json:"identity,omitempty"`
	IdentityReference      bool                `json:"identity_reference,omitempty"`
	IdentityReferenceCount int                 `json:"identity_reference_count,omitempty"`
	IdentityPreset         string              `json:"identity_preset,omitempty"`
	IdentityAutoPrompt     bool                `json:"identity_auto_prompt,omitempty"`
	IdentityUserPrompt     bool                `json:"identity_user_prompt,omitempty"`
	IdentityPreserveItems  []string            `json:"identity_preserve_items,omitempty"`
	IdentityPreserveCustom string              `json:"identity_preserve_custom,omitempty"`
	IdentityMask           bool                `json:"identity_mask,omitempty"`
	StrictMask             bool                `json:"strict_mask,omitempty"`
	StrictMaskGrow         int                 `json:"strict_mask_grow,omitempty"`
	StrictMaskFeather      float64             `json:"strict_mask_feather,omitempty"`
	VAEMode                string              `json:"vae_mode,omitempty"`
	IdentityFitMode        string              `json:"identity_fit_mode,omitempty"`
	IdentityModel          string              `json:"identity_model,omitempty"`
	IdentityEncoder        string              `json:"identity_encoder,omitempty"`
	IdentityStrength       float64             `json:"identity_strength,omitempty"`
	ReferenceBoost         float64             `json:"ref_boost,omitempty"`
	SourceReferenceBoost   float64             `json:"source_ref_boost,omitempty"`
	GroundingPixels        int                 `json:"grounding_px,omitempty"`
	Depth                  bool                `json:"depth,omitempty"`
	DepthPosePrompt        string              `json:"depth_pose_prompt,omitempty"`
	PreparePoseReference   bool                `json:"prepare_pose_reference,omitempty"`
	DepthStrength          float64             `json:"depth_strength,omitempty"`
	Style                  string              `json:"style,omitempty"`
	StyleStrength          float64             `json:"style_strength,omitempty"`
	Styles                 []styleSelection    `json:"styles,omitempty"`
	UserLoRAs              []userLoRASelection `json:"user_loras,omitempty"`
	Vision                 bool                `json:"vision,omitempty"`
	VisionCount            int                 `json:"vision_count,omitempty"`
	VisionMode             string              `json:"vision_mode,omitempty"`
	VisionMegapixels       float64             `json:"vision_megapixels,omitempty"`
	StyleReference         bool                `json:"style_reference,omitempty"`
	StyleReferenceCount    int                 `json:"style_reference_count,omitempty"`
	StyleReferenceStrength float64             `json:"style_reference_strength,omitempty"`
	NK2E                   bool                `json:"nk2e,omitempty"`
	NK2EMode               string              `json:"nk2e_mode,omitempty"`
	NK2EStrength           float64             `json:"nk2e_strength,omitempty"`
	NK2EPreprocessed       bool                `json:"nk2e_preprocessed,omitempty"`
	AnyPaint               bool                `json:"anypaint,omitempty"`
	AnyPaintMask           bool                `json:"anypaint_mask,omitempty"`
	OutpaintLeft           int                 `json:"outpaint_left,omitempty"`
	OutpaintTop            int                 `json:"outpaint_top,omitempty"`
	OutpaintRight          int                 `json:"outpaint_right,omitempty"`
	OutpaintBottom         int                 `json:"outpaint_bottom,omitempty"`
	AnyPaintStrength       float64             `json:"anypaint_strength,omitempty"`
	AnyPaintBoundary       int                 `json:"anypaint_boundary_redraw_px,omitempty"`
	FilterMode             string              `json:"filter_mode,omitempty"`
	FilterStrength         float64             `json:"filter_strength,omitempty"`
	PromptEnhancer         bool                `json:"prompt_enhancer,omitempty"`
	PromptEnhancerStrength float64             `json:"prompt_enhancer_strength,omitempty"`
	PromptTextScale        float64             `json:"prompt_text_scale,omitempty"`
	SamplingPreset         string              `json:"sampling_preset,omitempty"`
	Sampler                string              `json:"sampler,omitempty"`
	Scheduler              string              `json:"scheduler,omitempty"`
	Steps                  int                 `json:"steps,omitempty"`

	EnhancedPrompt           string  `json:"enhanced_prompt,omitempty"`
	Stage                    string  `json:"stage,omitempty"`
	QueuedAt                 string  `json:"queued_at,omitempty"`
	SequenceID               string  `json:"sequence_id,omitempty"`
	SequenceIndex            int     `json:"sequence_index,omitempty"`
	SequenceTotal            int     `json:"sequence_total,omitempty"`
	SequenceIdentityStrength float64 `json:"sequence_identity_strength,omitempty"`
	SequencePreviousJobID    string  `json:"sequence_previous_job_id,omitempty"`
	SequenceRegion           string  `json:"sequence_region,omitempty"`
}

func newImageJobParams() imageJobParams {
	return imageJobParams{
		Seed:                     -1,
		Mode:                     "create",
		ControlType:              "canny",
		ControlStrength:          0.65,
		Checkpoint:               "official",
		IdentityStrength:         1,
		ReferenceBoost:           4,
		SourceReferenceBoost:     1,
		GroundingPixels:          768,
		DepthStrength:            0.8,
		VisionMode:               "descriptor",
		VisionMegapixels:         1,
		StyleReferenceStrength:   1,
		NK2EMode:                 "edit",
		NK2EStrength:             0.7,
		AnyPaintStrength:         1,
		AnyPaintBoundary:         32,
		VAEMode:                  "default",
		IdentityFitMode:          "fit",
		IdentityModel:            "convrot",
		IdentityEncoder:          "heretic",
		FilterMode:               "balanced",
		FilterStrength:           1,
		PromptEnhancerStrength:   1,
		PromptTextScale:          1.75,
		SamplingPreset:           "default",
		Sampler:                  "euler",
		Scheduler:                "simple",
		Steps:                    8,
		SequenceIdentityStrength: 0.8,
		SequenceRegion:           "all",
	}
}

func decodeImageJobParams(values map[string]any) imageJobParams {
	result := newImageJobParams()
	data, err := json.Marshal(values)
	if err == nil {
		_ = json.Unmarshal(data, &result)
	}
	return result
}

func (p imageJobParams) toMap() map[string]any {
	result := map[string]any{
		"width": p.Width, "height": p.Height, "seed": p.Seed,
		"references": p.References, "mode": p.Mode, "model": p.Model,
	}
	if p.ParentJobID != "" {
		result["parent_job_id"] = p.ParentJobID
	}
	if p.Mode == "control" {
		result["control_type"] = p.ControlType
		result["control_strength"] = p.ControlStrength
	}
	if p.Mode == "create" {
		result["checkpoint"] = p.Checkpoint
		result["identity"] = p.Identity
		result["identity_reference"] = p.IdentityReference
		result["identity_reference_count"] = p.IdentityReferenceCount
		result["identity_preset"] = p.IdentityPreset
		result["identity_auto_prompt"] = p.IdentityAutoPrompt
		result["identity_user_prompt"] = p.IdentityUserPrompt
		result["identity_preserve_items"] = p.IdentityPreserveItems
		result["identity_preserve_custom"] = p.IdentityPreserveCustom
		result["depth"] = p.Depth
		if p.DepthPosePrompt != "" {
			result["depth_pose_prompt"] = p.DepthPosePrompt
		}
		result["prepare_pose_reference"] = p.PreparePoseReference
		result["style"] = p.Style
		result["styles"] = p.Styles
		result["user_loras"] = p.UserLoRAs
		result["vision"] = p.Vision
		result["vision_count"] = p.VisionCount
		result["style_reference"] = p.StyleReference
		result["style_reference_count"] = p.StyleReferenceCount
		result["nk2e"] = p.NK2E
		result["anypaint"] = p.AnyPaint
		result["identity_mask"] = p.IdentityMask
		result["strict_mask"] = p.StrictMask
		result["vae_mode"] = p.VAEMode
		result["identity_fit_mode"] = p.IdentityFitMode
		result["identity_model"] = p.IdentityModel
		result["identity_encoder"] = p.IdentityEncoder
		result["strict_mask_grow"] = p.StrictMaskGrow
		result["strict_mask_feather"] = p.StrictMaskFeather
		result["filter_mode"] = p.FilterMode
		result["filter_strength"] = p.FilterStrength
		result["prompt_enhancer"] = p.PromptEnhancer
		result["prompt_enhancer_strength"] = p.PromptEnhancerStrength
		result["prompt_text_scale"] = p.PromptTextScale
		result["sampling_preset"] = p.SamplingPreset
		result["sampler"] = p.Sampler
		result["scheduler"] = p.Scheduler
		result["steps"] = p.Steps
		if p.Vision {
			result["vision_mode"] = p.VisionMode
			result["vision_megapixels"] = p.VisionMegapixels
		}
		if p.StyleReference {
			result["style_reference_strength"] = p.StyleReferenceStrength
		}
		if p.NK2E {
			result["nk2e_mode"] = p.NK2EMode
			result["nk2e_strength"] = p.NK2EStrength
			result["nk2e_preprocessed"] = p.NK2EPreprocessed
		}
		if p.AnyPaint {
			result["anypaint_mask"] = p.AnyPaintMask
			result["outpaint_left"] = p.OutpaintLeft
			result["outpaint_top"] = p.OutpaintTop
			result["outpaint_right"] = p.OutpaintRight
			result["outpaint_bottom"] = p.OutpaintBottom
			result["anypaint_strength"] = p.AnyPaintStrength
			result["anypaint_boundary_redraw_px"] = p.AnyPaintBoundary
		}
		if p.Identity {
			result["identity_strength"] = p.IdentityStrength
			result["ref_boost"] = p.ReferenceBoost
			result["source_ref_boost"] = p.SourceReferenceBoost
			result["grounding_px"] = p.GroundingPixels
		}
		if p.Depth {
			result["depth_strength"] = p.DepthStrength
		}
		if len(p.Styles) > 0 {
			result["style"] = p.Styles[0].Name
			result["style_strength"] = p.Styles[0].Strength
		}
	}
	result["enhanced_prompt"] = p.EnhancedPrompt
	result["stage"] = p.Stage
	result["queued_at"] = p.QueuedAt
	if p.SequenceID != "" {
		result["sequence_id"] = p.SequenceID
		result["sequence_index"] = p.SequenceIndex
		result["sequence_total"] = p.SequenceTotal
		result["sequence_identity_strength"] = p.SequenceIdentityStrength
	}
	if p.SequencePreviousJobID != "" {
		result["sequence_previous_job_id"] = p.SequencePreviousJobID
	}
	if p.SequenceRegion != "" && p.SequencePreviousJobID != "" {
		result["sequence_region"] = p.SequenceRegion
	}
	return result
}

func imageJobParamsFromOptions(width, height int, seed int64, references int, mode, model, controlType string, controlStrength float64, options imageGenerationOptions) imageJobParams {
	p := newImageJobParams()
	p.Width, p.Height, p.Seed = width, height, seed
	p.References, p.Mode, p.Model = references, mode, model
	p.ControlType, p.ControlStrength = controlType, controlStrength
	p.Checkpoint = options.checkpoint
	p.Identity = options.identityPath != ""
	p.IdentityReference = len(options.identityRefPaths) > 0
	p.IdentityReferenceCount = len(options.identityRefPaths)
	p.IdentityPreset = options.identityPreset
	p.IdentityAutoPrompt = options.identityAutoPrompt
	p.IdentityUserPrompt = options.identityUserPrompt
	p.IdentityMask = options.identityMaskPath != ""
	p.StrictMask = options.strictMaskPath != ""
	p.StrictMaskGrow, p.StrictMaskFeather = options.strictMaskGrow, options.strictMaskFeather
	p.VAEMode, p.IdentityFitMode = options.vaeMode, options.identityFitMode
	p.IdentityModel, p.IdentityEncoder = options.identityModel, options.identityEncoder
	p.IdentityStrength = options.identityStrength
	p.ReferenceBoost, p.SourceReferenceBoost = options.refBoost, options.sourceRefBoost
	p.GroundingPixels = options.groundingPX
	p.Depth, p.DepthPosePrompt = options.depthPath != "", options.depthPrompt
	p.PreparePoseReference, p.DepthStrength = options.preparePoseRef, options.depthStrength
	p.Style, p.StyleStrength = options.style, options.styleStrength
	p.Styles, p.UserLoRAs = options.styles, options.userLoras
	p.Vision, p.VisionCount = len(options.visionPaths) > 0, len(options.visionPaths)
	p.VisionMode, p.VisionMegapixels = options.visionMode, options.visionMegapixels
	p.StyleReference, p.StyleReferenceCount = len(options.styleRefPaths) > 0, len(options.styleRefPaths)
	p.StyleReferenceStrength = options.styleRefStrength
	p.NK2E, p.NK2EMode = options.nk2ePath != "", options.nk2eMode
	p.NK2EStrength, p.NK2EPreprocessed = options.nk2eStrength, options.nk2ePreprocessed
	p.AnyPaint, p.AnyPaintMask = options.anypaintPath != "", options.anypaintMaskPath != ""
	p.OutpaintLeft, p.OutpaintTop = options.outpaintLeft, options.outpaintTop
	p.OutpaintRight, p.OutpaintBottom = options.outpaintRight, options.outpaintBottom
	p.AnyPaintStrength, p.AnyPaintBoundary = options.anypaintStrength, options.anypaintBoundary
	p.FilterMode, p.FilterStrength = options.filterMode, options.filterStrength
	p.PromptEnhancer, p.PromptEnhancerStrength = options.promptEnhancer, options.promptEnhStrength
	p.PromptTextScale = options.promptTextScale
	p.SamplingPreset, p.Sampler, p.Scheduler = options.samplingPreset, options.sampler, options.scheduler
	p.Steps = options.steps
	if len(p.Styles) > 0 {
		p.Style, p.StyleStrength = p.Styles[0].Name, p.Styles[0].Strength
	}
	return p
}

func (p imageJobParams) generationOptions(paths map[string][]string) imageGenerationOptions {
	options := imageGenerationOptions{
		checkpoint:         p.Checkpoint,
		identityPreset:     p.IdentityPreset,
		identityAutoPrompt: p.IdentityAutoPrompt,
		identityUserPrompt: p.IdentityUserPrompt,
		identityPath:       firstImagePath(paths["identity"]),
		identityRefPaths:   paths["identity_reference"],
		identityMaskPath:   firstImagePath(paths["identity_mask"]),
		strictMaskPath:     firstImagePath(paths["strict_mask"]),
		depthPath:          firstImagePath(paths["depth"]),
		visionPaths:        paths["vision"],
		styleRefPaths:      paths["style_reference"],
		nk2ePath:           firstImagePath(paths["nk2e"]),
		anypaintPath:       firstImagePath(paths["anypaint"]),
		anypaintMaskPath:   firstImagePath(paths["anypaint_mask"]),
		identityStrength:   p.IdentityStrength,
		refBoost:           p.ReferenceBoost,
		sourceRefBoost:     p.SourceReferenceBoost,
		groundingPX:        p.GroundingPixels,
		steps:              p.Steps,
		samplingPreset:     p.SamplingPreset,
		sampler:            p.Sampler,
		scheduler:          p.Scheduler,
		depthStrength:      p.DepthStrength,
		depthPrompt:        p.DepthPosePrompt,
		preparePoseRef:     p.PreparePoseReference,
		visionMode:         p.VisionMode,
		visionMegapixels:   p.VisionMegapixels,
		styleRefStrength:   p.StyleReferenceStrength,
		nk2eMode:           p.NK2EMode,
		nk2eStrength:       p.NK2EStrength,
		nk2ePreprocessed:   p.NK2EPreprocessed,
		outpaintLeft:       p.OutpaintLeft,
		outpaintTop:        p.OutpaintTop,
		outpaintRight:      p.OutpaintRight,
		outpaintBottom:     p.OutpaintBottom,
		anypaintStrength:   p.AnyPaintStrength,
		anypaintBoundary:   p.AnyPaintBoundary,
		strictMaskGrow:     p.StrictMaskGrow,
		strictMaskFeather:  p.StrictMaskFeather,
		vaeMode:            p.VAEMode,
		identityFitMode:    p.IdentityFitMode,
		identityModel:      p.IdentityModel,
		identityEncoder:    p.IdentityEncoder,
		filterMode:         p.FilterMode,
		filterStrength:     p.FilterStrength,
		promptEnhancer:     p.PromptEnhancer,
		promptEnhStrength:  p.PromptEnhancerStrength,
		promptTextScale:    p.PromptTextScale,
		styles:             p.Styles,
		userLoras:          p.UserLoRAs,
	}
	if len(options.styles) > 0 {
		options.style, options.styleStrength = options.styles[0].Name, options.styles[0].Strength
	}
	return options
}
