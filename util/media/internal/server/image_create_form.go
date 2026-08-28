package server

import (
	"encoding/json"
	"fmt"
	"mediaapp/internal/config"
	"net/http"
	"strings"
)

type imageSequenceForm struct {
	Planned          bool
	Prompts          []string
	EnhancedPrompts  []string
	Strategies       []string
	SharedPrompt     string
	CanonicalPrompt  string
	Regions          []string
	IdentityStrength float64
	BaseJobID        string
}

type imageCreateForm struct {
	EffectivePrompt        string
	OriginalPrompt         string
	Width                  int
	Height                 int
	Seed                   int64
	Mode                   string
	ControlType            string
	ControlStrength        float64
	ParentJobID            string
	IdentityPreserveItems  []string
	IdentityPreserveCustom string
	Options                imageGenerationOptions
	Sequence               imageSequenceForm
}

func parseImageCreateForm(r *http.Request, cfg config.Config) (imageCreateForm, error) {
	result := imageCreateForm{IdentityPreserveItems: []string{}}
	var err error
	result.Sequence.Prompts, err = parseImageSequencePrompts(r.FormValue("sequence_prompts"))
	if err != nil {
		return result, err
	}
	result.Sequence.Planned = strings.TrimSpace(r.FormValue("sequence_enhanced_prompts")) != ""
	result.Sequence.Regions, err = parseImageSequenceRegions(r.FormValue("sequence_regions"), len(result.Sequence.Prompts))
	if err != nil {
		return result, err
	}
	result.Sequence.EnhancedPrompts, err = parseImageSequenceEnhancedPrompts(r.FormValue("sequence_enhanced_prompts"), result.Sequence.Prompts)
	if err != nil {
		return result, err
	}
	result.Sequence.Strategies, err = parseImageSequenceStrategies(r.FormValue("sequence_strategies"), len(result.Sequence.Prompts))
	if err != nil {
		return result, err
	}
	result.Sequence.SharedPrompt = strings.TrimSpace(r.FormValue("sequence_shared_prompt"))
	if len([]rune(result.Sequence.SharedPrompt)) > 12000 {
		return result, fmt.Errorf("sequence shared prompt is too long")
	}
	result.Sequence.CanonicalPrompt = strings.TrimSpace(r.FormValue("sequence_canonical_prompt"))
	if len([]rune(result.Sequence.CanonicalPrompt)) > 16000 {
		return result, fmt.Errorf("sequence canonical prompt is too long")
	}
	result.Sequence.IdentityStrength = formFloat64(r, "sequence_identity_strength", 0.8)
	if len(result.Sequence.Prompts) > 0 && (result.Sequence.IdentityStrength < 0 || result.Sequence.IdentityStrength > 2) {
		return result, fmt.Errorf("sequence identity strength must be between 0 and 2")
	}
	result.EffectivePrompt = strings.TrimSpace(r.FormValue("prompt"))
	if len(result.Sequence.Prompts) > 0 {
		result.EffectivePrompt = result.Sequence.Prompts[0]
	}
	if result.EffectivePrompt == "" {
		outpaintPadding := formInt(r, "outpaint_left", 0) + formInt(r, "outpaint_top", 0) + formInt(r, "outpaint_right", 0) + formInt(r, "outpaint_bottom", 0)
		hasSource := len(r.MultipartForm.File["anypaint_image"]) > 0 || len(r.MultipartForm.Value["reuse_anypaint_image"]) > 0
		hasMask := len(r.MultipartForm.File["anypaint_mask"]) > 0 || len(r.MultipartForm.Value["reuse_anypaint_mask"]) > 0
		if outpaintPadding > 0 && hasSource && !hasMask {
			result.EffectivePrompt = "Extend the original image naturally into a complete, coherent composition while preserving its subjects, style, lighting, perspective, and visual continuity."
		} else {
			return result, fmt.Errorf("prompt is required")
		}
	}
	result.OriginalPrompt = strings.TrimSpace(r.FormValue("original_prompt"))
	if result.OriginalPrompt == "" {
		result.OriginalPrompt = result.EffectivePrompt
	}
	result.Width = formInt(r, "width", cfg.Image.DefaultWidth)
	result.Height = formInt(r, "height", cfg.Image.DefaultHeight)
	result.Seed = formInt64(r, "seed", -1)
	result.Mode = strings.ToLower(strings.TrimSpace(r.FormValue("mode")))
	if result.Mode == "" {
		result.Mode = cfg.Image.DefaultMode
	}
	if _, ok := cfg.Image.Backends[result.Mode]; !ok {
		return result, fmt.Errorf("unsupported image mode")
	}
	result.ControlType = strings.ToLower(strings.TrimSpace(r.FormValue("control_type")))
	result.ControlStrength = formFloat64(r, "control_strength", 0.65)
	result.ParentJobID = strings.TrimSpace(r.FormValue("parent_job_id"))
	result.Sequence.BaseJobID = strings.TrimSpace(r.FormValue("sequence_base_job_id"))

	result.Options = imageGenerationOptions{
		checkpoint:         strings.ToLower(strings.TrimSpace(r.FormValue("checkpoint"))),
		identityAutoPrompt: strings.EqualFold(r.FormValue("identity_auto_prompt"), "true"),
		identityUserPrompt: strings.EqualFold(r.FormValue("identity_user_prompt"), "true"),
		identityStrength:   formFloat64(r, "identity_strength", 1),
		refBoost:           formFloat64(r, "ref_boost", 4),
		sourceRefBoost:     formFloat64(r, "source_ref_boost", 1),
		groundingPX:        formInt(r, "grounding_px", 768),
		steps:              formInt(r, "steps", 0),
		samplingPreset:     strings.ToLower(strings.TrimSpace(r.FormValue("sampling_preset"))),
		style:              strings.ToLower(strings.TrimSpace(r.FormValue("style"))),
		styleStrength:      formFloat64(r, "style_strength", 1),
		depthStrength:      formFloat64(r, "depth_strength", 0.8),
		depthPrompt:        strings.TrimSpace(r.FormValue("depth_pose_prompt")),
		preparePoseRef:     strings.EqualFold(r.FormValue("prepare_pose_reference"), "true"),
		visionMode:         strings.ToLower(strings.TrimSpace(r.FormValue("vision_mode"))),
		visionMegapixels:   formFloat64(r, "vision_megapixels", 1),
		styleRefStrength:   formFloat64(r, "style_reference_strength", 1),
		nk2eMode:           strings.ToLower(strings.TrimSpace(r.FormValue("nk2e_mode"))),
		nk2eStrength:       formFloat64(r, "nk2e_strength", 0.7),
		outpaintLeft:       formInt(r, "outpaint_left", 0),
		outpaintTop:        formInt(r, "outpaint_top", 0),
		outpaintRight:      formInt(r, "outpaint_right", 0),
		outpaintBottom:     formInt(r, "outpaint_bottom", 0),
		anypaintStrength:   formFloat64(r, "anypaint_strength", 1),
		anypaintBoundary:   formInt(r, "anypaint_boundary_redraw_px", 32),
		strictMaskGrow:     formInt(r, "strict_mask_grow", 0),
		strictMaskFeather:  formFloat64(r, "strict_mask_feather", 0),
		vaeMode:            strings.ToLower(strings.TrimSpace(r.FormValue("vae_mode"))),
		identityFitMode:    strings.ToLower(strings.TrimSpace(r.FormValue("identity_fit_mode"))),
		identityModel:      strings.ToLower(strings.TrimSpace(r.FormValue("identity_model"))),
		identityEncoder:    strings.ToLower(strings.TrimSpace(r.FormValue("identity_encoder"))),
		nk2ePreprocessed:   strings.EqualFold(r.FormValue("nk2e_preprocessed"), "true"),
		filterMode:         strings.ToLower(strings.TrimSpace(r.FormValue("filter_mode"))),
		filterStrength:     formFloat64(r, "filter_strength", 1),
		promptEnhancer:     strings.EqualFold(r.FormValue("prompt_enhancer"), "true"),
		promptEnhStrength:  formFloat64(r, "prompt_enhancer_strength", 1),
		promptTextScale:    formFloat64(r, "prompt_text_scale", 1.75),
	}
	result.Options.identityPreset = strings.TrimSpace(r.FormValue("identity_preset"))
	validIdentityPresets := map[string]bool{"": true, "restage": true, "sheet": true, "faceSwap": true, "headSwap": true, "personSwap": true, "tryon": true, "replace": true}
	if !validIdentityPresets[result.Options.identityPreset] {
		return result, fmt.Errorf("unsupported identity preset")
	}
	if raw := strings.TrimSpace(r.FormValue("identity_preserve_items")); raw != "" {
		if err := json.Unmarshal([]byte(raw), &result.IdentityPreserveItems); err != nil {
			return result, fmt.Errorf("invalid identity preservation selection")
		}
		allowed := map[string]bool{"identity": true, "face": true, "hair": true, "body": true, "clothing": true, "pose": true, "background": true, "lighting": true, "composition": true, "untouched": true}
		for _, item := range result.IdentityPreserveItems {
			if !allowed[item] {
				return result, fmt.Errorf("invalid identity preservation item")
			}
		}
	}
	result.IdentityPreserveCustom = strings.TrimSpace(r.FormValue("identity_preserve_custom"))
	if len(result.IdentityPreserveCustom) > 500 {
		return result, fmt.Errorf("identity custom preservation text is too long")
	}
	if err := normalizeImageGenerationOptions(r, cfg, &result.Options); err != nil {
		return result, err
	}
	return result, nil
}

func normalizeImageGenerationOptions(r *http.Request, cfg config.Config, options *imageGenerationOptions) error {
	if options.checkpoint == "" {
		options.checkpoint = cfg.Image.DefaultCheckpoint
	}
	validCheckpoints := map[string]bool{
		"official": true, "ray-v1": true, "ray-v2": true, "ray-v2-nvfp4": true,
		"ray-v3": true, "ray-v4": true, "ray-v4-nvfp4": true,
		"moody-v7": true, "moody-cutie-v4": true, "moody-amateur-v1": true,
		"chriscole-edit-v1.1": true,
	}
	if !validCheckpoints[options.checkpoint] {
		return fmt.Errorf("unsupported Krea checkpoint")
	}
	if options.vaeMode == "" {
		options.vaeMode = "default"
	}
	if options.identityFitMode == "" {
		options.identityFitMode = "fit"
	}
	if options.identityModel == "" {
		options.identityModel = "convrot"
	}
	if options.identityEncoder == "" {
		options.identityEncoder = "heretic"
	}
	if options.identityModel != "selected" && options.identityModel != "convrot" {
		return fmt.Errorf("identity model must be selected or convrot")
	}
	if options.identityEncoder != "default" && options.identityEncoder != "heretic" {
		return fmt.Errorf("identity encoder must be default or heretic")
	}
	if options.filterMode == "" {
		if options.checkpoint == "official" {
			options.filterMode = "balanced"
		} else {
			options.filterMode = "off"
		}
	}
	if options.checkpoint != "official" && options.filterMode != "off" {
		return fmt.Errorf("third-party checkpoints already include tuning; select original filter mode")
	}
	if options.samplingPreset == "" {
		options.samplingPreset = "default"
	}
	switch options.samplingPreset {
	case "default":
		options.sampler, options.scheduler = "euler", "simple"
	case "detail":
		options.sampler, options.scheduler = "er_sde", "simple"
	case "moody":
		options.sampler, options.scheduler = "euler_ancestral", "beta"
	default:
		return fmt.Errorf("sampling preset must be default, detail, or moody")
	}
	if rawStyles := strings.TrimSpace(r.FormValue("styles")); rawStyles != "" {
		if err := json.Unmarshal([]byte(rawStyles), &options.styles); err != nil {
			return fmt.Errorf("invalid Krea styles")
		}
	} else if options.style != "" {
		options.styles = []styleSelection{{Name: options.style, Strength: options.styleStrength}}
	}
	if rawUserLoras := strings.TrimSpace(r.FormValue("user_loras")); rawUserLoras != "" {
		if err := json.Unmarshal([]byte(rawUserLoras), &options.userLoras); err != nil {
			return fmt.Errorf("invalid user LoRA selection")
		}
	}
	if options.visionMode == "" {
		options.visionMode = "descriptor"
	}
	if options.nk2eMode == "" {
		options.nk2eMode = "edit"
	}
	return nil
}
