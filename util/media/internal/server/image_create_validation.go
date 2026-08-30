package server

import (
	"fmt"
	"path/filepath"
	"strings"
)

func validateImageCreate(mode string, references []string, width, height int, sequencePrompts []string, controlType *string, controlStrength float64, options *imageGenerationOptions) error {
	switch mode {
	case "create":
		if len(references) != 0 {
			return fmt.Errorf("high quality generation does not accept reference images")
		}
		if options.steps == 0 {
			options.steps = 8
			if options.identityPath != "" {
				options.steps = 10
			}
		}
		if options.reidPath != "" && len(sequencePrompts) == 0 {
			return fmt.Errorf("character ReID is only available for multi-scene generation")
		}
		if options.reidPath != "" && !isOfficialKreaCheckpoint(options.checkpoint) {
			return fmt.Errorf("character ReID currently requires the official Krea checkpoint")
		}
		if len(options.identityRefPaths) > 0 && options.identityPath == "" {
			return fmt.Errorf("a primary identity image is required before an additional reference")
		}
		if (options.identityMaskPath != "" || options.strictMaskPath != "") && options.identityPath == "" {
			return fmt.Errorf("identity masks require a primary identity image")
		}
		if options.strictMaskGrow < 0 || options.strictMaskGrow > 128 || options.strictMaskFeather < 0 || options.strictMaskFeather > 128 {
			return fmt.Errorf("strict mask grow and feather must be between 0 and 128")
		}
		if options.vaeMode != "default" && options.vaeMode != "wan" && options.vaeMode != "real" {
			return fmt.Errorf("VAE mode must be default, wan, or real")
		}
		if options.identityFitMode != "fit" && options.identityFitMode != "crop" {
			return fmt.Errorf("identity fit mode must be fit or crop")
		}
		if options.filterMode != "off" && options.filterMode != "adherence" && options.filterMode != "balanced" && options.filterMode != "strong" {
			return fmt.Errorf("filter mode must be off, adherence, balanced, or strong")
		}
		if options.filterStrength < 0 || options.filterStrength > 10 || options.promptEnhStrength < 0 || options.promptEnhStrength > 2 || options.promptTextScale < 0.25 || options.promptTextScale > 4 {
			return fmt.Errorf("invalid Krea filter or prompt adherence settings")
		}
		if err := validateImageStyles(options.styles); err != nil {
			return err
		}
		if err := validateImageUserLoRAs(options.userLoras); err != nil {
			return err
		}
		if options.identityStrength < 0 || options.identityStrength > 2 || options.depthStrength < 0 || options.depthStrength > 2 || options.styleRefStrength < 0 || options.styleRefStrength > 2 || options.nk2eStrength < 0 || options.nk2eStrength > 2 || options.anypaintStrength < 0 || options.anypaintStrength > 2 {
			return fmt.Errorf("Krea module strength must be between 0 and 2")
		}
		if options.nk2eMode != "edit" && options.nk2eMode != "canny" {
			return fmt.Errorf("NK2E mode must be edit or canny")
		}
		if options.refBoost < 0 || options.refBoost > 20 || options.groundingPX < 384 || options.groundingPX > 1024 {
			return fmt.Errorf("invalid Krea identity fidelity settings")
		}
		if options.steps < 1 || options.steps > 20 {
			return fmt.Errorf("Krea steps must be between 1 and 20")
		}
		if options.visionMode != "descriptor" && options.visionMode != "instruct" {
			return fmt.Errorf("vision mode must be descriptor or instruct")
		}
		if options.visionMegapixels < 0.1 || options.visionMegapixels > 4 {
			return fmt.Errorf("vision megapixels must be between 0.1 and 4")
		}
		if len(options.styleRefPaths) > 0 && (len(options.visionPaths) > 0 || options.identityPath != "" || options.depthPath != "" || len(options.styles) > 0 || len(options.userLoras) > 0) {
			return fmt.Errorf("style reference cannot be combined with other Krea modules yet")
		}
		if len(options.visionPaths) > 0 && options.identityPath != "" {
			return fmt.Errorf("vision reference cannot be combined with identity yet")
		}
		if options.nk2ePath != "" && (options.identityPath != "" || options.depthPath != "" || len(options.styles) > 0 || len(options.userLoras) > 0 || len(options.visionPaths) > 0 || len(options.styleRefPaths) > 0) {
			return fmt.Errorf("NK2E cannot be combined with other Krea modules yet")
		}
		if options.anypaintMaskPath != "" && options.anypaintPath == "" {
			return fmt.Errorf("AnyPaint mask requires a source image")
		}
		if err := validateAnyPaintOptions(options); err != nil {
			return err
		}
		if (options.identityPath != "" || options.reidPath != "") && width*height > 2*1024*1024 {
			return fmt.Errorf("Krea Identity Edit output must not exceed 2 megapixels")
		}
		if len(sequencePrompts) > 0 {
			if options.identityPath != "" || options.depthPath != "" || len(options.visionPaths) > 0 || len(options.styleRefPaths) > 0 || options.nk2ePath != "" || options.anypaintPath != "" {
				return fmt.Errorf("sequence generation cannot be combined with reference, depth, vision, structure, or partial-edit modules")
			}
		}
	case "edit":
		if len(references) == 0 {
			return fmt.Errorf("reference editing requires at least one image")
		}
	case "control":
		if len(references) != 1 {
			return fmt.Errorf("structure control requires exactly one image")
		}
		if *controlType == "" {
			*controlType = "canny"
		}
		if *controlType != "canny" {
			return fmt.Errorf("only canny control is currently available")
		}
		if controlStrength < 0 || controlStrength > 2 {
			return fmt.Errorf("control strength must be between 0 and 2")
		}
	}
	return nil
}

func validateImageStyles(styles []styleSelection) error {
	valid := map[string]bool{"darkbrush": true, "dotmatrix": true, "kidsdrawing": true, "neondrip": true, "rainywindow": true, "retroanime": true, "softwatercolor": true, "sunsetblur": true, "vintagetarot": true}
	seen := make(map[string]bool, len(styles))
	for _, style := range styles {
		if !valid[style.Name] || seen[style.Name] || style.Strength < 0 || style.Strength > 2 {
			return fmt.Errorf("invalid Krea style LoRA selection")
		}
		seen[style.Name] = true
	}
	if len(styles) > len(valid) {
		return fmt.Errorf("too many Krea style LoRAs")
	}
	return nil
}

func validateImageUserLoRAs(selections []userLoRASelection) error {
	seen := make(map[string]bool, len(selections))
	for _, selection := range selections {
		if selection.Filename == "" || filepath.Base(selection.Filename) != selection.Filename || !strings.HasSuffix(strings.ToLower(selection.Filename), ".safetensors") || seen[selection.Filename] || selection.Strength < -2 || selection.Strength > 2 {
			return fmt.Errorf("invalid user LoRA selection")
		}
		seen[selection.Filename] = true
	}
	if len(selections) > 5 {
		return fmt.Errorf("at most five user LoRAs may be stacked")
	}
	return nil
}

func validateAnyPaintOptions(options *imageGenerationOptions) error {
	if options.anypaintPath == "" {
		return nil
	}
	if options.identityPath != "" || options.depthPath != "" || len(options.styles) > 0 || len(options.userLoras) > 0 || len(options.visionPaths) > 0 || len(options.styleRefPaths) > 0 || options.nk2ePath != "" {
		return fmt.Errorf("AnyPaint cannot be combined with other Krea modules yet")
	}
	for _, padding := range []int{options.outpaintLeft, options.outpaintTop, options.outpaintRight, options.outpaintBottom} {
		if padding < 0 || padding > 1536 || padding%16 != 0 {
			return fmt.Errorf("outpaint padding must be 0..1536 in multiples of 16")
		}
	}
	if options.anypaintMaskPath == "" && options.outpaintLeft+options.outpaintTop+options.outpaintRight+options.outpaintBottom == 0 {
		return fmt.Errorf("AnyPaint requires a mask or at least one expansion direction")
	}
	if options.anypaintBoundary < 0 || options.anypaintBoundary > 256 {
		return fmt.Errorf("AnyPaint boundary redraw must be between 0 and 256")
	}
	return nil
}
