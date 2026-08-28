package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"mediaapp/internal/config"
	"os"
)

// buildKreaRequest is shared by runtime preparation and generation. A warm-up
// must exercise the exact checkpoint, encoder, VAE and LoRA graph that the
// following request will use; a generic health probe cannot provide that
// guarantee with ComfyUI's lazy model manager.
func buildKreaRequest(backend config.ImageBackend, prompt string, width, height int, seed int64, krea imageGenerationOptions) (map[string]any, error) {
	request := map[string]any{
		"model": backend.Model, "prompt": prompt,
		"checkpoint": krea.checkpoint,
		"size":       fmt.Sprintf("%dx%d", width, height), "response_format": "b64_json", "output_format": "png",
		"filter_mode": krea.filterMode, "filter_strength": krea.filterStrength,
		"prompt_enhancer": krea.promptEnhancer, "prompt_enhancer_strength": krea.promptEnhStrength,
		"prompt_text_scale": krea.promptTextScale,
		"sampler_name":      krea.sampler, "scheduler": krea.scheduler,
	}
	for field, path := range map[string]string{
		"source_image": krea.identityPath, "reid_image": krea.reidPath, "control_image": krea.depthPath, "nk2e_image": krea.nk2ePath,
		"identity_mask": krea.identityMaskPath, "strict_mask": krea.strictMaskPath,
		"anypaint_image": krea.anypaintPath, "anypaint_mask": krea.anypaintMaskPath,
	} {
		if path == "" {
			continue
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		request[field] = base64.StdEncoding.EncodeToString(data)
	}
	for field, paths := range map[string][]string{
		"reference_images": krea.identityRefPaths, "vision_images": krea.visionPaths, "style_reference_images": krea.styleRefPaths,
	} {
		encoded := make([]string, 0, len(paths))
		for _, path := range paths {
			data, err := os.ReadFile(path)
			if err != nil {
				return nil, err
			}
			encoded = append(encoded, base64.StdEncoding.EncodeToString(data))
		}
		if len(encoded) > 0 {
			request[field] = encoded
		}
	}
	if krea.identityPath != "" {
		request["identity_strength"] = krea.identityStrength
		request["ref_boost"] = krea.refBoost
		request["source_ref_boost"] = krea.sourceRefBoost
		request["grounding_px"] = krea.groundingPX
		request["strict_mask_grow"] = krea.strictMaskGrow
		request["strict_mask_feather"] = krea.strictMaskFeather
		request["vae_mode"] = krea.vaeMode
		request["identity_fit_mode"] = krea.identityFitMode
		request["identity_model"] = krea.identityModel
		request["identity_encoder"] = krea.identityEncoder
	}
	if krea.depthPath != "" {
		request["control_strength"] = krea.depthStrength
		request["control_prompt"] = krea.depthPrompt
		request["prepare_pose_reference"] = krea.preparePoseRef
	}
	if len(krea.styles) > 0 {
		request["styles"] = krea.styles
		request["style"] = krea.styles[0].Name
		request["style_strength"] = krea.styles[0].Strength
	}
	if len(krea.userLoras) > 0 {
		request["user_loras"] = krea.userLoras
	}
	if len(krea.visionPaths) > 0 {
		request["vision_mode"] = krea.visionMode
		request["vision_megapixels"] = krea.visionMegapixels
	}
	if len(krea.styleRefPaths) > 0 {
		request["style_reference_strength"] = krea.styleRefStrength
	}
	if krea.nk2ePath != "" {
		request["nk2e_mode"] = krea.nk2eMode
		request["nk2e_strength"] = krea.nk2eStrength
		request["nk2e_preprocessed"] = krea.nk2ePreprocessed
	}
	if krea.anypaintPath != "" {
		request["outpaint_left"] = krea.outpaintLeft
		request["outpaint_top"] = krea.outpaintTop
		request["outpaint_right"] = krea.outpaintRight
		request["outpaint_bottom"] = krea.outpaintBottom
		request["anypaint_strength"] = krea.anypaintStrength
		request["anypaint_boundary_redraw_px"] = krea.anypaintBoundary
		request["anypaint_reference_max_edge"] = 384
		request["anypaint_vlm_reference"] = true
	}
	if krea.steps > 0 {
		request["steps"] = krea.steps
	}
	if seed >= 0 {
		request["seed"] = seed
	}
	return request, nil
}

// prepareKreaCreate runs a single-step version of the exact upcoming graph.
// The engine discards the image and records the loaded runtime signature.
func (s *Server) prepareKreaCreate(ctx context.Context, backend config.ImageBackend, operationID, prompt string, width, height int, seed int64, krea imageGenerationOptions, profile string) (map[string]any, error) {
	request, err := buildKreaRequest(backend, prompt, width, height, seed, krea)
	if err != nil {
		return nil, err
	}
	request["steps"] = 1
	request["prepare_only"] = true
	request["runtime_profile"] = profile
	request["operation_id"] = operationID
	data, err := s.generateImageWithEngine(ctx, backend, request)
	if err != nil {
		return nil, err
	}
	result := map[string]any{}
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("invalid Krea preparation response: %w", err)
	}
	return result, nil
}

// generateKreaCreate performs one Krea create/edit request. Keeping request
// construction here lets sequence strategies compose multiple passes without
// duplicating the public job/result lifecycle in runImage.
func (s *Server) generateKreaCreate(ctx context.Context, backend config.ImageBackend, operationID, prompt string, width, height int, seed int64, krea imageGenerationOptions) ([]byte, error) {
	request, err := buildKreaRequest(backend, prompt, width, height, seed, krea)
	if err != nil {
		return nil, err
	}
	request["operation_id"] = operationID
	return s.generateImageWithEngine(ctx, backend, request)
}
