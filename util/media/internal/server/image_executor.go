package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"image"
	"mediaapp/internal/jobs"
	"os"
	"strconv"
	"strings"
)

func (s *Server) runImage(ctx context.Context, j jobs.Job, effectivePrompt string, refs []string, width, height int, seed int64, mode, controlType string, controlStrength float64, krea imageGenerationOptions) {
	cfg := s.config()
	s.heavyMu.Lock()
	defer s.heavyMu.Unlock()
	if krea.identityAutoPrompt && krea.identityPreset == "tryon" && len(krea.identityRefPaths) > 0 {
		outfit, describeErr := s.describeOutfitReference(krea.identityRefPaths[0])
		if describeErr != nil {
			s.fail(j, fmt.Errorf("outfit reference description: %w", describeErr))
			return
		}
		pronoun, describeErr := s.describeSubjectPronoun(krea.identityPath)
		if describeErr != nil {
			s.fail(j, fmt.Errorf("source subject description: %w", describeErr))
			return
		}
		outfit = strings.TrimSpace(strings.TrimSuffix(outfit, "."))
		lowerOutfit := strings.ToLower(outfit)
		switch {
		case strings.HasPrefix(lowerOutfit, "an "):
			outfit = "the " + strings.TrimSpace(outfit[3:])
		case strings.HasPrefix(lowerOutfit, "a "):
			outfit = "the " + strings.TrimSpace(outfit[2:])
		}
		pronoun = strings.ToLower(strings.TrimSpace(strings.Trim(pronoun, ".,;:!?\"'")))
		if pronoun != "she" && pronoun != "he" && pronoun != "they" && pronoun != "it" {
			pronoun = "they"
		}
		verb := "is"
		if pronoun == "they" {
			verb = "are"
		}
		// Preserve the lowercase training-style wording used by the published
		// Identity Edit workflow. With otherwise identical inputs and seed,
		// capitalizing this leading pronoun caused the source shirt to survive.
		modulePrompt := pronoun + " " + verb + " now wearing " + outfit + "."
		hasExtraUserPrompt := krea.identityUserPrompt && strings.TrimSpace(effectivePrompt) != "" && !isTryOnModuleFallback(j.Prompt, krea.depthPath != "")
		if hasExtraUserPrompt {
			if composed, composeErr := s.composeIdentityEditPrompt(pronoun, verb, outfit, effectivePrompt, krea.depthPath != ""); composeErr == nil {
				modulePrompt = composed
			} else {
				modulePrompt += "\n" + strings.TrimSpace(effectivePrompt)
				if krea.depthPath != "" {
					modulePrompt += "\n" + pronoun + " now holds the same pose."
				}
			}
		} else if krea.depthPath != "" {
			modulePrompt += "\n" + pronoun + " now holds the same pose."
		}
		effectivePrompt = modulePrompt
		j.Params["generated_edit_prompt"] = effectivePrompt
		_ = s.jobs.Save(j)
	}
	if krea.identityPath != "" && krea.depthPath != "" && krea.depthPrompt == "" && cfg.PromptEnhancement.VisionEnabled {
		poseDescription, describeErr := s.describePoseReference(krea.depthPath)
		if describeErr != nil {
			s.fail(j, fmt.Errorf("pose reference description: %w", describeErr))
			return
		}
		krea.depthPrompt = poseDescription
		j.Params["depth_pose_prompt"] = poseDescription
		_ = s.jobs.Save(j)
	}
	backend := cfg.Image.Backends[mode]
	observer := s.startRuntimeObserver(ctx, j.ID, backend.Endpoint)
	var response []byte
	var err error
	if mode == "control" {
		controlImage, readErr := os.ReadFile(refs[0])
		if readErr != nil {
			s.fail(j, readErr)
			return
		}
		request := map[string]any{
			"model": backend.Model, "prompt": effectivePrompt,
			"checkpoint": krea.checkpoint,
			"size":       fmt.Sprintf("%dx%d", width, height), "response_format": "b64_json", "output_format": "png",
			"control_image":    base64.StdEncoding.EncodeToString(controlImage),
			"operation_id":     j.ID,
			"control_strength": controlStrength, "control_strategy": "split4", "control_type": controlType,
		}
		if seed >= 0 {
			request["seed"] = seed
		}
		response, err = s.generateImageWithEngine(ctx, backend, request)
	} else if mode == "edit" {
		fields := map[string]string{
			"model": backend.Model, "prompt": effectivePrompt,
			"size": fmt.Sprintf("%dx%d", width, height), "response_format": "b64_json", "output_format": "png",
			"operation_id": j.ID,
		}
		if seed >= 0 {
			fields["seed"] = strconv.FormatInt(seed, 10)
		}
		response, err = s.editImageWithEngine(ctx, backend, fields, refs)
	} else {
		response, err = s.generateKreaCreate(ctx, backend, j.ID, effectivePrompt, width, height, seed, krea)
	}
	observer.Stop()
	if err != nil {
		if s.requeueGenerationAfterEngineConflict(j, err) {
			return
		}
		s.fail(j, err)
		return
	}
	if actualSeed, ok := decodeImageSeed(response); ok {
		j.Params["seed"] = actualSeed
	}
	data, err := decodeImage(response)
	if err != nil {
		s.fail(j, err)
		return
	}
	if err = s.writeImageResult(&j, data, effectivePrompt); err != nil {
		s.fail(j, err)
		return
	}
}

func isTryOnModuleFallback(prompt string, withPose bool) bool {
	expected := "Use the complete outfit shown in the supporting clothing reference"
	if withPose {
		expected += ". Apply the pose, body orientation, framing, and camera viewpoint from the pose reference"
	}
	return strings.EqualFold(strings.TrimSpace(prompt), expected)
}

func (s *Server) writeImageResult(j *jobs.Job, data []byte, effectivePrompt string) error {
	if output, _, err := image.DecodeConfig(bytes.NewReader(data)); err == nil {
		j.Params["width"] = output.Width
		j.Params["height"] = output.Height
	}
	data = embedImageEXIF(data, metadataForImageJob(*j, effectivePrompt, s.config().ImageMetadata))
	name := j.ID + ".png"
	if err := os.WriteFile(s.jobs.OutputPath(name), data, 0o644); err != nil {
		return err
	}
	completed, err := s.completeGenerationJob(j, "/api/outputs/"+name, func() {
		_ = os.Remove(s.jobs.OutputPath(name))
	})
	if !completed && err == nil {
		return context.Canceled
	}
	return err
}
