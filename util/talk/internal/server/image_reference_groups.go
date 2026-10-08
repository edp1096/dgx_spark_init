package server

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"sparktalk/internal/db"
)

// Candidate photos are reviewed by the conversation model. Only the explicitly
// selected scene/head photo enters that engine call; never pretend all did.
type imageReferenceGroup struct {
	Subject         string   `json:"subject"`
	ImageIDs        []string `json:"image_ids"`
	SceneImageID    string   `json:"scene_image_id"`
	HeadImageID     string   `json:"head_image_id"`
	Target          string   `json:"target"`
	SelectionReason string   `json:"selection_reason"`
}

type imageReferenceGroupEvidence struct {
	imageReferenceGroup
	Candidates []imageInputEvidence `json:"candidates"`
}

func imageReferenceGroupsSchema() map[string]any {
	fields := map[string]any{}
	for _, key := range []string{"subject", "scene_image_id", "head_image_id", "target", "selection_reason"} {
		fields[key] = map[string]any{"type": "string", "minLength": 1, "maxLength": 2048}
	}
	fields["image_ids"] = map[string]any{"type": "array", "minItems": 1, "maxItems": 12, "items": map[string]any{"type": "string"}}
	return map[string]any{"type": "array", "minItems": 1, "maxItems": 4, "description": "For several photos of the SAME person/character, preserve one group per subject. Include ALL supplied candidate IDs in image_ids after visually reviewing them, and select scene_image_id and head_image_id FROM that group with selection_reason explaining pose/clarity. target identifies that subject's position/appearance in the intended scene. At most 4 subjects and 12 candidate photos total. For paint reference_generate use these groups INSTEAD OF reference_images/head_targets; Talk derives one scene reference and one BFS head plan per subject. Three photos of one person mean ONE subject, not three characters. Candidates are reviewed by Qwen; only the selected image is sent per engine stage.", "items": map[string]any{"type": "object", "properties": fields, "required": []string{"subject", "image_ids", "scene_image_id", "head_image_id", "target", "selection_reason"}, "additionalProperties": false}}
}

func expandReferenceGroups(args *imageGenerationArgs) error {
	if len(args.ReferenceGroups) == 0 {
		return nil
	}
	if args.Operation != "reference_generate" || !args.PaintMode {
		return fmt.Errorf("reference_groups requires paint reference_generate")
	}
	if len(args.ReferenceImages) > 0 || args.HeadTargets != nil {
		return fmt.Errorf("use reference_groups instead of reference_images and head_targets; the server derives stage selections")
	}
	if len(args.ReferenceGroups) > 4 {
		return fmt.Errorf("at most four subject groups are supported")
	}
	seen := map[string]bool{}
	subjects := map[string]bool{}
	refs := []imageReference{}
	heads := []imageHeadTarget{}
	for _, g := range args.ReferenceGroups {
		for _, v := range []string{g.Subject, g.Target, g.SelectionReason} {
			if strings.TrimSpace(v) == "" || utf8.RuneCountInString(v) > 2048 {
				return fmt.Errorf("each subject group requires subject, target and selection_reason of 1..2048 characters")
			}
		}
		if subjects[g.Subject] {
			return fmt.Errorf("duplicate subject %q; combine its photos in one group", g.Subject)
		}
		subjects[g.Subject] = true
		if len(g.ImageIDs) == 0 || len(g.ImageIDs) > 12 {
			return fmt.Errorf("each subject requires 1..12 candidate images")
		}
		own := map[string]bool{}
		for _, id := range g.ImageIDs {
			if id == "" || seen[id] {
				return fmt.Errorf("candidate image %s must belong to exactly one subject group", id)
			}
			seen[id] = true
			own[id] = true
		}
		if !own[g.SceneImageID] || !own[g.HeadImageID] {
			return fmt.Errorf("scene/head selection for %s must belong to that subject's image_ids", g.Subject)
		}
		refs = append(refs, imageReference{ImageID: g.SceneImageID, Description: g.Subject + "; target: " + g.Target})
		heads = append(heads, imageHeadTarget{ReferenceImageID: g.HeadImageID, Target: g.Subject + ": " + g.Target})
	}
	if len(seen) > 12 {
		return fmt.Errorf("at most twelve candidate images are supported per grouped request")
	}
	args.ReferenceImages = refs
	args.HeadTargets = &heads
	return nil
}

func (s *Server) referenceGroupEvidence(groups []imageReferenceGroup, attachments map[string]db.Attachment) ([]imageReferenceGroupEvidence, error) {
	out := []imageReferenceGroupEvidence{}
	for _, g := range groups {
		r := imageReferenceGroupEvidence{imageReferenceGroup: g}
		for _, id := range g.ImageIDs {
			item, err := s.imageEvidence(id, "candidate", g.Subject, attachments)
			if err != nil {
				return nil, err
			}
			r.Candidates = append(r.Candidates, item)
		}
		out = append(out, r)
	}
	return out, nil
}
