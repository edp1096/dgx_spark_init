package server

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/workflows"
)

func workflowImagePhase(ctx context.Context) string {
	if stage, ok := ctx.Value(workflowStageKey{}).(*workflowStage); ok {
		return stage.ImagePhase
	}
	return ""
}

// Rebuild from server-recorded successful effects, including an interrupted
// current stage. Completed heads are not repeated after resume.
func (s *Server) restoreWorkflowImages(ctx context.Context, run workflows.Run) (context.Context, error) {
	images := &turnImages{sessionID: run.SessionID, items: map[string]db.Attachment{}, requiredOriginals: requestedOriginalImages(s, run.SessionID)}
	ctx = context.WithValue(ctx, turnImageKey{}, images)
	for i, state := range run.Steps {
		if i > run.Current {
			break
		}
		if i < run.Current && state.Status != "completed" {
			continue
		}
		for _, proof := range state.Evidence {
			if proof.Tool != "image_generate" || proof.Error != "" {
				continue
			}
			var a imageGenerationArgs
			var out struct {
				Attachments []db.Attachment `json:"attachments"`
			}
			if json.Unmarshal([]byte(proof.Arguments), &a) != nil || json.Unmarshal([]byte(proof.Result), &out) != nil {
				continue
			}
			if len(a.ReferenceGroups) > 0 {
				a.PaintMode = true
				if err := expandReferenceGroups(&a); err != nil {
					return ctx, err
				}
			}
			for _, item := range out.Attachments {
				if !strings.HasPrefix(item.MIME, "image/") {
					continue
				}
				if s.media == nil {
					return ctx, fmt.Errorf("image storage is unavailable")
				}
				f, err := s.media.Open(item)
				if err != nil {
					return ctx, fmt.Errorf("workflow image %s: %w", item.ID, err)
				}
				f.Close()
				images.items[item.ID] = item
			}
			if len(out.Attachments) == 1 {
				recordHeadPlan(ctx, a, out.Attachments[0].ID)
			}
			images.referencesSatisfied = true
		}
	}
	return ctx, nil
}

func (s *Server) workflowImageInputs(ctx context.Context, phase string) ([]llm.Message, error) {
	if phase == "" || phase == "scene" {
		return nil, nil
	}
	images, ok := ctx.Value(turnImageKey{}).(*turnImages)
	if !ok {
		return nil, nil
	}
	images.mu.Lock()
	selected := []db.Attachment{}
	if phase == "review" && images.headScene != "" && images.headScene != images.headBase {
		if a, ok := images.items[images.headScene]; ok {
			selected = append(selected, a)
		}
	}
	if a, ok := images.items[images.headBase]; ok {
		selected = append(selected, a)
	}
	images.mu.Unlock()
	if len(selected) == 0 {
		return nil, fmt.Errorf("this image stage has no generated scene from an earlier stage")
	}
	if phase == "composition" || phase == "review" {
		b, _ := json.Marshal(selected)
		return []llm.Message{{Role: "user", Content: "Read these actual generated images with attachment_read before reporting this stage. For final review, compare the scene before correction (first) against the latest result (last); report visible defects, not the preceding stage's claims. Required images: " + string(b)}}, nil
	}
	cfg, _ := s.snapshot()
	return s.llmMessages(context.WithValue(ctx, imageAttachmentOriginKey{}, "assistant"), []db.Message{{Role: "user", Content: "Actual images from preceding workflow stages. If two are supplied, the first is the scene before face correction and the last is the latest result. Inspect these pixels and use their exact attachment IDs in attachment_read. Do not infer success from the previous stage's claims.", Attachments: selected}}, cfg)
}

func requiredWorkflowImageReads(ctx context.Context, phase string) []string {
	if phase != "composition" && phase != "review" {
		return nil
	}
	images, ok := ctx.Value(turnImageKey{}).(*turnImages)
	if !ok {
		return nil
	}
	images.mu.Lock()
	defer images.mu.Unlock()
	ids := []string{images.headBase}
	if phase == "review" && images.headScene != "" && images.headScene != images.headBase {
		ids = append(ids, images.headScene)
	}
	return ids
}
