package server

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"unicode/utf8"

	"sparktalk/internal/llm"
)

type imageHeadTarget struct {
	ReferenceImageID string `json:"reference_image_id"`
	Target           string `json:"target"`
}

func imageHeadTargetsSchema() map[string]any {
	return map[string]any{"type": "array", "maxItems": 4, "description": "Required for reference_generate in paint mode unless reference_groups is used. Declare one head correction target per person/character identity reference: reference_image_id plus an unambiguous English position/appearance description in the generated scene. Talk tracks and completes these BFS steps after composition. Use [] for objects, landscapes, or style references without a head identity to transfer. Preserve the user's action in the generation prompt, including on retries.", "items": map[string]any{"type": "object", "properties": map[string]any{"reference_image_id": map[string]any{"type": "string"}, "target": map[string]any{"type": "string", "minLength": 1, "maxLength": 2048}}, "required": []string{"reference_image_id", "target"}, "additionalProperties": false}}
}

func validateHeadPlan(ctx context.Context, args imageGenerationArgs) error {
	switch workflowImagePhase(ctx) {
	case "composition", "review":
		return fmt.Errorf("this stage only reviews existing images; report failed if the scene needs correction")
	case "scene":
		if args.Operation == "head_swap" {
			return fmt.Errorf("finish scene composition and its review before the head correction stage")
		}
	case "heads":
		if args.Operation != "head_swap" {
			return fmt.Errorf("this stage only performs the declared head_swap plan; do not regenerate style or composition")
		}
	}
	if args.HeadTargets != nil && (!args.PaintMode || args.Operation != "reference_generate") {
		return fmt.Errorf("head_targets requires reference_generate in paint mode")
	}
	if args.PaintMode && args.Operation == "reference_generate" {
		if args.HeadTargets == nil {
			return fmt.Errorf("reference_generate requires head_targets: declare each person/character reference_image_id and its target position/appearance; use [] only for non-head object, landscape, or style references")
		}
		if len(*args.HeadTargets) > 4 {
			return fmt.Errorf("at most four head targets are supported")
		}
		refs := map[string]bool{}
		for _, g := range args.ReferenceGroups {
			refs[g.HeadImageID] = true
		}
		for _, r := range args.ReferenceImages {
			refs[r.ImageID] = true
		}
		seen := map[string]bool{}
		for _, p := range *args.HeadTargets {
			if !refs[p.ReferenceImageID] || seen[p.ReferenceImageID] || strings.TrimSpace(p.Target) == "" || utf8.RuneCountInString(p.Target) > 2048 {
				return fmt.Errorf("each head target must use a unique declared reference_image_id and a non-empty target description up to 2048 characters")
			}
			seen[p.ReferenceImageID] = true
		}
		if turn, ok := ctx.Value(turnImageKey{}).(*turnImages); ok {
			turn.mu.Lock()
			pending := append([]imageHeadTarget(nil), turn.headPlan...)
			turn.mu.Unlock()
			for _, p := range pending {
				if !seen[p.ReferenceImageID] {
					return fmt.Errorf("composition retry must retain pending head target %s", p.ReferenceImageID)
				}
			}
		}
	}
	if turn, ok := ctx.Value(turnImageKey{}).(*turnImages); ok {
		turn.mu.Lock()
		defer turn.mu.Unlock()
		if len(turn.headPlan) > 0 && args.Operation != "reference_generate" {
			if args.SourceImageID != turn.headBase {
				return fmt.Errorf("pending head corrections require latest scene source_image_id=%s; do not restart from an older image", turn.headBase)
			}
			if args.Operation == "head_swap" {
				found := false
				if len(args.ReferenceImages) == 1 {
					for _, p := range turn.headPlan {
						if p.ReferenceImageID == args.ReferenceImages[0].ImageID {
							found = true
						}
					}
				}
				if !found {
					return fmt.Errorf("head_swap must use one pending original identity reference")
				}
			}
		}
	}
	return nil
}

func recordHeadPlan(ctx context.Context, args imageGenerationArgs, outputID string) {
	turn, ok := ctx.Value(turnImageKey{}).(*turnImages)
	if !ok {
		return
	}
	turn.mu.Lock()
	defer turn.mu.Unlock()
	if args.Operation == "reference_generate" && args.HeadTargets != nil {
		turn.headPlan = append([]imageHeadTarget(nil), (*args.HeadTargets)...)
		turn.headBase = outputID
		turn.headScene = outputID
		turn.headPrompt = args.Prompt
		turn.headFailures = nil
	} else if len(turn.headPlan) > 0 && args.SourceImageID == turn.headBase {
		if args.Operation == "head_swap" && len(args.ReferenceImages) == 1 {
			for i, p := range turn.headPlan {
				if p.ReferenceImageID == args.ReferenceImages[0].ImageID {
					turn.headPlan = append(turn.headPlan[:i], turn.headPlan[i+1:]...)
					break
				}
			}
		}
		turn.headBase = outputID
		if args.Operation != "head_swap" {
			turn.headScene = outputID
		}
	}
}

func pendingHeadCall(ctx context.Context) *llm.ToolCall {
	if phase := workflowImagePhase(ctx); phase == "scene" || phase == "composition" || phase == "review" {
		return nil
	}
	turn, ok := ctx.Value(turnImageKey{}).(*turnImages)
	if !ok {
		return nil
	}
	turn.mu.Lock()
	defer turn.mu.Unlock()
	if len(turn.headPlan) == 0 {
		return nil
	}
	p := turn.headPlan[0]
	args := imageGenerationArgs{Operation: "head_swap", SourceImageID: turn.headBase, ReferenceImages: []imageReference{{ImageID: p.ReferenceImageID, Description: "Original head identity for " + p.Target}}, Prompt: "Replace only the head of " + p.Target + " using the original reference head. Preserve the existing rendering style, expression, pose, all other people and scene details. Preserve the requested action and composition: " + turn.headPrompt}
	b, _ := json.Marshal(args)
	return &llm.ToolCall{ID: "head-" + turn.headBase + "-" + p.ReferenceImageID, Type: "function", Function: llm.FunctionCall{Name: "image_generate", Arguments: string(b)}}
}

// A failed correction is recorded once, not silently retried on every final answer.
func failHeadCall(ctx context.Context, call llm.ToolCall, err error) {
	if err == nil || call.Function.Name != "image_generate" {
		return
	}
	var a imageGenerationArgs
	if json.Unmarshal([]byte(call.Function.Arguments), &a) != nil || a.Operation != "head_swap" || len(a.ReferenceImages) != 1 {
		return
	}
	turn, ok := ctx.Value(turnImageKey{}).(*turnImages)
	if !ok {
		return
	}
	turn.mu.Lock()
	defer turn.mu.Unlock()
	if a.SourceImageID != turn.headBase {
		return
	}
	for i, p := range turn.headPlan {
		if p.ReferenceImageID == a.ReferenceImages[0].ImageID {
			turn.headFailures = append(turn.headFailures, p.Target+": "+err.Error())
			turn.headPlan = append(turn.headPlan[:i], turn.headPlan[i+1:]...)
			break
		}
	}
}

func headPlanFailureNotice(ctx context.Context) string {
	turn, ok := ctx.Value(turnImageKey{}).(*turnImages)
	if !ok {
		return ""
	}
	turn.mu.Lock()
	defer turn.mu.Unlock()
	if len(turn.headFailures) == 0 {
		return ""
	}
	return "얼굴 보정 일부가 실패했습니다. 생성된 장면은 있으나 요청한 얼굴 보정이 모두 완료된 결과는 아닙니다. 상세 오류는 도구 기록에서 확인할 수 있습니다."
}
