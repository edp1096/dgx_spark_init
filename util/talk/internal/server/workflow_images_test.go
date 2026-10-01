package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image/color"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/workflows"
)

func TestImageWorkflowOrdersStagesAndRequiresBothReviewImages(t *testing.T) {
	for _, omit := range []bool{false, true} {
		t.Run(fmt.Sprint(omit), func(t *testing.T) {
			s, ref := testImageServer(t)
			second := addReferenceFixture(t, s, "user", color.RGBA{B: 255, A: 255}, 64, 64)
			ops := []string{}
			worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				op := "reference_generate"
				if !strings.Contains(r.Header.Get("Content-Type"), "multipart") {
					var a map[string]any
					json.NewDecoder(r.Body).Decode(&a)
					op, _ = a["operation"].(string)
				}
				ops = append(ops, op)
				json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString(onePixelPNG)}}})
			}))
			defer worker.Close()
			s.cfg.Image = config.ImageConfig{Enabled: true, Mode: "paint", Endpoint: worker.URL, Model: "test", DefaultSize: "512x512", Timeout: "2s"}
			cfg := config.ToolsConfig{Enabled: true, SkillsEnabled: true, MaxRounds: 6}
			s.cfg.Tools = cfg
			requests := map[int]int{}
			backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var req struct {
					Messages []llm.Message `json:"messages"`
				}
				json.NewDecoder(r.Body).Decode(&req)
				stage := 0
				var images []db.Attachment
				for _, m := range req.Messages {
					if text, ok := m.Content.(string); ok {
						for n := 1; n <= 4; n++ {
							if strings.Contains(text, fmt.Sprintf("Current stage %d/4", n)) {
								stage = n
							}
						}
						if i := strings.Index(text, "Required images: "); i >= 0 {
							json.Unmarshal([]byte(text[i+len("Required images: "):]), &images)
						}
					}
				}
				requests[stage]++
				call := func(id, name string, args any) map[string]any {
					b, _ := json.Marshal(args)
					return map[string]any{"index": 0, "id": id, "type": "function", "function": map[string]string{"name": name, "arguments": string(b)}}
				}
				evidence := []string{}
				calls := []map[string]any{}
				switch stage {
				case 1:
					if requests[stage] == 1 {
						heads := []imageHeadTarget{{ref.ID, "left person"}, {second.ID, "right person"}}
						calls = append(calls, call("scene", "image_generate", imageGenerationArgs{Operation: "reference_generate", Prompt: "two people", ReferenceImages: []imageReference{{ref.ID, "left identity"}, {second.ID, "right identity"}}, HeadTargets: &heads}))
					} else {
						evidence = []string{"scene"}
					}
				case 2, 4:
					if len(images) == 0 {
						t.Error("generated image handoff missing")
					}
					if requests[stage] == 1 {
						for i, a := range images {
							if omit && stage == 4 && i == 0 {
								continue
							}
							x := call(fmt.Sprintf("read-%d-%d", stage, i), "attachment_read", map[string]string{"attachment_id": a.ID})
							x["index"] = len(calls)
							calls = append(calls, x)
						}
					} else {
						for i := range images {
							if omit && stage == 4 && i == 0 {
								continue
							}
							evidence = append(evidence, fmt.Sprintf("read-%d-%d", stage, i))
						}
					}
				case 3: // Attempt to skip BFS; the tracked plan must run both heads first.
				default:
					t.Errorf("unknown stage %d", stage)
				}
				if len(calls) == 0 {
					calls = append(calls, call("report", "workflow_report", workflows.Report{Status: "completed", Summary: fmt.Sprintf("stage %d reviewed", stage), Evidence: evidence}))
				}
				b, _ := json.Marshal(map[string]any{"choices": []map[string]any{{"delta": map[string]any{"tool_calls": calls}}}})
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", b)
			}))
			defer backend.Close()
			result, err := s.runWorkflow(context.Background(), "session", "@workflow:identity-image-production", "make two people", []llm.Message{{Role: "user", Content: "make two people"}}, llm.New(backend.URL, "test", ""), "test", "none", "", cfg, true, func(string, any) error { return nil }, func(db.Attachment) error { return nil })
			if omit {
				if err == nil {
					t.Fatal("final review passed without reading pre-correction scene")
				}
			} else if err != nil {
				t.Fatal(err)
			}
			if strings.Join(ops, ",") != "reference_generate,head_swap,head_swap" {
				t.Fatalf("wrong order or repeated work: %v", ops)
			}
			if len(result.Attachments) != 3 {
				t.Fatalf("lost outputs: %d", len(result.Attachments))
			}
			if requests[2] != 2 || requests[3] != 3 {
				t.Fatalf("stage gates bypassed: %v", requests)
			}
		})
	}
}

func TestWorkflowRestoresPartialHeadPlanAndMedia(t *testing.T) {
	s, ref := testImageServer(t)
	second := addReferenceFixture(t, s, "user", color.RGBA{B: 255, A: 255}, 64, 64)
	scene := addReferenceFixture(t, s, "assistant", color.RGBA{R: 255, A: 255}, 64, 64)
	corrected := addReferenceFixture(t, s, "assistant", color.RGBA{G: 255, A: 255}, 64, 64)
	heads := []imageHeadTarget{{ref.ID, "left"}, {second.ID, "right"}}
	proof := func(a imageGenerationArgs, out db.Attachment) workflows.Evidence {
		b, _ := json.Marshal(a)
		r, _ := json.Marshal(map[string]any{"attachments": []db.Attachment{out}})
		return workflows.Evidence{Tool: "image_generate", Arguments: string(b), Result: string(r)}
	}
	run := workflows.Run{SessionID: "session", Current: 2, Steps: []workflows.StepState{{Status: "completed", Evidence: []workflows.Evidence{proof(imageGenerationArgs{Operation: "reference_generate", HeadTargets: &heads}, scene)}}, {Status: "completed"}, {Status: "paused", Evidence: []workflows.Evidence{proof(imageGenerationArgs{Operation: "head_swap", SourceImageID: scene.ID, ReferenceImages: []imageReference{{ref.ID, "left"}}}, corrected)}}}}
	ctx, err := s.restoreWorkflowImages(context.Background(), run)
	if err != nil {
		t.Fatal(err)
	}
	var next imageGenerationArgs
	json.Unmarshal([]byte(pendingHeadCall(ctx).Function.Arguments), &next)
	if next.SourceImageID != corrected.ID || next.ReferenceImages[0].ImageID != second.ID {
		t.Fatalf("repeated completed head or stale base: %+v", next)
	}
	ids := requiredWorkflowImageReads(ctx, "review")
	if len(ids) != 2 || ids[0] != corrected.ID || ids[1] != scene.ID {
		t.Fatal(ids)
	}
	for _, phase := range []string{"scene", "composition", "review"} {
		phaseCtx := context.WithValue(ctx, workflowStageKey{}, &workflowStage{ImagePhase: phase})
		if pendingHeadCall(phaseCtx) != nil {
			t.Fatal("BFS leaked into", phase)
		}
		next.PaintMode = true
		if validateHeadPlan(phaseCtx, next) == nil {
			t.Fatal("wrong-stage mutation accepted", phase)
		}
	}
	if evidenceMatches(workflows.Step{VerifyTool: "attachment_read"}, workflows.Evidence{Tool: "attachment_read", Arguments: `{}`, Result: `{"attachments":[]}`}) {
		t.Fatal("listing accepted as image reading")
	}
}

func TestImageWorkflowSelectableWithImageCapability(t *testing.T) {
	s, _ := testImageServer(t)
	s.cfg.Image = config.ImageConfig{Enabled: true, Mode: "paint", Endpoint: "http://127.0.0.1:1", Model: "test"}
	s.cfg.Tools = config.ToolsConfig{Enabled: true, SkillsEnabled: true}
	w := httptest.NewRecorder()
	s.workflowCatalog(w, httptest.NewRequest("GET", "/api/workflows?selection=1", nil))
	if w.Code != 200 {
		t.Fatal(w.Body.String())
	}
	var choices []struct {
		Name    string   `json:"name"`
		Missing []string `json:"missing"`
	}
	json.Unmarshal(w.Body.Bytes(), &choices)
	for _, x := range choices {
		if x.Name == "identity-image-production" {
			if len(x.Missing) != 0 {
				t.Fatal(x.Missing)
			}
			return
		}
	}
	t.Fatal("new procedure missing")
}
