package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"image/color"
	"net/http"
	"net/http/httptest"
	"testing"

	"sparktalk/internal/config"
	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/workflows"
)

func TestSixPhotosRemainTwoSubjectsWithSeparateSceneAndHeadSelections(t *testing.T) {
	s, _ := testImageServer(t)
	photos := []db.Attachment{}
	for i := 0; i < 6; i++ {
		photos = append(photos, addReferenceFixture(t, s, "user", color.RGBA{R: uint8(i * 30), A: 255}, 64, 64))
	}
	groups := []imageReferenceGroup{
		{Subject: "person A", ImageIDs: []string{photos[0].ID, photos[1].ID, photos[2].ID}, SceneImageID: photos[0].ID, HeadImageID: photos[1].ID, Target: "left person", SelectionReason: "full pose for scene; frontal head for correction"},
		{Subject: "person B", ImageIDs: []string{photos[3].ID, photos[4].ID, photos[5].ID}, SceneImageID: photos[3].ID, HeadImageID: photos[5].ID, Target: "right person", SelectionReason: "clear pose and head"},
	}
	selected := 0
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(8 << 20); err != nil {
			t.Fatal(err)
		}
		defer r.MultipartForm.RemoveAll()
		selected = len(r.MultipartForm.File["image"])
		if selected == 0 {
			selected = len(r.MultipartForm.File["image[]"])
		}
		json.NewEncoder(w).Encode(map[string]any{"data": []map[string]string{{"b64_json": base64.StdEncoding.EncodeToString(onePixelPNG)}}})
	}))
	defer worker.Close()
	turn := &turnImages{sessionID: "session", items: map[string]db.Attachment{}}
	ctx := context.WithValue(context.Background(), turnImageKey{}, turn)
	args := imageGenerationArgs{Operation: "reference_generate", ReferenceGroups: groups, Prompt: "exactly two people", Size: "512x512"}
	b, _ := json.Marshal(args)
	result, err := s.executeImageGenerateTool(ctx, "session", config.ImageConfig{Enabled: true, Mode: "paint", Endpoint: worker.URL, Model: "test", Timeout: "2s"}, llm.ToolCall{Function: llm.FunctionCall{Arguments: string(b)}}, func(string, any) error { return nil })
	if err != nil {
		t.Fatal(err)
	}
	if selected != 2 {
		t.Fatalf("six candidates flattened into %d engine inputs", selected)
	}
	var report struct {
		Groups []imageReferenceGroupEvidence `json:"reference_groups"`
		Inputs []imageInputEvidence          `json:"input_images"`
	}
	json.Unmarshal([]byte(result.Result), &report)
	if len(report.Groups) != 2 || len(report.Groups[0].Candidates) != 3 || len(report.Groups[1].Candidates) != 3 || len(report.Inputs) != 2 {
		t.Fatalf("lost groups/provenance: %s", result.Result)
	}
	if report.Inputs[0].ID != photos[0].ID || report.Inputs[1].ID != photos[3].ID {
		t.Fatal("wrong scene selections")
	}
	var next imageGenerationArgs
	json.Unmarshal([]byte(pendingHeadCall(ctx).Function.Arguments), &next)
	if next.ReferenceImages[0].ImageID != photos[1].ID {
		t.Fatal("head selection replaced by scene selection")
	}
	recordHeadPlan(ctx, next, "next-image")
	json.Unmarshal([]byte(pendingHeadCall(ctx).Function.Arguments), &next)
	if next.ReferenceImages[0].ImageID != photos[5].ID || next.SourceImageID != "next-image" {
		t.Fatal("wrong second subject")
	}
	run := workflows.Run{SessionID: "session", Current: 0, Steps: []workflows.StepState{{Status: "paused", Evidence: []workflows.Evidence{{Tool: "image_generate", Arguments: string(b), Result: result.Result}}}}}
	restored, e := s.restoreWorkflowImages(context.Background(), run)
	if e != nil {
		t.Fatal(e)
	}
	json.Unmarshal([]byte(pendingHeadCall(restored).Function.Arguments), &next)
	if next.ReferenceImages[0].ImageID != photos[1].ID {
		t.Fatal("resume lost grouped head selection")
	}
}

func TestGroupsRejectCrossIdentitySelectionsAndMissingCandidates(t *testing.T) {
	valid := imageReferenceGroup{Subject: "A", ImageIDs: []string{"a", "b"}, SceneImageID: "a", HeadImageID: "b", Target: "left", SelectionReason: "clear"}
	bad := valid
	bad.HeadImageID = "other-person"
	duplicate := valid
	duplicate.Subject = "B"
	for _, groups := range [][]imageReferenceGroup{{bad}, {valid, duplicate}} {
		a := imageGenerationArgs{PaintMode: true, Operation: "reference_generate", ReferenceGroups: groups}
		if expandReferenceGroups(&a) == nil {
			t.Fatal("invalid grouping accepted")
		}
	}
	s, photo := testImageServer(t)
	valid.ImageIDs = []string{photo.ID, "missing"}
	valid.SceneImageID = photo.ID
	valid.HeadImageID = photo.ID
	if _, err := s.referenceGroupEvidence([]imageReferenceGroup{valid}, map[string]db.Attachment{photo.ID: photo}); err == nil {
		t.Fatal("unavailable unselected candidate silently ignored")
	}
}
