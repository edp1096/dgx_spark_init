package server

import (
	"context"
	"encoding/json"
	"fmt"
	"image"
	"io"
	"strings"
	"unicode/utf8"

	"sparktalk/internal/db"
	"sparktalk/internal/imagegen"
)

type imageReference struct {
	ImageID     string `json:"image_id"`
	Description string `json:"description"`
}

// Tool-produced image messages use the user role for multimodal followups;
// preserve their actual generated provenance in the adjacent visual labels.
type imageAttachmentOriginKey struct{}

type imageInputEvidence struct {
	Index       int    `json:"index"`
	ID          string `json:"id"`
	Name        string `json:"name"`
	URL         string `json:"url"`
	Role        string `json:"role"`
	Description string `json:"description,omitempty"`
	Origin      string `json:"origin"`
	Width       int    `json:"width"`
	Height      int    `json:"height"`
	CropBox     []int  `json:"crop_box,omitempty"`
}

func imageReferenceSchema() map[string]any {
	return map[string]any{"type": "array", "minItems": 1, "maxItems": 4, "description": "Ordered identity/character references. Use reference_generate for a new scene; identity_edit uses source_image_id as image 1, then these references. head_swap requires exactly ONE original head reference here; the base scene is source_image_id. At most four images total. Describe each actual image and map it to the requested subject; never guess from identical filenames.", "items": map[string]any{
		"type": "object", "properties": map[string]any{"image_id": map[string]any{"type": "string"}, "description": map[string]any{"type": "string", "minLength": 1, "maxLength": 2048}}, "required": []string{"image_id", "description"}, "additionalProperties": false,
	}}
}

const imageReferenceInstructions = `When the user groups several photos as the same person/character (for example images 1,2,3 = A and 4,5,6 = B), use reference_groups with ALL candidate IDs grouped exactly as requested. Inspect the candidates; select a clear pose-appropriate scene_image_id and head_image_id within each group, explain selection_reason, and omit reference_images/head_targets because the server derives them. Never flatten six photos into six engine inputs or six people. Candidate count is different from subject count. Never claim all candidates entered Klein/BFS: report the actual selected inputs. If a numbered photo is missing, do not fabricate an ID; identify the missing photo or use an explicitly relevant existing original with explanation. For a new scene based on uploaded characters/people, use reference_generate with reference_images, not text-only generate. In paint mode reference_generate without reference_groups MUST declare head_targets: for each person/character head identity, provide its reference_image_id and target description (position/appearance in the intended scene). Use [] only for non-head object, landscape or style references. Talk tracks this correction plan and will run missing BFS steps if you attempt to finish early. Inspect composition first; if regeneration is necessary, retain the plan and the original user-requested action. A request to hit/slap must not silently become pointing/scolding. After each head correction inspect likeness and collateral changes; report remaining defects honestly. For retouching an existing scene, use identity_edit with its source_image_id and add original identity references when needed. In paint mode, for replacing a specific face/head with an uploaded photo use head_swap (BFS Klein 4B V1), source_image_id for the base scene and exactly ONE original head in reference_images. For multiple people, name the selected person precisely and edit one head per call; feed the result into the next head_swap. By default omit mask_box and reference_crop_box: visually guessed rectangles can clip the chin or hair and produce duplicate faces. Supply these boxes only for coordinates explicitly supplied/selected by the user. Whole-scene head_swap can change other details; inspect the result and never claim pixel-exact preservation without an explicit target region. A clear cropped head portrait works better than a full-body reference. For head_swap visual selection set head_box_coordinates="normalized_1000": each image spans 0..1000 horizontally and vertically. mask_box uses the base scene; reference_crop_box uses the original reference. Never exceed 1000. The server converts each separately to pixels. Other operations keep original pixel coordinates. Omit size: head_swap keeps original output dimensions. LoRA strength defaults to 1.0, range 0.1..1.5. Keep the base style if requested; never add chibi, exaggerated facial proportions, sunglasses, or a new age unless requested. Check the visible result before claiming the face matches. Ordinary image changes still use identity_edit; inpaint/outpaint do not replace identity references. Maximum four images total, in the exact declared order. Inspect each attachment's visual content and ID; if uncertain, use attachment_read for that exact ID. Same filenames do not imply the same photo. Keep subject names and actions requested by the user; do not silently replace them with generic characters. A generated intermediate is not a substitute for original reference photos. Multi-stage work is supported when requested or when a visible defect needs correction: generate a scene, inspect the returned image, then use that generated image ID as source_image_id with the original character photos in reference_images. Clearly describe which subject or face to correct and what must remain unchanged. This reference edit can alter the whole scene; it is not a pixel-exact face paste or a masked edit. When the user requires faithful faces and the generated result visibly differs, use head_swap once per affected head with original references before the final response. If likeness is still poor, report that limit instead of claiming success or continuing endless retries. Do not add chibi or exaggerated facial proportions when likeness is requested. Inspect each result before another step, stop when satisfactory, and do not repeat an unchanged successful call or keep retrying without a visible reason. Use only defined tool fields; preserve_identity is not an available option. Check the returned input_images list before reporting which photos were used. identity_verified=false means only image delivery is verified: inspect the result and report any drift rather than claiming exact likeness. A successful API call does not prove identity, pose, or text preservation.`

func imageResultInstruction(inputs []imageInputEvidence) string {
	b, _ := json.Marshal(inputs)
	return "These are the images produced by image_generate. Actual input_images=" + string(b) + ". Only these images were sent to the image engine; do not claim any other attachment was used. Identity/pose/text preservation has not been verified. Compare the visible output with the user's actual inputs and acknowledge differences. Generated attachment IDs can be used for subsequent requested editing steps; retain original identity references when needed. Otherwise answer without repeating a successful operation."
}

// Only explicit references to historical user images impose a history guard.
// A fresh upload is considered an input, except an explicit instruction not to use it.
// This guard protects transport; it cannot verify the semantics of a face label.
type originalImageRequestKey struct{}

// Retries and edits use a selected branch that may differ from the saved latest
// turn. Carry its reference requirements through workflow stages as well.
func requestedOriginalImagesForContext(ctx context.Context, s *Server, sessionID string) map[string]bool {
	if ids, ok := ctx.Value(originalImageRequestKey{}).(map[string]bool); ok {
		return ids
	}
	return requestedOriginalImages(s, sessionID)
}

func requestedOriginalImages(s *Server, sessionID string) map[string]bool {
	if s == nil || s.db == nil {
		return nil
	}
	messages, err := s.db.Messages(sessionID)
	if err != nil {
		return nil
	}
	return requestedOriginalImagesInMessages(messages)
}

func requestedOriginalImagesInMessages(messages []db.Message) map[string]bool {
	latest := -1
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			latest = i
			break
		}
	}
	if latest < 0 {
		return nil
	}
	text := strings.ToLower(userTurnContent(messages[latest]))
	for _, phrase := range []string{"참조하지", "참고하지", "사진 무시", "이미지 무시", "참조 없이", "참고 없이", "without reference", "ignore the image", "ignore the photo", "do not use the image", "don't use the image"} {
		if strings.Contains(text, phrase) {
			return nil
		}
	}
	explicit := false
	for _, phrase := range []string{"올린 사진", "올린 이미지", "원본 사진", "원본 이미지", "참조", "샘플 사진", "샘플 이미지", "사진을", "사진 두", "사진 2", "reference", "uploaded", "original photo", "original image", "these photos", "these images"} {
		if strings.Contains(text, phrase) {
			explicit = true
			break
		}
	}
	for i := latest; i >= 0; i-- {
		if messages[i].Role != "user" {
			continue
		}
		ids := map[string]bool{}
		for _, a := range messages[i].Attachments {
			if strings.HasPrefix(a.MIME, "image/") {
				ids[a.ID] = true
			}
		}
		if len(ids) > 0 && (i == latest || explicit) {
			return ids
		}
		if i == latest && !explicit {
			return nil
		}
	}
	return nil
}

func (s *Server) imageEvidence(id, role, description string, attachments map[string]db.Attachment) (imageInputEvidence, error) {
	a, ok := attachments[id]
	if !ok {
		return imageInputEvidence{}, fmt.Errorf("image attachment %q is not available in this conversation", id)
	}
	file, err := s.media.Open(a)
	if err != nil {
		return imageInputEvidence{}, err
	}
	defer file.Close()
	info, _, err := image.DecodeConfig(file)
	if err != nil {
		return imageInputEvidence{}, fmt.Errorf("invalid input image %s: %w", id, err)
	}
	return imageInputEvidence{ID: id, Name: a.Name, URL: a.URL, Role: role, Description: description, Width: info.Width, Height: info.Height}, nil
}

func (s *Server) normalizeHeadBoxes(args *imageGenerationArgs, attachments map[string]db.Attachment) error {
	if args.HeadBoxCoordinates == "" {
		return nil
	}
	if args.Operation != "head_swap" {
		return fmt.Errorf("head_box_coordinates requires head_swap")
	}
	if args.HeadBoxCoordinates == "pixels" {
		return nil
	}
	if args.HeadBoxCoordinates != "normalized_1000" {
		return fmt.Errorf("head_box_coordinates must be pixels or normalized_1000")
	}
	if len(args.ReferenceImages) != 1 {
		return fmt.Errorf("head_swap requires exactly one original head reference")
	}
	for i, box := range []*[]int{&args.MaskBox, &args.ReferenceCropBox} {
		if len(*box) == 0 {
			continue
		}
		b := *box
		if len(b) != 4 || b[0] < 0 || b[1] < 0 || b[2] > 1000 || b[3] > 1000 || b[2] <= b[0] || b[3] <= b[1] {
			return fmt.Errorf("head_swap normalized box %v must be [left,top,right,bottom] within 0..1000 with increasing edges", b)
		}
		id := args.SourceImageID
		if i == 1 {
			id = args.ReferenceImages[0].ImageID
		}
		input, err := s.imageEvidence(id, "", "", attachments)
		if err != nil {
			return err
		}
		*box = []int{b[0] * input.Width / 1000, b[1] * input.Height / 1000, (b[2]*input.Width + 999) / 1000, (b[3]*input.Height + 999) / 1000}
	}
	return nil
}

func (s *Server) validateImageReferences(ctx context.Context, sessionID, mode string, args imageGenerationArgs, attachments map[string]db.Attachment) ([]imageInputEvidence, error) {
	inputs := make([]imageInputEvidence, 0)
	portable := mode == "paint" || mode == "reference" || mode == "qwen-image21"
	if !portable && (len(args.ReferenceImages) > 0 || args.SourceImageDescription != "" || args.Operation == "reference_generate") {
		return nil, fmt.Errorf("ordered multi-reference input requires reference or paint image mode")
	}
	if portable {
		if len(args.ReferenceImages) > 0 && args.Operation != "reference_generate" && args.Operation != "identity_edit" && args.Operation != "head_swap" {
			return nil, fmt.Errorf("reference_images requires reference_generate or identity_edit; text-only generate cannot use references")
		}
		if args.Operation == "reference_generate" && (len(args.ReferenceImages) == 0 || (args.SourceImageID != "" && args.SourceImageID != args.ReferenceImages[0].ImageID)) {
			return nil, fmt.Errorf("reference_generate requires ordered reference_images; omit source_image_id or use the same first reference ID")
		}
		if args.SourceImageDescription != "" && args.SourceImageID == "" {
			return nil, fmt.Errorf("source_image_description requires source_image_id")
		}
		if utf8.RuneCountInString(args.SourceImageDescription) > 2048 {
			return nil, fmt.Errorf("source_image_description is limited to 2048 characters")
		}
	}
	seen := map[string]bool{}
	add := func(id, role, description string) error {
		if id == "" {
			return nil
		}
		if seen[id] {
			return fmt.Errorf("duplicate input image %s; declare each image once", id)
		}
		item, err := s.imageEvidence(id, role, description, attachments)
		if err != nil {
			return err
		}
		seen[id] = true
		item.Index = len(inputs) + 1
		inputs = append(inputs, item)
		return nil
	}
	if args.Operation != "generate" && args.Operation != "reference_generate" {
		if err := add(args.SourceImageID, "source", strings.TrimSpace(args.SourceImageDescription)); err != nil {
			return nil, err
		}
	}
	for _, ref := range args.ReferenceImages {
		description := strings.TrimSpace(ref.Description)
		if ref.ImageID == "" || description == "" || utf8.RuneCountInString(description) > 2048 {
			return nil, fmt.Errorf("each reference requires an image_id and a checked description of 1..2048 characters")
		}
		if err := add(ref.ImageID, "reference", description); err != nil {
			return nil, err
		}
	}
	if portable && len(inputs) > 4 {
		return nil, fmt.Errorf("This Talk image integration accepts at most four selected engine inputs per call. For multiple photos of the same subject, use reference_groups to retain all candidates and select one scene/head reference per subject.")
	}
	if args.MaskImageID != "" && (args.Operation == "inpaint" || args.Operation == "object_remove" || (mode == "extended" && args.Operation == "identity_edit")) {
		if err := add(args.MaskImageID, "mask", ""); err != nil {
			return nil, err
		}
	}
	if mode == "extended" {
		for _, ref := range []struct{ ID, Role string }{{args.ReferenceImageID, "reference"}, {args.ControlImageID, "control"}} {
			if ref.ID != "" && (args.Operation == "identity_edit" || (ref.Role == "control" && (args.Operation == "depth" || args.Operation == "nk2e_canny"))) {
				if err := add(ref.ID, ref.Role, ""); err != nil {
					return nil, err
				}
			}
		}
		if args.Operation == "vision_reference" {
			for _, id := range args.VisionImageIDs {
				if err := add(id, "reference", ""); err != nil {
					return nil, err
				}
			}
		}
		if args.Operation == "style_reference" {
			for _, id := range args.StyleReferenceImageIDs {
				if err := add(id, "style", ""); err != nil {
					return nil, err
				}
			}
		}
	}
	// Record provenance separately from the model-written subject labels.
	attachmentsWithOrigins, err := s.conversationAttachments(ctx, sessionID)
	if err != nil {
		return nil, err
	}
	origins := map[string]string{}
	for _, item := range attachmentsWithOrigins {
		origins[item.Attachment.ID] = item.Origin
	}
	for i := range inputs {
		inputs[i].Origin = origins[inputs[i].ID]
		if inputs[i].Origin == "" {
			inputs[i].Origin = "generated_this_turn"
		}
	}
	if args.Operation == "head_swap" {
		if (mode != "paint" && mode != "qwen-image21") || args.SourceImageID == "" || len(args.ReferenceImages) != 1 || len(inputs) != 2 {
			return nil, fmt.Errorf("head_swap requires paint mode, one source_image_id and exactly one original head in reference_images")
		}
		if args.HeadSwapStrength != nil && !(*args.HeadSwapStrength >= .1 && *args.HeadSwapStrength <= 1.5) {
			return nil, fmt.Errorf("head_swap_strength must be between 0.1 and 1.5")
		}
		for i, box := range [][]int{args.MaskBox, args.ReferenceCropBox} {
			if len(box) == 0 {
				continue
			}
			input := &inputs[i]
			if len(box) != 4 || box[0] < 0 || box[1] < 0 || box[2] > input.Width || box[3] > input.Height || box[2]-box[0] < 32 || box[3]-box[1] < 32 {
				return nil, fmt.Errorf("head_swap box %v for image %s must fit its %dx%d original pixels and have each side at least 32px", box, input.ID, input.Width, input.Height)
			}
			input.CropBox = append([]int{}, box...)
		}
	}
	if portable {
		if turn, ok := ctx.Value(turnImageKey{}).(*turnImages); ok && turn.sessionID == sessionID {
			turn.mu.Lock()
			defer turn.mu.Unlock()
			if len(turn.requiredOriginals) > 0 && !turn.referencesSatisfied {
				matched := false
				for _, item := range inputs {
					if turn.requiredOriginals[item.ID] {
						matched = true
						break
					}
				}
				if !matched {
					return nil, fmt.Errorf("this request uses uploaded/original images, but no requested original image was selected; use reference_generate with reference_images or identity_edit with the correct source and identity references; do not substitute a generated intermediate")
				}
			}
		}
	}
	return inputs, nil
}

func (s *Server) generateWithReferences(ctx context.Context, client *imagegen.Client, args imageGenerationArgs, attachments map[string]db.Attachment) (generatedImage, error) {
	refs := append([]imageReference{}, args.ReferenceImages...)
	if args.SourceImageID != "" && args.Operation == "identity_edit" {
		refs = append([]imageReference{{ImageID: args.SourceImageID, Description: defaultString(args.SourceImageDescription, "base scene to edit; preserve its composition except requested changes")}}, refs...)
	}
	images := make([]imagegen.ReferenceImage, 0, len(refs))
	var mapping strings.Builder
	mapping.WriteString("Ordered input image mapping (use the actual supplied images, not a guessed likeness):\n")
	for i, ref := range refs {
		a, ok := attachments[ref.ImageID]
		if !ok {
			return generatedImage{}, fmt.Errorf("reference image unavailable: %s", ref.ImageID)
		}
		f, err := s.media.Open(a)
		if err != nil {
			return generatedImage{}, err
		}
		data, err := io.ReadAll(io.LimitReader(f, 40<<20))
		f.Close()
		if err != nil {
			return generatedImage{}, err
		}
		if len(data) >= 40<<20 {
			return generatedImage{}, fmt.Errorf("reference image exceeds 40 MiB")
		}
		images = append(images, imagegen.ReferenceImage{Data: data, MIME: a.MIME})
		fmt.Fprintf(&mapping, "Image %d: %s\n", i+1, ref.Description)
	}
	mapping.WriteString("Keep the reference identities and their assigned roles distinct.\nRequested scene/edit: " + args.Prompt)
	var operation []string
	if args.NativeQwenImage {
		operation = []string{args.Operation}
	}
	result, err := client.Edit(ctx, images, mapping.String(), args.Size, args.Seed, operation...)
	if err != nil {
		return generatedImage{}, err
	}
	return generatedImage{Data: result.Image, Seed: result.Seed, Name: fmt.Sprintf("image-%s-%d.png", args.Operation, result.Seed)}, nil
}
