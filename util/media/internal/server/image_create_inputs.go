package server

import (
	"net/http"
	"path/filepath"
)

type imageInputSpec struct {
	uploadField string
	reuseField  string
	directory   string
	limit       int
}

func (s *Server) persistImageCreateInputs(r *http.Request, jobID, mode string, maxReferences int, options *imageGenerationOptions) ([]string, error) {
	root := filepath.Join(s.dataDir, "inputs", jobID)
	references, err := s.persistImageInputSpec(r, root, imageInputSpec{
		uploadField: "references", reuseField: "reuse_references", limit: maxReferences,
	})
	if err != nil || mode != "create" {
		return references, err
	}

	specs := []struct {
		role string
		spec imageInputSpec
	}{
		{"identity", imageInputSpec{"identity_image", "reuse_identity_image", "identity", 1}},
		{"sequence_character", imageInputSpec{"sequence_character_image", "reuse_sequence_character_image", "sequence-character", 1}},
		{"identity_reference", imageInputSpec{"identity_reference", "reuse_identity_reference", "identity-reference", 3}},
		{"depth", imageInputSpec{"depth_image", "reuse_depth_image", "depth", 1}},
		{"identity_mask", imageInputSpec{"identity_mask", "reuse_identity_mask", "identity-mask", 1}},
		{"strict_mask", imageInputSpec{"strict_mask", "reuse_strict_mask", "strict-mask", 1}},
		{"vision", imageInputSpec{"vision_images", "reuse_vision_images", "vision", 4}},
		{"style_reference", imageInputSpec{"style_reference_images", "reuse_style_reference_images", "style-reference", 2}},
		{"nk2e", imageInputSpec{"nk2e_image", "reuse_nk2e_image", "nk2e", 1}},
		{"anypaint", imageInputSpec{"anypaint_image", "reuse_anypaint_image", "anypaint", 1}},
		{"anypaint_mask", imageInputSpec{"anypaint_mask", "reuse_anypaint_mask", "anypaint-mask", 1}},
	}
	values := make(map[string][]string, len(specs))
	for _, item := range specs {
		values[item.role], err = s.persistImageInputSpec(r, root, item.spec)
		if err != nil {
			return nil, err
		}
	}
	options.identityPath = firstImagePath(values["identity"])
	options.reidPath = firstImagePath(values["sequence_character"])
	options.identityRefPaths = values["identity_reference"]
	options.depthPath = firstImagePath(values["depth"])
	options.identityMaskPath = firstImagePath(values["identity_mask"])
	options.strictMaskPath = firstImagePath(values["strict_mask"])
	options.visionPaths = values["vision"]
	options.styleRefPaths = values["style_reference"]
	options.nk2ePath = firstImagePath(values["nk2e"])
	options.anypaintPath = firstImagePath(values["anypaint"])
	options.anypaintMaskPath = firstImagePath(values["anypaint_mask"])
	if options.identityPath != "" && options.depthPath != "" {
		options.preparePoseRef = true
	}
	return references, nil
}

func (s *Server) persistImageInputSpec(r *http.Request, root string, spec imageInputSpec) ([]string, error) {
	directory := root
	if spec.directory != "" {
		directory = filepath.Join(root, spec.directory)
	}
	paths, err := saveUploads(r, spec.uploadField, directory, spec.limit)
	if err != nil {
		return nil, err
	}
	return s.appendReusedImageInputs(r, spec.reuseField, directory, spec.limit, paths)
}
