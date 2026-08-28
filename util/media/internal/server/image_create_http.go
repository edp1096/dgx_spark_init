package server

import (
	"fmt"
	"mediaapp/internal/jobs"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

func (s *Server) createImage(w http.ResponseWriter, r *http.Request) {
	cfg := s.config()
	if err := r.ParseMultipartForm(80 << 20); err != nil {
		http.Error(w, "invalid form", http.StatusBadRequest)
		return
	}
	form, err := parseImageCreateForm(r, cfg)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	rootID := newID()
	references, err := s.persistImageCreateInputs(r, rootID, form.Mode, cfg.Image.MaxReferenceImages, &form.Options)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := validateImageCreate(form.Mode, references, form.Width, form.Height, form.Sequence.Prompts, &form.ControlType, form.ControlStrength, &form.Options); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	params := imageJobParamsFromOptions(
		form.Width, form.Height, form.Seed, len(references), form.Mode,
		cfg.Image.Backends[form.Mode].Model, form.ControlType, form.ControlStrength, form.Options,
	)
	params.IdentityPreserveItems = form.IdentityPreserveItems
	params.IdentityPreserveCustom = form.IdentityPreserveCustom
	if form.ParentJobID != "" {
		parent, ok := s.jobs.Get(form.ParentJobID)
		if !ok || parent.Kind != "image" || parent.Status != "completed" {
			http.Error(w, "parent image no longer exists", http.StatusBadRequest)
			return
		}
		params.ParentJobID = form.ParentJobID
	}
	params.EnhancedPrompt = valueIfDifferent(form.EffectivePrompt, form.OriginalPrompt)
	params.Stage = "queued"
	now := time.Now()
	params.QueuedAt = now.Format(time.RFC3339Nano)

	sequenceBase, err := s.resolveImageSequenceBase(form)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := validateImageSequenceMasks(r, form.Sequence); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	planned := buildImageJobPlan(rootID, form, params, now, sequenceBase)
	if err := s.persistImageJobPlan(r, planned); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	s.wakeGenerationQueue()
	if len(form.Sequence.Prompts) > 0 {
		writeJSON(w, http.StatusAccepted, map[string]any{"sequence_id": rootID, "jobs": planned})
		return
	}
	writeJSON(w, http.StatusAccepted, planned[0])
}

func (s *Server) resolveImageSequenceBase(form imageCreateForm) (*jobs.Job, error) {
	if form.Sequence.BaseJobID == "" {
		return nil, nil
	}
	if len(form.Sequence.Prompts) == 0 {
		return nil, fmt.Errorf("sequence base requires sequence prompts")
	}
	base, ok := s.jobs.Get(form.Sequence.BaseJobID)
	if !ok || base.Kind != "image" || base.Status != "completed" || base.OutputURL == "" {
		return nil, fmt.Errorf("selected sequence base image is not available")
	}
	return &base, nil
}

func validateImageSequenceMasks(r *http.Request, sequence imageSequenceForm) error {
	for index := 1; index < len(sequence.Regions); index++ {
		if sequence.Regions[index] == "custom" && len(r.MultipartForm.File[fmt.Sprintf("sequence_mask_%d", index)]) != 1 {
			return fmt.Errorf("scene %d requires a painted mask", index+1)
		}
	}
	return nil
}

func (s *Server) persistImageJobPlan(r *http.Request, planned []jobs.Job) error {
	if len(planned) > 1 && decodeImageJobParams(planned[0].Params).SequenceReID {
		paths, err := s.imageInputFiles(planned[0].ID, "sequence_character")
		if err != nil || len(paths) != 1 {
			return fmt.Errorf("sequence character reference is missing")
		}
		for _, child := range planned[1:] {
			directory := filepath.Join(s.dataDir, "inputs", child.ID, "sequence-character")
			if err := os.MkdirAll(directory, 0o755); err != nil {
				return err
			}
			destination := filepath.Join(directory, "0"+filepath.Ext(paths[0]))
			if err := linkOrCopyFile(paths[0], destination); err != nil {
				return fmt.Errorf("persist sequence character reference: %w", err)
			}
		}
	}
	for _, job := range planned {
		params := decodeImageJobParams(job.Params)
		if params.SequenceRegion == "custom" {
			field := fmt.Sprintf("sequence_mask_%d", params.SequenceIndex-1)
			directory := filepath.Join(s.dataDir, "inputs", job.ID, "anypaint-mask")
			masks, err := saveUploads(r, field, directory, 1)
			if err != nil {
				return err
			}
			if len(masks) != 1 {
				return fmt.Errorf("scene %d requires a painted mask", params.SequenceIndex)
			}
		}
		if err := s.jobs.Save(job); err != nil {
			return err
		}
	}
	return nil
}
