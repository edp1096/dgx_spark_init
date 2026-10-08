package server

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"mediaapp/internal/jobs"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

type faceSwapEngineResponse struct {
	Data []struct {
		B64JSON string `json:"b64_json"`
	} `json:"data"`
	Model           string `json:"model"`
	TargetFaceIndex int    `json:"target_face_index"`
	SourceFaceIndex int    `json:"source_face_index"`
}

func parseFaceIndex(value string) (int, error) {
	if strings.TrimSpace(value) == "" {
		return 0, nil
	}
	index, err := strconv.Atoi(value)
	if err != nil || index < 0 || index > 15 {
		return 0, fmt.Errorf("face index must be between 0 and 15")
	}
	return index, nil
}

func (s *Server) createFaceSwap(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(128 << 20); err != nil {
		http.Error(w, "invalid or oversized face swap form", http.StatusBadRequest)
		return
	}
	targetIndex, err := parseFaceIndex(r.FormValue("target_face_index"))
	if err != nil {
		http.Error(w, "invalid target "+err.Error(), http.StatusBadRequest)
		return
	}
	sourceIndex, err := parseFaceIndex(r.FormValue("source_face_index"))
	if err != nil {
		http.Error(w, "invalid source "+err.Error(), http.StatusBadRequest)
		return
	}

	id := newID()
	inputDir := filepath.Join(s.dataDir, "inputs", id)
	targets, err := saveUploads(r, "target", filepath.Join(inputDir, "face-swap-target"), 1)
	if err == nil {
		targets, err = s.appendReusedImageInputs(r, "reuse_target", filepath.Join(inputDir, "face-swap-target"), 1, targets)
	}
	if err != nil || len(targets) != 1 {
		if err == nil {
			err = fmt.Errorf("one target image is required")
		}
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	sources, err := saveUploads(r, "source", filepath.Join(inputDir, "face-swap-source"), 1)
	if err == nil {
		sources, err = s.appendReusedImageInputs(r, "reuse_source", filepath.Join(inputDir, "face-swap-source"), 1, sources)
	}
	if err != nil || len(sources) != 1 {
		if err == nil {
			err = fmt.Errorf("one source face image is required")
		}
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	targetFile, err := os.Open(targets[0])
	if err != nil {
		http.Error(w, "target image is unavailable", http.StatusBadRequest)
		return
	}
	dimensions, _, err := image.DecodeConfig(targetFile)
	_ = targetFile.Close()
	if err != nil {
		http.Error(w, "target image is invalid", http.StatusBadRequest)
		return
	}
	params := map[string]any{
		"mode": "face_swap", "model": "ReActor · INSwapper 128",
		"target_face_index": targetIndex, "source_face_index": sourceIndex,
		"width": dimensions.Width, "height": dimensions.Height,
		"stage": "queued", "queued_at": time.Now().Format(time.RFC3339Nano),
	}
	job := jobs.Job{ID: id, Kind: "image", Status: "queued", Prompt: "ReActor 얼굴 교체", Params: params, CreatedAt: time.Now()}
	if err := s.jobs.Save(job); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	s.wakeGenerationQueue()
	writeJSON(w, http.StatusAccepted, job)
}

func (s *Server) runFaceSwap(ctx context.Context, job jobs.Job) {
	targets, targetErr := s.imageInputFiles(job.ID, "face_swap_target")
	sources, sourceErr := s.imageInputFiles(job.ID, "face_swap_source")
	if targetErr != nil || sourceErr != nil || len(targets) != 1 || len(sources) != 1 {
		s.fail(job, fmt.Errorf("saved face swap images are missing"))
		return
	}
	fields := map[string]string{
		"target_face_index": strconv.Itoa(intParam(job.Params, "target_face_index", 0)),
		"source_face_index": strconv.Itoa(intParam(job.Params, "source_face_index", 0)),
		"operation_id":      job.ID,
	}
	endpoint := s.config().Engines["faceswap"].Endpoint
	observer := s.startRuntimeObserver(ctx, job.ID, endpoint)
	data, _, err := s.callMultipartFilesContext(ctx, endpoint+"/v1/faces/swap", fields, map[string][]string{
		"target": targets,
		"source": sources,
	})
	observer.Stop()
	if err != nil {
		s.fail(job, err)
		return
	}
	var response faceSwapEngineResponse
	if err := json.Unmarshal(data, &response); err != nil || len(response.Data) == 0 || response.Data[0].B64JSON == "" {
		if err == nil {
			err = fmt.Errorf("face swap engine returned no image")
		}
		s.fail(job, fmt.Errorf("decode face swap response: %w", err))
		return
	}
	result, err := base64.StdEncoding.DecodeString(response.Data[0].B64JSON)
	if err != nil {
		s.fail(job, fmt.Errorf("decode face swap image: %w", err))
		return
	}
	job.Params["model"] = response.Model
	job.Params["target_face_index"] = response.TargetFaceIndex
	job.Params["source_face_index"] = response.SourceFaceIndex
	if err := s.writeImageResult(&job, result, job.Prompt); err != nil {
		s.fail(job, err)
	}
}
