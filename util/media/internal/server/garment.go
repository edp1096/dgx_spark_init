package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"mediaapp/internal/jobs"
)

var garmentTargetLabels = map[string]string{
	"all": "전체 의상", "upper": "상의", "lower": "하의", "dress": "원피스",
	"outer": "외투·상의", "shoes": "신발", "accessories": "모자·스카프·장신구",
}

var garmentTargetOrder = []string{"upper", "lower", "dress", "outer", "shoes", "accessories"}

func normalizeGarmentTargets(value string) (string, string, bool) {
	value = strings.TrimSpace(value)
	if value == "" || value == "all" {
		return "all", garmentTargetLabels["all"], true
	}
	selected := map[string]bool{}
	for _, item := range strings.Split(value, ",") {
		item = strings.TrimSpace(item)
		if item == "all" {
			return "all", garmentTargetLabels["all"], true
		}
		if _, ok := garmentTargetLabels[item]; !ok || item == "" {
			return "", "", false
		}
		selected[item] = true
	}
	keys, labels := make([]string, 0, len(selected)), make([]string, 0, len(selected))
	for _, item := range garmentTargetOrder {
		if selected[item] {
			keys = append(keys, item)
			labels = append(labels, garmentTargetLabels[item])
		}
	}
	if len(keys) == 0 {
		return "", "", false
	}
	return strings.Join(keys, ","), strings.Join(labels, " + "), true
}

type garmentEngineResponse struct {
	Model         string           `json:"model"`
	Target        string           `json:"target"`
	SelectedIndex int              `json:"selected_index"`
	Width         int              `json:"width"`
	Height        int              `json:"height"`
	Coverage      float64          `json:"coverage"`
	CutoutB64     string           `json:"cutout_b64"`
	MaskB64       string           `json:"mask_b64"`
	ReferenceB64  string           `json:"reference_b64"`
	Candidates    []map[string]any `json:"candidates"`
	Failures      []map[string]any `json:"failures"`
}

func sanitizeTransparentRGB(data []byte) ([]byte, error) {
	source, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("decode garment cutout: %w", err)
	}
	bounds := source.Bounds()
	clean := image.NewNRGBA(image.Rect(0, 0, bounds.Dx(), bounds.Dy()))
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			pixel := color.NRGBAModel.Convert(source.At(x, y)).(color.NRGBA)
			if pixel.A == 0 {
				pixel.R, pixel.G, pixel.B = 0, 0, 0
			}
			clean.SetNRGBA(x-bounds.Min.X, y-bounds.Min.Y, pixel)
		}
	}
	var output bytes.Buffer
	if err := png.Encode(&output, clean); err != nil {
		return nil, fmt.Errorf("encode garment cutout: %w", err)
	}
	return output.Bytes(), nil
}

func (s *Server) createGarmentExtraction(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(192 << 20); err != nil {
		http.Error(w, "invalid or oversized garment form", http.StatusBadRequest)
		return
	}
	target, targetLabel, ok := normalizeGarmentTargets(r.FormValue("target"))
	if !ok {
		http.Error(w, "unsupported garment target", http.StatusBadRequest)
		return
	}
	feather, err := strconv.ParseFloat(strings.TrimSpace(r.FormValue("feather")), 64)
	if err != nil && strings.TrimSpace(r.FormValue("feather")) != "" {
		http.Error(w, "invalid feather value", http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(r.FormValue("feather")) == "" {
		feather = 1
	}
	if feather < 0 || feather > 8 {
		http.Error(w, "feather must be between 0 and 8", http.StatusBadRequest)
		return
	}

	id := newID()
	inputDir := filepath.Join(s.dataDir, "inputs", id)
	sources, err := saveUploads(r, "source", filepath.Join(inputDir, "garment-source"), 1)
	if err == nil {
		sources, err = s.appendReusedImageInputs(r, "reuse_source", filepath.Join(inputDir, "garment-source"), 1, sources)
	}
	if err != nil || len(sources) != 1 {
		if err == nil {
			err = fmt.Errorf("one source image is required")
		}
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	references, err := saveUploads(r, "references", filepath.Join(inputDir, "garment-reference"), 4)
	if err == nil {
		references, err = s.appendReusedImageInputs(r, "reuse_references", filepath.Join(inputDir, "garment-reference"), 4, references)
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	sourceFile, err := os.Open(sources[0])
	if err != nil {
		http.Error(w, "source image is unavailable", http.StatusBadRequest)
		return
	}
	config, _, err := image.DecodeConfig(sourceFile)
	_ = sourceFile.Close()
	if err != nil {
		http.Error(w, "source image is invalid", http.StatusBadRequest)
		return
	}
	params := map[string]any{
		"mode": "garment_extract", "model": "fashn-ai/fashn-human-parser",
		"target": target, "target_label": targetLabel, "feather": feather,
		"garment_source": true, "garment_reference_count": len(references),
		"width": config.Width, "height": config.Height,
		"stage": "queued", "queued_at": time.Now().Format(time.RFC3339Nano),
	}
	job := jobs.Job{ID: id, Kind: "image", Status: "queued", Prompt: "의상 추출 · " + targetLabel, Params: params, CreatedAt: time.Now()}
	if err := s.jobs.Save(job); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	s.wakeGenerationQueue()
	writeJSON(w, http.StatusAccepted, job)
}

func (s *Server) runGarmentExtraction(ctx context.Context, job jobs.Job) {
	sources, err := s.imageInputFiles(job.ID, "garment_source")
	if err != nil || len(sources) != 1 {
		s.fail(job, fmt.Errorf("saved garment source is missing"))
		return
	}
	references, err := s.imageInputFiles(job.ID, "garment_reference")
	if err != nil {
		s.fail(job, err)
		return
	}
	paths := append(append([]string{}, sources...), references...)
	fields := map[string]string{
		"target":       stringParam(job.Params, "target", "all"),
		"feather":      strconv.FormatFloat(floatParam(job.Params, "feather", 1), 'f', 2, 64),
		"operation_id": job.ID,
	}
	endpoint := s.config().Engines["garment"].Endpoint
	observer := s.startRuntimeObserver(ctx, job.ID, endpoint)
	data, _, err := s.callMultipartContext(ctx, endpoint+"/v1/garments/extract", fields, "images", paths)
	observer.Stop()
	if err != nil {
		s.fail(job, err)
		return
	}
	var result garmentEngineResponse
	if err := json.Unmarshal(data, &result); err != nil {
		s.fail(job, fmt.Errorf("decode garment response: %w", err))
		return
	}
	cutout, err := base64.StdEncoding.DecodeString(result.CutoutB64)
	if err != nil {
		s.fail(job, fmt.Errorf("decode garment cutout: %w", err))
		return
	}
	cutout, err = sanitizeTransparentRGB(cutout)
	if err != nil {
		s.fail(job, err)
		return
	}
	mask, err := base64.StdEncoding.DecodeString(result.MaskB64)
	if err != nil {
		s.fail(job, fmt.Errorf("decode garment mask: %w", err))
		return
	}
	reference, err := base64.StdEncoding.DecodeString(result.ReferenceB64)
	if err != nil {
		s.fail(job, fmt.Errorf("decode garment reference: %w", err))
		return
	}
	if len(reference) == 0 {
		s.fail(job, fmt.Errorf("garment engine returned an empty reference"))
		return
	}
	maskName := job.ID + "-mask.png"
	if err := os.WriteFile(s.jobs.OutputPath(maskName), mask, 0o644); err != nil {
		s.fail(job, err)
		return
	}
	referenceName := job.ID + "-reference.png"
	if err := os.WriteFile(s.jobs.OutputPath(referenceName), reference, 0o644); err != nil {
		_ = os.Remove(s.jobs.OutputPath(maskName))
		s.fail(job, err)
		return
	}
	job.Params["model"] = result.Model
	job.Params["selected_source_index"] = result.SelectedIndex
	job.Params["coverage"] = result.Coverage
	job.Params["candidate_scores"] = result.Candidates
	job.Params["candidate_failures"] = result.Failures
	job.Params["width"] = result.Width
	job.Params["height"] = result.Height
	job.Outputs = map[string]string{
		"mask":      "/api/outputs/" + maskName,
		"reference": "/api/outputs/" + referenceName,
	}
	if err := s.writeImageResult(&job, cutout, job.Prompt); err != nil {
		s.fail(job, err)
	}
}
