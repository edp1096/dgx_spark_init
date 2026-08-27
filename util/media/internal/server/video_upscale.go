package server

import (
	"context"
	"fmt"
	"mediaapp/internal/jobs"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

func (s *Server) runVideoUpscale(ctx context.Context, j jobs.Job) {
	cfg := s.config()
	params := decodeVideoJobParams(j.Params)
	s.heavyMu.Lock()
	defer s.heavyMu.Unlock()
	j.Params["stage"] = "upscaling"
	_ = s.jobs.Save(j)

	source, ok := s.jobs.Get(params.SourceJobID)
	if !ok || (source.Kind == "video" && source.Status != "completed") {
		s.fail(j, fmt.Errorf("source video is no longer available"))
		return
	}
	fields := map[string]string{
		"scale":            strconv.FormatFloat(params.UpscaleScale, 'f', -1, 64),
		"seed":             strconv.FormatInt(params.Seed, 10),
		"batch_size":       strconv.Itoa(params.BatchSize),
		"temporal_overlap": strconv.Itoa(params.TemporalOverlap),
		"start_time":       strconv.FormatFloat(params.SourceStartTime, 'f', -1, 64),
		"end_time":         strconv.FormatFloat(params.SourceEndTime, 'f', -1, 64),
	}
	endpoint := strings.TrimRight(cfg.Engines["upscale"].Endpoint, "/") + "/v1/videos/upscale"
	output := s.jobs.OutputPath(j.ID + ".mp4")
	var headers http.Header
	var err error
	if source.Kind == "video" && source.OutputURL != "" {
		path := s.jobs.OutputPath(filepath.Base(source.OutputURL))
		err = s.callMultipartToFileStreamingContext(ctx, endpoint, fields, "video", []string{path}, output)
		headers = make(http.Header)
	} else if source.Kind == "recognition" && source.MediaAssetID != "" {
		assetURL := strings.TrimRight(cfg.Engines["media"].Endpoint, "/") + "/v1/media/assets/" + source.MediaAssetID
		headers, err = s.callRemoteVideoMultipartToFile(ctx, endpoint, assetURL, fields, output)
	} else {
		err = fmt.Errorf("source video is no longer available")
	}
	if err != nil {
		_ = os.Remove(output)
		if s.requeueGenerationAfterEngineConflict(j, err) {
			return
		}
		s.fail(j, err)
		return
	}
	j.Params["stage"] = "completed"
	if actual := headers.Get("X-SeedVR2-Seed"); actual != "" {
		if seed, parseErr := strconv.ParseInt(actual, 10, 64); parseErr == nil {
			j.Params["seed"] = seed
		}
	}
	completed, err := s.completeGenerationJob(&j, "/api/outputs/"+filepath.Base(output), func() { _ = os.Remove(output) })
	if err != nil || !completed {
		return
	}
	go func() { _ = s.ensureVideoPreview(j.ID, output) }()
}
