package server

import (
	"context"
	"encoding/json"
	"mediaapp/internal/jobs"
	"os"
	"strconv"
)

func (s *Server) runVideo(ctx context.Context, j jobs.Job, effectivePrompt string, conditions []videoConditioningInput, audioPaths []string, audioStarts []float64, width, height, frames int, fps float64, seed int64) {
	cfg := s.config()
	params := decodeVideoJobParams(j.Params)
	s.heavyMu.Lock()
	defer s.heavyMu.Unlock()
	fields := map[string]string{
		"prompt": effectivePrompt,
		"width":  strconv.Itoa(width), "height": strconv.Itoa(height),
		"num_frames": strconv.Itoa(frames), "fps": strconv.FormatFloat(fps, 'f', -1, 64),
		"seed":         strconv.FormatInt(seed, 10),
		"operation_id": j.ID,
	}
	motionStrength := 0.0
	if params.MotionLoRAEnabled {
		motionStrength = params.MotionLoRAStrength
	}
	fields["motion_lora_strength"] = strconv.FormatFloat(motionStrength, 'f', -1, 64)
	acceleration := params.AccelerationRequested
	if acceleration == "" {
		acceleration = cfg.Video.Acceleration
	}
	fields["acceleration"] = acceleration
	paths := make([]string, 0, len(conditions))
	frameIndices := make([]int, 0, len(conditions))
	strengths := make([]float64, 0, len(conditions))
	for _, condition := range conditions {
		paths = append(paths, condition.Path)
		frameIndices = append(frameIndices, condition.FrameIdx)
		strengths = append(strengths, condition.Strength)
	}
	frameJSON, _ := json.Marshal(frameIndices)
	strengthJSON, _ := json.Marshal(strengths)
	fields["frame_indices"] = string(frameJSON)
	fields["image_strengths"] = string(strengthJSON)
	name := j.ID + ".mp4"
	output := s.jobs.OutputPath(name)
	files := map[string][]string{"images": paths}
	if len(audioPaths) > 0 {
		files["audios"] = audioPaths
		startsJSON, _ := json.Marshal(audioStarts)
		fields["audio_start_times"] = string(startsJSON)
		fields["audio_max_duration"] = strconv.FormatFloat(float64(frames)/fps, 'f', -1, 64)
		j.Params["stage"] = "a2v"
		_ = s.jobs.Save(j)
	}
	observer := s.startRuntimeObserver(ctx, j.ID, cfg.Engines["video"].Endpoint)
	headers, err := s.generateVideoWithEngine(ctx, fields, files, output)
	observer.Stop()
	if err != nil {
		_ = os.Remove(output)
		if s.requeueGenerationAfterEngineConflict(j, err) {
			return
		}
		s.fail(j, err)
		return
	}
	if actual := headers.Get("X-LTX-Acceleration"); actual != "" {
		j.Params["acceleration"] = actual
	}
	completed, err := s.completeGenerationJob(&j, "/api/outputs/"+name, func() { _ = os.Remove(output) })
	if err != nil || !completed {
		return
	}
	go func() { _ = s.ensureVideoPreview(j.ID, output) }()
}
