package server

import (
	"fmt"
	"mediaapp/internal/jobs"
	"strconv"
)

type generationVideoExecution struct {
	prompt        string
	conditions    []videoConditioningInput
	audioPaths    []string
	audioStarts   []float64
	width, height int
	frames        int
	fps           float64
	seed          int64
}

func (s *Server) loadVideoExecution(job jobs.Job) (generationVideoExecution, error) {
	params := decodeVideoJobParams(job.Params)
	result := generationVideoExecution{
		prompt: params.EnhancedPrompt,
		width:  params.Width,
		height: params.Height,
		frames: params.Frames,
		fps:    params.FPS,
		seed:   params.Seed,
	}
	if result.prompt == "" {
		result.prompt = job.Prompt
	}
	if result.width == 0 {
		result.width = s.config().Video.DefaultWidth
	}
	if result.height == 0 {
		result.height = s.config().Video.DefaultHeight
	}
	if result.frames == 0 {
		result.frames = s.config().Video.DefaultFrames
	}
	if result.fps == 0 {
		result.fps = s.config().Video.DefaultFPS
	}
	if params.Audio {
		if len(params.AudioClips) == 0 {
			path, err := s.savedVideoInput(job.ID, "audio")
			if err != nil {
				return result, err
			}
			if path == "" {
				path, err = s.savedVideoInput(job.ID, "audio-0")
				if err != nil {
					return result, err
				}
			}
			if path == "" {
				return result, fmt.Errorf("saved A2V audio is missing")
			}
			result.audioPaths = []string{path}
			result.audioStarts = []float64{0}
		} else {
			for _, clip := range params.AudioClips {
				path, err := s.savedVideoInput(job.ID, "audio-"+strconv.Itoa(clip.Index))
				if err != nil {
					return result, err
				}
				if path == "" {
					return result, fmt.Errorf("saved A2V audio clip %d is missing", clip.Index+1)
				}
				result.audioPaths = append(result.audioPaths, path)
				result.audioStarts = append(result.audioStarts, clip.Start)
			}
		}
	}
	saved := params.Conditions
	if len(saved) == 0 {
		if params.StartImage {
			saved = append(saved, savedVideoCondition{Role: "start", FrameIdx: 0, Strength: params.ImageStrength})
		}
		count := params.Keyframes
		for index := 0; index < count; index++ {
			saved = append(saved, savedVideoCondition{Role: "keyframe", Index: index, FrameIdx: int(float64(result.frames-1) * float64(index+1) / float64(count+1)), Strength: 1})
		}
		if params.EndImage {
			saved = append(saved, savedVideoCondition{Role: "end", FrameIdx: result.frames - 1, Strength: 1})
		}
	}
	for _, condition := range saved {
		dir := condition.Role
		if condition.Role == "keyframe" {
			dir = "keyframe-" + strconv.Itoa(condition.Index)
		}
		path, err := s.savedVideoInput(job.ID, dir)
		if err != nil {
			return result, err
		}
		if path == "" {
			return result, fmt.Errorf("saved %s image is missing", condition.Role)
		}
		result.conditions = append(result.conditions, videoConditioningInput{Path: path, FrameIdx: condition.FrameIdx, Strength: condition.Strength, Role: condition.Role})
	}
	return result, nil
}
