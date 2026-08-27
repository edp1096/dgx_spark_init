package server

import "encoding/json"

// videoJobParams is the persisted contract shared by video creation, queue
// replay, retry, and upscale execution. Its map representation intentionally
// matches the existing API so saved jobs and the web client remain compatible.
type videoJobParams struct {
	Mode                  string                `json:"mode"`
	Width                 int                   `json:"width"`
	Height                int                   `json:"height"`
	Frames                int                   `json:"num_frames,omitempty"`
	FPS                   float64               `json:"fps,omitempty"`
	Seed                  int64                 `json:"seed"`
	ImageStrength         float64               `json:"image_strength,omitempty"`
	Image                 bool                  `json:"image,omitempty"`
	StartImage            bool                  `json:"start_image,omitempty"`
	EndImage              bool                  `json:"end_image,omitempty"`
	Keyframes             int                   `json:"keyframes,omitempty"`
	Conditions            []savedVideoCondition `json:"video_conditions,omitempty"`
	Audio                 bool                  `json:"audio,omitempty"`
	AudioSourceJobID      string                `json:"audio_source_job_id,omitempty"`
	AudioSourceJobIDs     []string              `json:"audio_source_job_ids,omitempty"`
	AudioClips            []savedVideoAudioClip `json:"audio_clips,omitempty"`
	MotionLoRAEnabled     bool                  `json:"motion_lora_enabled,omitempty"`
	MotionLoRAStrength    float64               `json:"motion_lora_strength,omitempty"`
	AccelerationRequested string                `json:"acceleration_requested,omitempty"`
	Acceleration          string                `json:"acceleration,omitempty"`
	EnhancedPrompt        string                `json:"enhanced_prompt,omitempty"`
	Stage                 string                `json:"stage,omitempty"`
	QueuedAt              string                `json:"queued_at,omitempty"`

	SourceJobID     string  `json:"source_job_id,omitempty"`
	SourceKind      string  `json:"source_kind,omitempty"`
	UpscaleEngine   string  `json:"upscale_engine,omitempty"`
	Model           string  `json:"model,omitempty"`
	UpscaleScale    float64 `json:"upscale_scale,omitempty"`
	BatchSize       int     `json:"batch_size,omitempty"`
	TemporalOverlap int     `json:"temporal_overlap,omitempty"`
	SourceWidth     int     `json:"source_width,omitempty"`
	SourceHeight    int     `json:"source_height,omitempty"`
	Duration        float64 `json:"duration,omitempty"`
	SourceStartTime float64 `json:"source_start_time,omitempty"`
	SourceEndTime   float64 `json:"source_end_time,omitempty"`
}

func newVideoJobParams() videoJobParams {
	return videoJobParams{
		Mode:            "generate",
		Seed:            -1,
		ImageStrength:   1,
		UpscaleScale:    2,
		BatchSize:       5,
		TemporalOverlap: 1,
		UpscaleEngine:   "seedvr2-3b-fp8",
		Model:           "seedvr2-3b-fp8",
	}
}

func decodeVideoJobParams(values map[string]any) videoJobParams {
	result := newVideoJobParams()
	data, err := json.Marshal(values)
	if err == nil {
		_ = json.Unmarshal(data, &result)
	}
	return result
}

func (p videoJobParams) toMap() map[string]any {
	if p.Mode == "upscale" {
		result := map[string]any{
			"mode": p.Mode, "source_job_id": p.SourceJobID, "source_kind": p.SourceKind,
			"upscale_engine": p.UpscaleEngine, "model": p.Model,
			"upscale_scale": p.UpscaleScale, "batch_size": p.BatchSize,
			"temporal_overlap": p.TemporalOverlap, "seed": p.Seed,
			"width": p.Width, "height": p.Height,
			"source_width": p.SourceWidth, "source_height": p.SourceHeight,
			"duration": p.Duration, "stage": p.Stage, "queued_at": p.QueuedAt,
		}
		if p.SourceEndTime > 0 {
			result["source_start_time"] = p.SourceStartTime
			result["source_end_time"] = p.SourceEndTime
		}
		if p.FPS > 0 {
			result["fps"] = p.FPS
		}
		return result
	}
	return map[string]any{
		"width": p.Width, "height": p.Height, "num_frames": p.Frames,
		"fps": p.FPS, "seed": p.Seed, "image_strength": p.ImageStrength,
		"image": p.Image, "start_image": p.StartImage, "end_image": p.EndImage,
		"keyframes": p.Keyframes, "video_conditions": p.Conditions,
		"audio": p.Audio, "audio_source_job_id": p.AudioSourceJobID,
		"audio_source_job_ids": p.AudioSourceJobIDs, "audio_clips": p.AudioClips,
		"mode": p.Mode, "motion_lora_enabled": p.MotionLoRAEnabled,
		"motion_lora_strength":   p.MotionLoRAStrength,
		"acceleration_requested": p.AccelerationRequested,
		"enhanced_prompt":        p.EnhancedPrompt, "stage": p.Stage, "queued_at": p.QueuedAt,
	}
}
