package server

type videoConditioningInput struct {
	Path     string
	FrameIdx int
	Strength float64
	Role     string
}

type savedVideoCondition struct {
	Role     string  `json:"role"`
	Index    int     `json:"index,omitempty"`
	FrameIdx int     `json:"frame_idx"`
	Strength float64 `json:"strength"`
}

type savedVideoAudioClip struct {
	Index       int     `json:"index"`
	SourceJobID string  `json:"source_job_id"`
	Start       float64 `json:"start"`
	Duration    float64 `json:"duration,omitempty"`
}
