package server

type assistantChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type assistantChatRequest struct {
	Messages      []assistantChatMessage  `json:"messages"`
	State         map[string]any          `json:"state"`
	VisualContext *assistantVisualContext `json:"visual_context,omitempty"`
}

type assistantVisualContext struct {
	Kind     string   `json:"kind"`
	ImageURL string   `json:"image_url"`
	Labels   []string `json:"labels,omitempty"`
}

type assistantAction struct {
	Type            string  `json:"type"`
	Tab             string  `json:"tab,omitempty"`
	Prompt          string  `json:"prompt,omitempty"`
	Text            string  `json:"text,omitempty"`
	Instructions    string  `json:"instructions,omitempty"`
	Context         string  `json:"context,omitempty"`
	Language        string  `json:"language,omitempty"`
	Speaker         string  `json:"speaker,omitempty"`
	TargetLanguage  string  `json:"target_language,omitempty"`
	TranslationMode string  `json:"translation_mode,omitempty"`
	Width           int     `json:"width,omitempty"`
	Height          int     `json:"height,omitempty"`
	Seed            *int64  `json:"seed,omitempty"`
	FPS             float64 `json:"fps,omitempty"`
	Duration        float64 `json:"duration,omitempty"`
	EnhanceEnabled  *bool   `json:"enhance_enabled,omitempty"`
	Module          string  `json:"module,omitempty"`
	Preset          string  `json:"preset,omitempty"`
	Enabled         *bool   `json:"enabled,omitempty"`
	ImageIndex      int     `json:"image_index,omitempty"`
	Target          string  `json:"target,omitempty"`
	OutpaintLeft    int     `json:"outpaint_left,omitempty"`
	OutpaintTop     int     `json:"outpaint_top,omitempty"`
	OutpaintRight   int     `json:"outpaint_right,omitempty"`
	OutpaintBottom  int     `json:"outpaint_bottom,omitempty"`
}

type assistantChatResponse struct {
	Reply        string            `json:"reply"`
	Actions      []assistantAction `json:"actions"`
	Confirmation string            `json:"confirmation,omitempty"`
	VisionUsed   bool              `json:"vision_used,omitempty"`
}
