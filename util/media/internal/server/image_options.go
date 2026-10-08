package server

type imageGenerationOptions struct {
	checkpoint         string
	reidPath           string
	identityPath       string
	identityRefPaths   []string
	identityPreset     string
	identityAutoPrompt bool
	identityUserPrompt bool
	identityMaskPath   string
	strictMaskPath     string
	strictMaskGrow     int
	strictMaskFeather  float64
	vaeMode            string
	identityFitMode    string
	identityModel      string
	identityEncoder    string
	depthPath          string
	depthPrompt        string
	preparePoseRef     bool
	identityStrength   float64
	refBoost           float64
	sourceRefBoost     float64
	groundingPX        int
	steps              int
	samplingPreset     string
	sampler            string
	scheduler          string
	style              string
	styleStrength      float64
	styles             []styleSelection
	userLoras          []userLoRASelection
	depthStrength      float64
	visionPaths        []string
	visionMode         string
	visionMegapixels   float64
	styleRefPaths      []string
	styleRefStrength   float64
	nk2ePath           string
	nk2eMode           string
	nk2eStrength       float64
	nk2ePreprocessed   bool
	anypaintPath       string
	anypaintMaskPath   string
	outpaintLeft       int
	outpaintTop        int
	outpaintRight      int
	outpaintBottom     int
	anypaintStrength   float64
	anypaintBoundary   int
	filterMode         string
	filterStrength     float64
	promptEnhancer     bool
	promptEnhStrength  float64
	promptTextScale    float64
}

type styleSelection struct {
	Name     string  `json:"name"`
	Strength float64 `json:"strength"`
}

type userLoRASelection struct {
	Filename string  `json:"filename"`
	Strength float64 `json:"strength"`
}
