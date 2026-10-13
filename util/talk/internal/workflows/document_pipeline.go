package workflows

import (
	"sparktalk/internal/llm"
	"sparktalk/internal/performance"
)

const DocumentPipelineVersion = "reviewed_v1"

// Profiles are request-local snapshots. They never modify chat configuration.
type DocumentProfile struct {
	Sampling        *llm.SamplingParameters `json:"sampling,omitempty"`
	Model           string                  `json:"model"`
	ModelType       string                  `json:"model_type"`
	Endpoint        string                  `json:"endpoint"`
	Effort          string                  `json:"effort"`
	Temperature     float64                 `json:"temperature"`
	WindowTokens    int                     `json:"window_tokens"`
	WindowSource    string                  `json:"window_source,omitempty"`
	ReasoningBudget int                     `json:"reasoning_budget"`
}

type DocumentProfiles struct {
	Translate DocumentProfile `json:"translate"`
	Review    DocumentProfile `json:"review"`
	Repair    DocumentProfile `json:"repair"`
}

type DocumentRequestAttempt struct {
	SourceSpans              []DocumentRequestSpan `json:"source_spans,omitempty"`
	FailureKind              string                `json:"failure_kind,omitempty"`
	ExpectedUnitOutputHashes map[int]string        `json:"expected_unit_output_hashes,omitempty"`
	ID                       string                `json:"id"`
	Stage                    string                `json:"stage"`
	StartedAt                string                `json:"started_at"`
	FinishedAt               string                `json:"finished_at"`
	Profile                  DocumentProfile       `json:"profile"`
	Units                    []int                 `json:"units"`
	InputHash                string                `json:"input_hash"`
	OutputHash               string                `json:"output_hash"`
	RawContent               string                `json:"raw_content"`
	Measurement              DocumentAttempt       `json:"measurement"`
}

type DocumentRequestSpan struct {
	ID         string `json:"id"`
	WireID     string `json:"wire_id,omitempty"`
	Source     string `json:"source"`
	Block      string `json:"block"`
	Start      int    `json:"start"`
	Length     int    `json:"length"`
	Annotation bool   `json:"annotation,omitempty"`
}

type DocumentEntityEvidence struct {
	InputIDs []string `json:"input_ids,omitempty"`
	InputID  string   `json:"input_id"`
	Quote    string   `json:"quote"`
}

type DocumentEntity struct {
	ID       string                   `json:"id"`
	Version  int                      `json:"version"`
	Source   string                   `json:"source"`
	Target   string                   `json:"target"`
	Kind     string                   `json:"kind"`
	Origin   string                   `json:"origin"`
	Status   string                   `json:"status"`
	Evidence []DocumentEntityEvidence `json:"evidence"`
}

type DocumentUnitQuality struct {
	ReviewError      string               `json:"review_error,omitempty"`
	ParentRunID      string               `json:"parent_run_id,omitempty"`
	TransportRetries map[string]int       `json:"transport_retries,omitempty"`
	RepairFailures   int                  `json:"repair_failures,omitempty"`
	Status           string               `json:"status"`
	OutputHash       string               `json:"output_hash"`
	ReviewIDs        []string             `json:"review_ids,omitempty"`
	Reviews          []DocumentReviewItem `json:"reviews,omitempty"`
	Repairs          int                  `json:"repairs"`
	ReviewRetries    int                  `json:"review_retries"`
	EntityVersions   map[string]int       `json:"entity_versions,omitempty"`
}

type DocumentStageMetric struct {
	Performance  *performance.Summary `json:"performance,omitempty"`
	Calls        int                  `json:"calls"`
	Seconds      float64              `json:"seconds"`
	PromptTokens int                  `json:"prompt_tokens"`
	OutputTokens int                  `json:"output_tokens"`
}

type DocumentManifestEntry struct {
	Name  string `json:"name"`
	MIME  string `json:"mime"`
	Hash  string `json:"hash"`
	Bytes int    `json:"bytes"`
}

type DocumentPublication struct {
	TextHash    string                  `json:"text_hash"`
	PublishedAt string                  `json:"published_at"`
	Files       []DocumentManifestEntry `json:"files"`
}
