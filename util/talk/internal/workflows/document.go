package workflows

import (
	"sparktalk/internal/llm"
	"sparktalk/internal/performance"
)

// Review warnings can remain after processing and export have finished. They
// are separate from a needs_review run that stopped with unfinished work.
func (r Run) DocumentFinished() bool {
	d := r.Document
	if d != nil && d.Pipeline != "" {
		return d.Terminal
	}
	return d != nil && (r.Status == "completed" ||
		r.Status == "needs_review" && r.Current >= 2 && d.Phase == "completed" && d.Total > 0 && d.Completed >= d.Total)
}

// DocumentTask is the repeatable, read-only document phase of a workflow.
// Effectful commands remain in the existing tool/approval workflow runner.
type DocumentTask struct {
	Terms                 *DocumentTerms                 `json:"terms,omitempty"`
	Sampling              *llm.SamplingParameters        `json:"sampling,omitempty"`
	OutputOnly            bool                           `json:"output_only,omitempty"`
	FinishedAt            string                         `json:"finished_at,omitempty"`
	Pipeline              string                         `json:"pipeline,omitempty"`
	ExecutionID           string                         `json:"execution_id,omitempty"`
	Terminal              bool                           `json:"terminal,omitempty"`
	QualityStatus         string                         `json:"quality_status,omitempty"`
	Reviewed              int                            `json:"reviewed,omitempty"`
	ReviewTotal           int                            `json:"review_total,omitempty"`
	Unresolved            int                            `json:"unresolved,omitempty"`
	Profiles              *DocumentProfiles              `json:"profiles,omitempty"`
	StageMetrics          map[string]DocumentStageMetric `json:"stage_metrics,omitempty"`
	Publication           *DocumentPublication           `json:"publication,omitempty"`
	DraftPublication      *DocumentPublication           `json:"draft_publication,omitempty"`
	RequestSpec           *DocumentRequestSpec           `json:"request_spec,omitempty"`
	SourceContext         string                         `json:"source_context,omitempty"`
	EntityContext         string                         `json:"entity_context,omitempty"`
	RepairContext         string                         `json:"repair_context,omitempty"`
	Performance           *performance.Summary           `json:"performance,omitempty"`
	LastPerformance       *performance.Summary           `json:"last_performance,omitempty"`
	ActiveUnits           []int                          `json:"active_units,omitempty"`
	ReasoningBudgetTokens int                            `json:"reasoning_budget_tokens,omitempty"`
	Activity              string                         `json:"activity,omitempty"`
	ReasoningChars        int                            `json:"reasoning_chars,omitempty"`
	ContentChars          int                            `json:"content_chars,omitempty"`
	UserGlossary          map[string]string              `json:"user_glossary,omitempty"`
	GlossaryKinds         map[string]string              `json:"glossary_kinds,omitempty"`
	SourceTitles          map[string]string              `json:"source_titles,omitempty"`
	Phase                 string                         `json:"phase,omitempty"`
	RequestedFormat       string                         `json:"requested_format,omitempty"`
	RequestID             string                         `json:"request_id,omitempty"`
	RequestHash           string                         `json:"request_hash,omitempty"`
	CurrentChapter        string                         `json:"current_chapter,omitempty"`
	ImageTotal            int                            `json:"image_total"`
	ImagesReady           int                            `json:"images_ready"`
	Formats               []string                       `json:"formats,omitempty"`
	OutputLimit           int                            `json:"output_limit"`
	Parent                string                         `json:"parent,omitempty"`
	Start                 int                            `json:"start,omitempty"`
	End                   int                            `json:"end,omitempty"`
	Glossary              map[string]string              `json:"glossary,omitempty"`
	SourceURLs            map[string]string              `json:"source_urls,omitempty"`
	ContextTail           string                         `json:"context_tail,omitempty"`
	Kind                  string                         `json:"kind"`
	Sources               []string                       `json:"sources"`
	Model                 string                         `json:"model"`
	ModelType             string                         `json:"model_type"`
	Endpoint              string                         `json:"endpoint"`
	Effort                string                         `json:"effort"`
	Instructions          string                         `json:"instructions"`
	Revision              int                            `json:"revision"`
	Calls                 int                            `json:"calls"`
	MaxCalls              int                            `json:"max_calls"`
	OutputTokens          int                            `json:"output_tokens"`
	MaxOutputTokens       int                            `json:"max_output_tokens"`
	MaxSeconds            int                            `json:"max_seconds"`
	ElapsedSeconds        int                            `json:"elapsed_seconds"`
	Completed             int                            `json:"completed"`
	Total                 int                            `json:"total"`
	Warnings              []string                       `json:"warnings,omitempty"`
	Error                 string                         `json:"error,omitempty"`
}

// Only the server binds message provenance. Tool-generated instructions never
// gain user authority merely by being placed in a tool argument.
type DocumentRequestSpec struct {
	Origin     string  `json:"origin"`
	MessageIDs []int64 `json:"message_ids,omitempty"`
	UserText   string  `json:"user_text"`
	Hash       string  `json:"hash"`
	ToolHints  string  `json:"tool_hints,omitempty"`
}
type DocumentInput struct {
	Whitespace string `json:"whitespace,omitempty"`
	Scope      string `json:"scope,omitempty"`
	Plain      bool   `json:"plain,omitempty"`
	Before     string `json:"before,omitempty"`
	Group      string `json:"group,omitempty"`
	Annotation bool   `json:"annotation,omitempty"`
	Chapter    string `json:"chapter,omitempty"`
	ID         string `json:"id"`
	Source     string `json:"source"`
	Block      string `json:"block"`
	Part       int    `json:"part"`
	Text       string `json:"text"`
}
type DocumentAttempt struct {
	Performance           *performance.Summary `json:"performance,omitempty"`
	BatchUnits            []int                `json:"batch_units,omitempty"`
	SharedWithOrdinal     *int                 `json:"shared_with_ordinal,omitempty"`
	ReasoningBudgetTokens int                  `json:"reasoning_budget_tokens,omitempty"`
	OutputTokensEstimated bool                 `json:"output_tokens_estimated,omitempty"`
	Protocol              string               `json:"protocol,omitempty"`
	ReasoningChars        int                  `json:"reasoning_chars"`
	ContentChars          int                  `json:"content_chars"`
	Revision              int                  `json:"revision"`
	Model                 string               `json:"model"`
	FinishReason          string               `json:"finish_reason"`
	PromptTokens          int                  `json:"prompt_tokens"`
	OutputTokens          int                  `json:"output_tokens"`
	Seconds               float64              `json:"seconds"`
	Error                 string               `json:"error,omitempty"`
}
type DocumentUnit struct {
	Quality        *DocumentUnitQuality `json:"quality,omitempty"`
	QualityRetries int                  `json:"quality_retries,omitempty"`
	OutputBudget   int                  `json:"output_budget,omitempty"`
	RetryBase      int                  `json:"retry_base,omitempty"`
	Ordinal        int                  `json:"ordinal"`
	Stage          string               `json:"stage"`
	Status         string               `json:"status"`
	Inputs         []DocumentInput      `json:"inputs"`
	Output         map[string]string    `json:"output,omitempty"`
	Attempts       []DocumentAttempt    `json:"attempts,omitempty"`
	Warnings       []string             `json:"warnings,omitempty"`
	Error          string               `json:"error,omitempty"`
}
