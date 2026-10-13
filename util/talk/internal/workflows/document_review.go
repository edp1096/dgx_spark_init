package workflows

type DocumentFinding struct {
	Category    string `json:"category"`
	Severity    string `json:"severity"`
	SourceQuote string `json:"source_quote"`
	TargetQuote string `json:"target_quote"`
	Explanation string `json:"explanation"`
	Uncertain   bool   `json:"uncertain"`
}

type DocumentReviewItem struct {
	ID         string            `json:"id"`
	InputIDs   []string          `json:"input_ids"`
	SourceHash string            `json:"source_hash"`
	OutputHash string            `json:"output_hash"`
	Findings   []DocumentFinding `json:"findings"`
}

type DocumentReview struct {
	Invalid   map[string]string    `json:"invalid,omitempty"`
	ID        string               `json:"id"`
	Protocol  string               `json:"protocol"`
	Items     []DocumentReviewItem `json:"items"`
	AttemptID string               `json:"attempt_id,omitempty"`
	Status    string               `json:"status"`
}
