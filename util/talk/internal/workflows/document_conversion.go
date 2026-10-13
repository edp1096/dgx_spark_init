package workflows

// DocumentConversion is the current file conversion attempt, independent of the
// translation run and its immutable publication manifest.
type DocumentConversion struct {
	ID         string  `json:"id"`
	Format     string  `json:"format"`
	Status     string  `json:"status"`
	StartedAt  string  `json:"started_at"`
	FinishedAt string  `json:"finished_at,omitempty"`
	Seconds    float64 `json:"seconds,omitempty"`
	InputHash  string  `json:"input_hash"`
	Converter  string  `json:"converter"`
	MIME       string  `json:"mime,omitempty"`
	Hash       string  `json:"hash,omitempty"`
	Bytes      int     `json:"bytes,omitempty"`
	Error      string  `json:"error,omitempty"`
	Current    bool    `json:"current,omitempty"`
}
