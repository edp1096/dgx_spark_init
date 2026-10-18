package workflows

// Terms are a narrow consistency check, never a claim of semantic review.
const DocumentTermsVersion = "terms_v1"

type DocumentTerms struct {
	Formats      []string               `json:"formats,omitempty"`
	Repair       bool                   `json:"repair"`
	Status       string                 `json:"status"`
	Candidates   []DocumentEntity       `json:"candidates,omitempty"`
	Cases        []DocumentTermCase     `json:"cases,omitempty"`
	Checked      int                    `json:"checked"`
	Changed      int                    `json:"changed"`
	Unresolved   int                    `json:"unresolved"`
	LayoutBlocks int                    `json:"layout_blocks"`
	Reused       int                    `json:"reused_references,omitempty"`
	Language     *DocumentLanguageCheck `json:"language,omitempty"`
}

type DocumentLanguageCheck struct {
	Version string                   `json:"version"`
	Scanned int                      `json:"scanned_paragraphs"`
	Checked int                      `json:"checked"`
	Records []DocumentLanguageRecord `json:"records"`
}
type DocumentLanguageRecord struct {
	DocumentTermReference
	PreserveLines bool                                `json:"preserve_lines,omitempty"`
	Mentions      []DocumentLanguageMention           `json:"mentions"`
	Checks        map[string]DocumentLanguageDecision `json:"checks,omitempty"`
	Attempts      int                                 `json:"attempts"`
}
type DocumentLanguageMention struct {
	ID    string `json:"id"`
	Start int    `json:"start"`
	End   int    `json:"end"`
	Text  string `json:"text"`
}
type DocumentLanguageDecision struct {
	Status    string `json:"status"`
	SourceRef string `json:"source_ref"`
	After     string `json:"after"`
	Reason    string `json:"reason"`
}

type DocumentTermCase struct {
	Source     string                  `json:"source,omitempty"`
	Pinned     string                  `json:"pinned,omitempty"`
	Status     string                  `json:"status"`
	Reason     string                  `json:"reason,omitempty"`
	References []DocumentTermReference `json:"references"`
	Decisions  []DocumentTermDecision  `json:"decisions,omitempty"`
	Attempts   map[string]int          `json:"attempts,omitempty"`
}

type DocumentTermReference struct {
	BeforeOutput map[string]string       `json:"before_output,omitempty"`
	ID           string                  `json:"id"`
	Ordinal      int                     `json:"ordinal"`
	InputIDs     []string                `json:"input_ids"`
	Source       string                  `json:"source,omitempty"`
	Target       string                  `json:"target,omitempty"`
	SourceHash   string                  `json:"source_hash"`
	TargetHash   string                  `json:"target_hash"`
	Status       string                  `json:"status"`
	Forms        []string                `json:"forms,omitempty"`
	Evidence     string                  `json:"evidence,omitempty"`
	Referent     string                  `json:"referent,omitempty"`
	Reason       string                  `json:"reason,omitempty"`
	Patches      []DocumentTermPatch     `json:"patches,omitempty"`
	Preserved    []DocumentTermPreserved `json:"preserved_mentions,omitempty"`
	ResultHash   string                  `json:"result_hash,omitempty"`
}

type DocumentTermPreserved struct {
	Source string `json:"source"`
	Target string `json:"target"`
	Reason string `json:"reason"`
}

type DocumentTermDecision struct {
	References []string `json:"references"`
	Canonical  string   `json:"canonical"`
	Status     string   `json:"status"`
	Reason     string   `json:"reason"`
}

type DocumentTermPatch struct {
	Before string `json:"before"`
	After  string `json:"after"`
}
