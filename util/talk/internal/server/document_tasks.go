package server

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"sparktalk/internal/llm"
	"sparktalk/internal/workflows"
)

type documentTurnEffortKey struct{}

type documentWorker struct {
	session string
	cancel  context.CancelFunc
	done    chan struct{}
}
type documentStart struct {
	Profiles              *workflows.DocumentProfiles    `json:"-"`
	Pipeline              string                         `json:"-"`
	RequestSpec           *workflows.DocumentRequestSpec `json:"-"`
	InstructionMessageIDs []int64                        `json:"instruction_message_ids,omitempty"`
	ExtendBudget          bool                           `json:"extend_budget,omitempty"`
	Glossary              map[string]string              `json:"glossary,omitempty"`
	Key                   string                         `json:"-"`
	Revision              int                            `json:"-"`
	Parent                string                         `json:"-"`
	Action                string                         `json:"action"`
	Kind                  string                         `json:"kind"`
	Sources               []string                       `json:"sources"`
	SourceID              string                         `json:"source_id,omitempty"`
	Instructions          string                         `json:"instructions"`
	Effort                string                         `json:"effort"`
	Format                string                         `json:"format"`
	RunID                 string                         `json:"run_id"`
	Start                 int                            `json:"start"`
	End                   int                            `json:"end"`
}

func documentDefinition() llm.Tool {
	return llm.Tool{Type: "function", Function: llm.ToolFunction{Name: "document_task", Description: "Start a durable whole-document translation, summary or multi-source comparison when requested. First obtain source_id with web_fetch or attachment_read, then pass sources: [source_id] when starting. A single source_id string is also accepted. action=start requires kind and a saved source; instructions carries the requested scope/style. Processing continues across chunks and browser disconnects. Use instructions to preserve the user's exact scope/style; default scope is the entire source. start/end optionally select character offsets in a single source. Returns actual run/progress/download URLs, not a completed deliverable. Never claim completion before status completed. action=reexport rebuilds an existing document format from a completed task using saved text in the background, keeping the current file until replacement succeeds; poll action=status for conversions. The top-level status describes the original translation; reconversion is complete only when the matching conversions entry has status=completed. It never repeats translation. action=review_terms checks recurring names and unintended foreign-language wording in a completed Korean translation. action=repair_terms uses saved source/text to fix verified name conflicts, wrong-language words or connected phrases, and TXT prose wrapping while preserving the original result. It also checks expressions outside the glossary, including a third-language pronoun used instead of a Korean pronoun. Legitimate names, acronyms, URLs and original-language glosses are preserved. Report the returned checked counts, actual corrections and remaining warnings. action=status reads an existing run; pause/resume/cancel control it. For a requested DOCX/PDF/HWP/HWPX deliverable, pass format when starting; it will be converted after processing. Document processing inherits the chat reasoning setting unless effort is explicitly specified; off/none selects a separate faster document setting. This setting does not change chat preferences. The glossary argument is only for spellings explicitly supplied by the user; omit inferred terms. This read-only processing tool cannot execute SSH, modify code or publish/upload anything.", Parameters: json.RawMessage(`{"type":"object","properties":{"action":{"type":"string","enum":["start","status","pause","resume","cancel","revise","export","reexport","retry_assets","retry_quality","review_terms","repair_terms"]},"kind":{"type":"string","enum":["translate","summarize","compare"]},"sources":{"type":"array","description":"Saved source IDs returned by web_fetch or attachment_read. Required for start unless source_id supplies one source.","items":{"type":"string","minLength":1},"minItems":1,"maxItems":8},"source_id":{"type":"string","minLength":1,"description":"Single-source shorthand for sources: [source_id]. Prefer sources; never pass conflicting values."},"instructions":{"type":"string"},"instruction_message_ids":{"type":"array","maxItems":8,"items":{"type":"integer","minimum":1},"description":"Optional exact IDs of earlier user messages that supplied relevant document instructions. Assistant/tool messages cannot authorize constraints."},"glossary":{"type":"object","maxProperties":256,"additionalProperties":{"type":"string"}},"format":{"type":"string","enum":["docx","pdf","hwp","hwpx"]},"effort":{"type":"string","enum":["off","none","low","medium","xhigh","on","inherit"]},"run_id":{"type":"string"},"extend_budget":{"type":"boolean","description":"Renew processing limits only when the user explicitly asks to continue a budget-paused task."},"start":{"type":"integer","minimum":0},"end":{"type":"integer","minimum":0}},"required":["action"],"additionalProperties":false}`)}}
}
func (s *Server) registerDocumentTasks(r *completionToolRegistry, session string) {
	r.register(documentDefinition(), func(ctx context.Context, call llm.ToolCall, _ []llm.Message, _ eventEmitter) (registeredToolResult, error) {
		var a documentStart
		if !json.Valid([]byte(call.Function.Arguments)) {
			return registeredToolResult{}, fmt.Errorf("document_task requires one valid JSON object")
		}
		decoder := json.NewDecoder(strings.NewReader(call.Function.Arguments))
		decoder.DisallowUnknownFields()
		if e := decoder.Decode(&a); e != nil {
			return registeredToolResult{}, fmt.Errorf("document_task arguments: %w; action=start uses kind, sources: [source_id], instructions", e)
		}
		if (a.Action == "start" || a.Action == "revise") && (a.Effort == "" || a.Effort == "inherit") {
			if effort, ok := ctx.Value(documentTurnEffortKey{}).(string); ok && effort != "" {
				a.Effort = effort
			}
		}
		if turn := turnFrom(ctx); turn != nil && turn.anchor > 0 && call.ID != "" && a.Action == "start" {
			a.Key = fmt.Sprintf("%d/%s", turn.anchor, call.ID)
		}
		if a.Action == "start" || a.Action == "revise" || a.Action == "repair_terms" || a.Action == "review_terms" {
			if err := s.bindDocumentRequest(ctx, session, &a); err != nil {
				return registeredToolResult{}, err
			}
		}
		if a.Action == "start" && a.Kind == "translate" {
			a.Pipeline = workflows.DocumentTermsVersion
		}
		run, e := s.documentAction(ctx, session, a)
		if e != nil {
			return registeredToolResult{}, e
		}
		data, _ := json.Marshal(s.documentResultView(run))
		result := registeredToolResult{Result: string(data)}
		if a.Action == "start" || a.Action == "resume" || a.Action == "revise" || a.Action == "retry_assets" || a.Action == "retry_quality" || a.Action == "review_terms" || a.Action == "repair_terms" {
			result.DeferredDocument = &run
		}
		return result, nil
	})
	r.prompts = append(r.prompts, "For requested whole-document translation, lengthy document summary or multi-source comparison, use document_task after acquiring source_id. A source preview is not the full source. Do not reduce the requested scope to the first chapter or ask whether to continue when the user already requested the whole document. Preserve the user's requested language/style in instructions. A started task is not yet complete; report its actual status and progress URL. To convert completed results to DOCX/PDF/HWP/HWPX use document_task action=export, format=...; the server reads saved results, so never reproduce the whole book in document_generate arguments. Ordinary questions should be answered normally without starting a task.")
}
func documentView(run workflows.Run) map[string]any {
	base := "/api/document-tasks/" + run.ID + "?session_id=" + run.SessionID
	formats := []string{"txt", "md", "html", "zip"}
	for _, format := range run.Document.Formats {
		present := false
		for _, existing := range formats {
			present = present || existing == format
		}
		if !present {
			formats = append(formats, format)
		}
	}
	outputs := map[string]string{}
	drafts := map[string]string{}
	if draft := run.Document.DraftPublication; draft != nil {
		for _, file := range draft.Files {
			drafts[strings.TrimPrefix(file.Name, "output.")] = "/api/document-tasks/" + run.ID + "/draft/" + file.Name + "?session_id=" + run.SessionID
		}
	}
	if run.Document.Pipeline != "" {
		if publication := run.Document.Publication; publication != nil {
			for _, file := range publication.Files {
				format := strings.TrimPrefix(file.Name, "output.")
				outputs[format] = "/api/document-tasks/" + run.ID + "/" + file.Name + "?session_id=" + run.SessionID
			}
		}
	} else if run.Current >= 2 {
		for _, format := range formats {
			outputs[format] = "/api/document-tasks/" + run.ID + "/output." + format + "?session_id=" + run.SessionID
		}
	}
	conversionPending := false
	for _, job := range run.Conversions {
		conversionPending = conversionPending || job.Status == "running"
		if job.Current {
			url := "/api/document-tasks/" + run.ID + "/output." + job.Format + "?session_id=" + run.SessionID
			if run.Document.Pipeline != "" && run.Document.Publication == nil {
				drafts[job.Format] = strings.Replace(url, "/output.", "/draft/output.", 1)
			} else {
				outputs[job.Format] = url
			}
		}
	}
	return map[string]any{"conversion_pending": conversionPending, "conversions": run.Conversions, "run_id": run.ID, "status": run.Status, "finished": run.DocumentFinished(), "document": documentPublicTask(run.Document), "progress_url": base, "outputs_ready": len(outputs) > 0, "outputs": outputs, "draft_outputs": drafts}
}

func (s *Server) documentAction(ctx context.Context, session string, a documentStart) (workflows.Run, error) {
	if _, e := s.db.Session(session); e != nil {
		return workflows.Run{}, e
	}
	if a.Action == "start" {
		return s.startDocument(ctx, session, a)
	}
	run, e := s.db.DocumentRun(session, a.RunID)
	if e != nil {
		return run, e
	}
	switch a.Action {
	case "review_terms", "repair_terms":
		if !s.docExportMu.TryLock() {
			return run, fmt.Errorf("document conversion already running")
		}
		s.docTaskMu.Lock()
		run, e = s.db.DocumentRun(session, a.RunID)
		// The published terminal checkpoint is authoritative. Its worker can
		// still be removing its bookkeeping entry after publication; rejecting
		// that tiny cleanup window makes an immediately requested recheck fail.
		if e == nil && !run.DocumentFinished() {
			e = fmt.Errorf("완료된 번역 작업이 필요합니다")
		}
		if e == nil {
			run, e = s.createDocumentTermsRevision(run, a.Action == "repair_terms", a.RequestSpec)
		}
		s.docTaskMu.Unlock()
		s.docExportMu.Unlock()
		if e != nil {
			return run, e
		}
		if run.Status == "running" || run.DocumentFinished() {
			return run, nil
		}
		return s.documentAction(ctx, session, documentStart{Action: "resume", RunID: run.ID})
	case "retry_quality":
		if !s.docExportMu.TryLock() {
			return run, fmt.Errorf("document conversion already running")
		}
		s.docTaskMu.Lock()
		run, e = s.db.DocumentRun(session, a.RunID)
		if e == nil && (run.Status == "running" || run.Status == "canceled" || run.Document.Kind != "translate" || s.docTasks[a.RunID] != nil) {
			e = fmt.Errorf("translation repair requires an inactive translation task")
		}
		if e == nil {
			run, e = s.createDocumentQualityRevision(run)
		}
		s.docTaskMu.Unlock()
		s.docExportMu.Unlock()
		if e != nil {
			return run, e
		}
		return s.documentAction(ctx, session, documentStart{Action: "resume", RunID: run.ID})
	case "retry_assets":
		if run.Document.Pipeline != "" {
			s.docTaskMu.Lock()
			if run.Status == "running" || run.Status == "canceled" || !run.DocumentFinished() || s.docTasks[run.ID] != nil {
				s.docTaskMu.Unlock()
				return run, fmt.Errorf("illustration revision requires an inactive completed translation")
			}
			child, err := s.createDocumentAssetRevision(run)
			s.docTaskMu.Unlock()
			if err != nil {
				return run, err
			}
			return s.documentAction(ctx, session, documentStart{Action: "resume", RunID: child.ID})
		}
		release, err := s.documentAssetSlot(ctx)
		if err != nil {
			return run, err
		}
		defer release()
		if run.Status == "running" || run.Status == "canceled" || run.Current < 2 || run.Document.Kind != "translate" {
			return run, fmt.Errorf("illustrations can be retried only on an inactive translation task")
		}
		reset := 0
		for _, id := range run.Document.Sources {
			book, err := s.db.Source(session, id, true)
			if err != nil {
				return run, err
			}
			for i := range book.Assets {
				if book.Assets[i].Error != "" {
					book.Assets[i].Pending = true
					book.Assets[i].Error = ""
					reset++
				}
			}
			warnings := book.Warnings[:0]
			for _, w := range book.Warnings {
				if !strings.HasPrefix(w, "image ") {
					warnings = append(warnings, w)
				}
			}
			book.Warnings = warnings
			if err = s.db.SaveSourceAssets(session, id, book); err != nil {
				return run, err
			}
		}
		if reset == 0 {
			return run, fmt.Errorf("no failed illustrations to retry")
		}
		warnings := run.Document.Warnings[:0]
		for _, w := range run.Document.Warnings {
			if !strings.HasPrefix(w, "image ") {
				warnings = append(warnings, w)
			}
		}
		run.Document.Warnings = warnings
		run.Status = "paused"
		if e = s.db.SaveDocumentProgress(&run); e != nil {
			return run, e
		}
		return s.documentAction(ctx, session, documentStart{Action: "resume", RunID: run.ID})
	case "reexport":
		return s.startDocumentReconversion(run, a.Format)
	case "export":
		if e = s.convertDocumentTask(ctx, run, a.Format); e != nil {
			return run, e
		}
		return s.db.DocumentRun(session, run.ID)
	case "status":
		return run, nil
	case "revise":
		if a.Instructions == "" {
			return run, fmt.Errorf("new instructions required")
		}
		if _, e = s.documentAction(ctx, session, documentStart{Action: "pause", RunID: run.ID}); e != nil {
			return run, e
		}
		return s.startDocument(ctx, session, documentStart{Pipeline: run.Document.Pipeline, RequestSpec: a.RequestSpec, Kind: run.Document.Kind, Sources: run.Document.Sources, Glossary: run.Document.UserGlossary, Instructions: a.Instructions, Effort: a.Effort, Format: run.Document.RequestedFormat, Start: run.Document.Start, End: run.Document.End, Revision: run.Document.Revision + 1, Parent: run.ID})
	case "pause", "cancel":
		s.docTaskMu.Lock()
		w := s.docTasks[a.RunID]
		if w != nil {
			w.cancel()
		}
		s.docTaskMu.Unlock()
		if w != nil {
			select {
			case <-w.done:
			case <-ctx.Done():
				return run, ctx.Err()
			}
		}
		s.docTaskMu.Lock()
		defer s.docTaskMu.Unlock()
		if s.docTasks[a.RunID] != nil {
			return run, fmt.Errorf("task was concurrently resumed")
		}
		run, e = s.db.DocumentRun(session, a.RunID)
		if e != nil {
			return run, e
		}
		if run.DocumentFinished() {
			return run, nil
		}
		run.Status = "paused"
		if a.Action == "cancel" {
			run.Status = "canceled"
		}
		e = s.db.SaveDocumentProgress(&run)
		return run, e
	case "resume":
		if run.DocumentFinished() {
			return run, fmt.Errorf("document task already finished; review warnings or use a specific repair action")
		}
		budgetReached := run.Document.ElapsedSeconds >= run.Document.MaxSeconds || run.Document.Calls >= run.Document.MaxCalls || run.Document.OutputTokens >= run.Document.MaxOutputTokens
		if budgetReached && !a.ExtendBudget {
			return run, fmt.Errorf("task budget reached; resume with extend_budget=true to authorize another bounded processing period")
		}
		cfg, _ := s.snapshot()
		if !run.Document.OutputOnly && (cfg.Model.DefaultModel != run.Document.Model || cfg.Model.Endpoint != run.Document.Endpoint || cfg.Model.ModelType != run.Document.ModelType) {
			return run, fmt.Errorf("task model/settings differ from the current model; select the original model before resuming")
		}
		if a.Instructions != "" && a.Instructions != run.Document.Instructions {
			return run, fmt.Errorf("changed instructions require a new task; completed outputs will not be silently mixed")
		}
		s.docTaskMu.Lock()
		defer s.docTaskMu.Unlock()
		if s.docTasks[a.RunID] != nil {
			return run, fmt.Errorf("task already running")
		}
		units, err := s.db.DocumentUnits(session, a.RunID)
		if err != nil {
			return run, err
		}
		for i := range units {
			if units[i].Status != "completed" {
				units[i].Status = "pending"
				units[i].RetryBase = len(units[i].Attempts)
			}
		}
		run, e = s.db.ClaimDocumentRun(session, a.RunID)
		if e != nil {
			return run, e
		}
		if budgetReached && a.ExtendBudget {
			run.Document.MaxCalls = max(run.Document.MaxCalls, run.Document.Calls+2048)
			run.Document.MaxOutputTokens = max(run.Document.MaxOutputTokens, run.Document.OutputTokens+2_000_000)
			run.Document.MaxSeconds = max(run.Document.MaxSeconds, run.Document.ElapsedSeconds+12*60*60)
		}
		if e = s.db.SaveDocumentProgress(&run, units...); e != nil {
			return run, e
		}
		if e = s.launchDocumentLocked(run); e != nil {
			run.Status = "paused"
			_ = s.db.SaveDocumentProgress(&run)
		}
		return run, e
	default:
		return run, fmt.Errorf("unknown document action")
	}
}
func (s *Server) startDocument(ctx context.Context, session string, a documentStart) (workflows.Run, error) {
	var run workflows.Run
	if a.SourceID != "" {
		if len(a.Sources) == 0 {
			a.Sources = []string{a.SourceID}
		} else if len(a.Sources) != 1 || a.Sources[0] != a.SourceID {
			return run, fmt.Errorf("document_task sources and source_id conflict; provide sources only, or one matching source_id")
		}
		a.SourceID = "" // Canonicalize before hashing for idempotent retries.
	}
	if len(a.Glossary) > 256 {
		return run, fmt.Errorf("glossary exceeds 256 terms")
	}
	for source, target := range a.Glossary {
		if source == "" || strings.TrimSpace(target) == "" || len(source) > 200 || len(target) > 200 {
			return run, fmt.Errorf("invalid glossary term")
		}
	}
	if len(a.Key) > 256 {
		return run, fmt.Errorf("idempotency key too long")
	}
	digestInput, _ := json.Marshal(a)
	hash := sha256.Sum256(digestInput)
	requestHash := hex.EncodeToString(hash[:])
	if a.Key != "" {
		if existing, err := s.db.DocumentRequest(session, a.Key); err == nil {
			if existing.Document.RequestHash != requestHash {
				return run, fmt.Errorf("idempotency key reused with different task inputs")
			}
			return existing, nil
		}
	}
	if a.Format != "" && a.Format != "docx" && a.Format != "pdf" && a.Format != "hwp" && a.Format != "hwpx" {
		return run, fmt.Errorf("invalid requested output format")
	}
	if a.Kind != "translate" && a.Kind != "summarize" && a.Kind != "compare" {
		return run, fmt.Errorf("unsupported document task")
	}
	if len(a.Sources) == 0 {
		return run, fmt.Errorf("document_task action=start requires sources: [\"<source_id from web_fetch or attachment_read>\"] (or source_id for one document)")
	}
	if len(a.Sources) > 8 {
		return run, fmt.Errorf("document_task supports at most 8 sources; received %d", len(a.Sources))
	}
	for _, id := range a.Sources {
		if strings.TrimSpace(id) == "" {
			return run, fmt.Errorf("document_task sources must contain nonempty saved source IDs")
		}
	}
	if len(a.Instructions) > 16000 {
		return run, fmt.Errorf("document_task instructions exceed 16000 UTF-8 bytes; received %d", len(a.Instructions))
	}
	if a.Kind == "translate" && len(a.Sources) != 1 {
		return run, fmt.Errorf("translate one document per task")
	}
	if a.Kind == "compare" && len(a.Sources) < 2 {
		return run, fmt.Errorf("comparison needs at least two sources")
	}
	if a.Start < 0 || a.End < 0 || a.End > 0 && a.End <= a.Start || len(a.Sources) > 1 && (a.Start != 0 || a.End != 0) {
		return run, fmt.Errorf("invalid source range")
	}
	cfg, _ := s.snapshot()
	effort := a.Effort
	if effort == "" || effort == "inherit" {
		effort = cfg.Model.ReasoningEffort
	}
	if effort == "" || effort == "off" {
		effort = "none"
	}
	effort = llm.NormalizeReasoningEffort(cfg.Model.ModelType, effort)
	run = workflows.Run{ID: newID(), SessionID: session, Status: "paused", Goal: a.Instructions, Definition: workflows.Definition{Name: "document-" + a.Kind, Enabled: true, Builtin: true}, Document: &workflows.DocumentTask{UserGlossary: a.Glossary, Phase: "process", RequestedFormat: a.Format, RequestID: a.Key, RequestHash: requestHash, Formats: []string{"txt", "md", "html", "zip"}, Kind: a.Kind, Sources: a.Sources, Model: cfg.Model.DefaultModel, ModelType: cfg.Model.ModelType, Endpoint: cfg.Model.Endpoint, Effort: effort, OutputLimit: cfg.Context.EffectiveOutputReserve(cfg.Context.WindowTokens), Instructions: a.Instructions, Revision: max(1, a.Revision), Parent: a.Parent, Start: a.Start, End: a.End, MaxCalls: 2048, MaxOutputTokens: 2_000_000, MaxSeconds: 12 * 60 * 60}}
	// Use a conservative rune ceiling until the backend's token budget is known.
	units := []workflows.DocumentUnit{}
	current := workflows.DocumentUnit{Stage: "process", Status: "pending"}
	chars := 0
	seen := map[string]bool{}
	for _, id := range a.Sources {
		if seen[id] {
			return run, fmt.Errorf("duplicate source")
		}
		seen[id] = true
		b, e := s.db.Source(session, id, false)
		if e != nil {
			return run, e
		}
		if b.Incomplete {
			return run, fmt.Errorf("source is an incomplete preview; acquire the complete document before starting a task")
		}
		if a.Kind == "translate" {
			planned := planDocumentTranslation(id, b, a.Start, a.End)
			for i := range planned {
				planned[i].Ordinal += len(units)
			}
			units = append(units, planned...)
		}
		for _, block := range b.Blocks {
			if a.Kind == "translate" {
				break
			} // translation uses structural paragraphs above
			if a.End > 0 && block.Start >= a.End || block.End <= a.Start {
				continue
			}
			text := []rune(block.Text)
			start := max(0, a.Start-block.Start)
			end := len(text)
			if a.End > 0 {
				end = min(end, a.End-block.Start)
			}
			if start >= end {
				continue
			}
			for pos := start; pos < end; {
				stop := min(end, pos+2200)
				if stop < end {
					for j := stop; j > pos+1100; j-- {
						if text[j-1] == '\n' || text[j-1] == '.' || text[j-1] == '。' {
							stop = j
							break
						}
					}
				}
				if chars+stop-pos > 2600 && len(current.Inputs) > 0 {
					current.Ordinal = len(units)
					units = append(units, current)
					current = workflows.DocumentUnit{Stage: "process", Status: "pending"}
					chars = 0
				}
				current.Inputs = append(current.Inputs, workflows.DocumentInput{ID: fmt.Sprintf("%s:%s:%d", id, block.ID, pos), Source: id, Block: block.ID, Chapter: block.Chapter, Part: pos, Text: string(text[pos:stop])})
				chars += stop - pos
				pos = stop
			}
		}
		if run.Document.SourceURLs == nil {
			run.Document.SourceURLs = map[string]string{}
		}
		run.Document.SourceURLs[id] = b.URL
		if run.Document.SourceTitles == nil {
			run.Document.SourceTitles = map[string]string{}
		}
		title := b.Title
		if author, ok := b.Metadata["author"].(string); ok && author != "" {
			title += " — " + author
		}
		run.Document.SourceTitles[id] = title
		run.Document.ImageTotal += len(b.Assets)
		for _, asset := range b.Assets {
			if asset.Error == "" && !asset.Pending {
				run.Document.ImagesReady++
			}
		}
		run.Document.Warnings = append(run.Document.Warnings, b.Warnings...)
	}
	if len(current.Inputs) > 0 {
		current.Ordinal = len(units)
		units = append(units, current)
	}
	if len(units) == 0 {
		return run, fmt.Errorf("selected range has no text")
	}
	run.Document.Total = len(units)
	run.Document.RequestSpec = a.RequestSpec
	if run.Document.RequestSpec == nil {
		run.Document.RequestSpec = &workflows.DocumentRequestSpec{Origin: "api", UserText: a.Instructions, Hash: documentTextHash(a.Instructions)}
	}
	if a.Pipeline != "" {
		if (a.Pipeline != workflows.DocumentPipelineVersion && a.Pipeline != workflows.DocumentTermsVersion) || a.Kind != "translate" {
			return run, fmt.Errorf("unsupported document pipeline")
		}
		run.Document.Pipeline = a.Pipeline
		if a.Pipeline == workflows.DocumentTermsVersion {
			run.Document.Terms = &workflows.DocumentTerms{Repair: true, Status: "pending"}
		}
		run.Document.Profiles = documentProfiles(cfg, effort)
		if a.Profiles != nil {
			profiles, err := documentPrepareProfiles(cfg, *a.Profiles)
			if err != nil {
				return run, err
			}
			run.Document.Profiles = profiles
		}
		for i := range units {
			units[i].Quality = &workflows.DocumentUnitQuality{Status: "pending"}
		}
	}
	run.Definition.Steps = []workflows.Step{{Name: "구간 처리", DoneWhen: "모든 대상 구간의 결과와 구조 검증"}, {Name: "종합·출력", DoneWhen: "작업별 종합 및 다운로드 생성"}}
	run.Steps = []workflows.StepState{{Status: "pending"}, {Status: "pending"}}
	if e := s.db.CreateDocumentRun(&run, units); e != nil {
		if a.Key != "" {
			if existing, err := s.db.DocumentRequest(session, a.Key); err == nil && existing.Document.RequestHash == requestHash {
				return existing, nil
			}
		}
		return run, e
	}
	s.docTaskMu.Lock()
	defer s.docTaskMu.Unlock()
	claimed, e := s.db.ClaimDocumentRun(session, run.ID)
	if e != nil {
		return run, e
	}
	run = claimed
	if e = s.launchDocumentLocked(run); e != nil {
		run.Status = "paused"
		_ = s.db.SaveDocumentProgress(&run)
	}
	return run, e
}
func (s *Server) launchDocumentLocked(run workflows.Run) error {
	if len(s.docTasks) >= 16 {
		return fmt.Errorf("document task admission limit reached (16); resume after another task finishes")
	}
	raw, err := json.Marshal(run)
	if err != nil {
		return err
	}
	var owned workflows.Run
	if err = json.Unmarshal(raw, &owned); err != nil {
		return err
	}
	run = owned
	ctx, finish, e := s.tasks.Track(context.Background())
	if e != nil {
		return e
	}
	ctx, cancel := context.WithCancel(ctx)
	if s.docTasks == nil {
		s.docTasks = map[string]*documentWorker{}
	}
	w := &documentWorker{session: run.SessionID, cancel: cancel, done: make(chan struct{})}
	s.docTasks[run.ID] = w
	go func() {
		defer finish()
		defer cancel()
		defer func() { s.docTaskMu.Lock(); delete(s.docTasks, run.ID); close(w.done); s.docTaskMu.Unlock() }()
		s.runDocument(ctx, run)
	}()
	return nil
}

// One document inference at a time. Interactive generations get admission first.
func (s *Server) documentSlot(ctx context.Context) error {
	for {
		if e := ctx.Err(); e != nil {
			return e
		}
		s.generationMu.Lock()
		busy := len(s.generations) > 0 || s.queueClearing
		s.generationMu.Unlock()
		if !busy && s.docInferMu.TryLock() {
			return nil
		}
		timer := time.NewTimer(250 * time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
}

func (s *Server) cancelSessionDocuments(ctx context.Context, session string) error {
	s.docTaskMu.Lock()
	workers := []*documentWorker{}
	for _, w := range s.docTasks {
		if w.session == session {
			w.cancel()
			workers = append(workers, w)
		}
	}
	s.docTaskMu.Unlock()
	for _, w := range workers {
		select {
		case <-w.done:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}

func (s *Server) documentAssetSlot(ctx context.Context) (func(), error) {
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if s.docAssetMu.TryLock() {
			return s.docAssetMu.Unlock, nil
		}
		timer := time.NewTimer(50 * time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, ctx.Err()
		case <-timer.C:
		}
	}
}
