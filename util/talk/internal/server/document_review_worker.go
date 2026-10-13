package server

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"time"
	"unicode/utf8"

	"sparktalk/internal/db"
	"sparktalk/internal/llm"
	"sparktalk/internal/performance"
	"sparktalk/internal/workflows"
)

// The legacy worker remains available for saved, unversioned runs. This worker
// never equates successful generation/JSON coverage with a semantic review.
func (s *Server) runReviewedDocument(ctx context.Context, run workflows.Run) {
	started, elapsed := time.Now(), run.Document.ElapsedSeconds
	updateTime := func() { run.Document.ElapsedSeconds = elapsed + int(time.Since(started).Seconds()) }
	fail := func(status string, err error) {
		if errors.Is(err, db.ErrStaleDocumentExecution) {
			return
		}
		updateTime()
		run.Status = status
		run.Document.Terminal = false
		run.Document.FinishedAt = ""
		run.Document.Error = err.Error()
		run.Document.Performance = nil
		_ = s.db.SaveDocumentProgress(&run)
	}
	defer func() {
		if p := recover(); p != nil {
			fail("failed", fmt.Errorf("reviewed document worker: %v", p))
		}
	}()
	if run.Document.Kind != "translate" || run.Document.Profiles == nil {
		fail("failed", fmt.Errorf("invalid reviewed pipeline configuration"))
		return
	}
	units, err := s.db.DocumentUnits(run.SessionID, run.ID)
	if err != nil {
		fail("failed", err)
		return
	}
	entities, err := s.db.DocumentEntities(run.SessionID, run.ID)
	if err != nil {
		fail("failed", err)
		return
	}
	for i := range units {
		if units[i].Status == "running" {
			units[i].Status = "pending"
		}
		if units[i].Quality == nil {
			units[i].Quality = &workflows.DocumentUnitQuality{Status: "pending"}
		}
	}
	for _, id := range run.Document.Sources {
		book, e := s.db.Source(run.SessionID, id, false)
		if e != nil {
			fail("failed", e)
			return
		}
		decorateDocumentInputs(units, id, book)
	}
	run.Document.Error = ""
	run.Document.Terminal = false
	run.Document.FinishedAt = ""
	run.Steps[0].Status = "running"
	for {
		if ctx.Err() != nil {
			fail("paused", ctx.Err())
			return
		}
		updateTime()
		documentQualityCounters(&run, units)
		if run.Document.OutputOnly {
			if run.Document.Completed != len(units) {
				fail("failed", fmt.Errorf("output revision contains incomplete source spans"))
				return
			}
			break
		}
		index := -1
		stage := ""
		for _, i := range documentOrderedIndices(units) {
			u := units[i]
			if u.Status != "completed" {
				index, stage = i, "translate"
				break
			}
			if !documentReviewCurrent(u) {
				index, stage = i, "review"
				break
			}
			if u.Quality.Status == "needs_review" && u.Quality.Repairs < 2 && u.Quality.RepairFailures < 3 && len(documentReviewSubset(u, documentNeedsRepair).Inputs) > 0 {
				index, stage = i, "repair"
				break
			}
		}
		if index < 0 {
			break
		}
		if run.Document.ElapsedSeconds >= run.Document.MaxSeconds || run.Document.Calls >= run.Document.MaxCalls || run.Document.OutputTokens >= run.Document.MaxOutputTokens {
			fail("paused", fmt.Errorf("task execution budget reached"))
			return
		}
		unit := &units[index]
		if stage == "translate" && len(unit.Attempts)-unit.RetryBase >= 3 {
			fail("needs_review", documentFailure(fmt.Errorf("%s", unit.Error), unit.Ordinal, 3))
			return
		}
		if stage == "review" && unit.Quality.ReviewRetries >= 3 {
			// Leave this paragraph explicitly unchecked. It may be retried through a
			// child quality revision; ordinary resume must not reset quality budgets.
			unit.Quality.Status = "needs_review"
			unit.Quality.OutputHash = documentUnitOutputHash(*unit)
			if e := s.db.SaveDocumentProgress(&run, *unit); e != nil {
				fail("failed", e)
				return
			}
			continue
		}
		p := documentStageProfile(run, stage)
		requestRun := documentProfileRun(run, p)
		request := *unit
		indices := []int{index}
		if stage == "translate" {
			request, indices = documentRequestBatch(requestRun, units, index, p.WindowTokens)
		}
		if stage == "review" {
			request = documentPendingReview(*unit)
		}
		if stage == "repair" {
			request = documentReviewSubset(*unit, documentNeedsRepair)
			requestRun.Document.RepairContext = documentRepairEvidence(request)
		}
		requestRun.Document.SourceContext = documentSourceContext(units, request, 1280)
		requestRun.Document.ContextTail = documentReviewedContext(units, request)
		entityBudget := max(0, min(1024, 2048-estimateTextTokens(requestRun.Document.SourceContext)-estimateTextTokens(requestRun.Document.ContextTail)-64))
		relevant, versions := documentRelevantEntities(request, entities, entityBudget)
		requestRun.Document.EntityContext = documentEntityContext(relevant)
		run.Document.Phase = stage
		run.Document.ActiveUnits = nil
		for _, i := range indices {
			run.Document.ActiveUnits = append(run.Document.ActiveUnits, units[i].Ordinal+1)
		}
		if len(request.Inputs) > 0 {
			run.Document.CurrentChapter = request.Inputs[0].Chapter
		}
		run.Document.ReasoningBudgetTokens = p.ReasoningBudget
		expected := map[int]string{}
		for _, i := range indices {
			expected[units[i].Ordinal] = documentUnitOutputHash(units[i])
		}
		result, attempt, requestErr := s.requestReviewedStage(ctx, &run, requestRun, request, p, stage, started, elapsed)
		if attempt == nil {
			status := "paused"
			if errors.Is(requestErr, errDocumentModelChanged) {
				status = "waiting_model"
			}
			fail(status, requestErr)
			return
		}
		attempt.ExpectedUnitOutputHashes = expected
		if unit.Quality.TransportRetries == nil {
			unit.Quality.TransportRetries = map[string]int{}
		}
		if requestErr != nil && !errors.Is(requestErr, llm.ErrOutputLimit) && !isContextOverflow(requestErr) {
			attempt.FailureKind = "transport"
			unit.Quality.TransportRetries[stage]++
			if e := s.db.SaveDocumentReviewProgress(&run, attempt, nil, nil, *unit); e != nil {
				fail("failed", e)
				return
			}
			if ctx.Err() != nil {
				fail("paused", ctx.Err())
				return
			}
			if documentModelUnavailable(requestErr) {
				fail("waiting_model", requestErr)
				return
			}
			if unit.Quality.TransportRetries[stage] >= 3 {
				fail("paused", requestErr)
				return
			}
			if e := documentRetryPause(ctx, documentRetryDelay(requestErr, unit.Quality.TransportRetries[stage])); e != nil {
				fail("paused", e)
				return
			}
			continue
		}
		unit.Quality.TransportRetries[stage] = 0
		var review *workflows.DocumentReview
		var proposed []workflows.DocumentEntity
		var changed []workflows.DocumentUnit
		if stage == "review" {
			unit.Quality.ReviewRetries++
			unit.Quality.ReviewError = ""
			if requestErr == nil {
				parsed, e := validateDocumentReviewPartial(request, result)
				if e != nil {
					requestErr = e
					unit.Quality.ReviewError = e.Error()
				} else {
					parsed.ID = rand.Text()
					parsed.AttemptID = attempt.ID
					review = &parsed
					documentMergeReview(unit, parsed)
					unit.Quality.EntityVersions = maps.Clone(versions)
					if len(parsed.Invalid) == 0 {
						unit.Quality.ReviewRetries = 0
					} else {
						raw, _ := json.Marshal(parsed.Invalid)
						unit.Quality.ReviewError = string(raw)
					}
				}
			}
			if requestErr != nil {
				unit.Quality.ReviewError = requestErr.Error()
			}
			changed = append(changed, *unit)
		} else {
			var output map[string]string
			if requestErr == nil {
				output, requestErr = validateDocumentOutput(request, "translate", result)
				if result.FinishReason != "stop" {
					requestErr = fmt.Errorf("incomplete generation: %s", result.FinishReason)
					output = nil
				}
			}
			if errors.Is(requestErr, llm.ErrOutputLimit) && len(result.ToolCalls) == 0 {
				if recovered, e := documentCompletePrefix(request, result.Content); e == nil {
					output = recovered
				}
			}
			request.Output = output
			if requestErr != nil {
				attempt.Measurement.Error = requestErr.Error()
			}
			// A failed transport or ambiguous JSON never provides a usable paragraph.
			if len(output) > 0 {
				bad := incompleteDocumentGroups(request)
				if stage == "repair" {
					beforeRepair := documentUnitOutputHash(*unit)
					unit.Quality.Repairs++
					for _, g := range documentTextGroups(request) {
						valid := true
						for _, in := range g.Inputs {
							if _, ok := bad[in.ID]; ok {
								valid = false
							}
						}
						if valid {
							for _, in := range g.Inputs {
								unit.Output[in.ID] = output[in.ID]
							}
						}
					}
					if documentUnitOutputHash(*unit) != beforeRepair {
						// A changed target requires a new review; failed reviews of the
						// previous target cannot exhaust that new target's retry budget.
						unit.Quality.ReviewRetries = 0
						unit.Quality.ReviewError = ""
					}
					documentMergeReview(unit, workflows.DocumentReview{}) // hashes invalidate changed paragraphs only
					changed = append(changed, *unit)
				} else {
					for _, i := range indices {
						current := units[i]
						current.Attempts = append(current.Attempts, attempt.Measurement)
						current.Output = map[string]string{}
						current.Error = ""
						for _, in := range current.Inputs {
							if v, ok := output[in.ID]; ok {
								current.Output[in.ID] = v
							}
						}
						// Keep good paragraphs independently when one sibling failed.
						var goodInputs, badInputs []workflows.DocumentInput
						for _, g := range documentTextGroups(current) {
							valid := true
							for _, in := range g.Inputs {
								if _, ok := bad[in.ID]; ok {
									valid = false
								}
							}
							if valid {
								goodInputs = append(goodInputs, g.Inputs...)
							} else {
								badInputs = append(badInputs, g.Inputs...)
							}
						}
						if len(goodInputs) > 0 {
							current.Inputs = goodInputs
							current.Status = "completed"
							current.Quality = &workflows.DocumentUnitQuality{Status: "pending", EntityVersions: maps.Clone(versions)}
							if len(badInputs) > 0 {
								retry := workflows.DocumentUnit{Ordinal: len(units), Stage: "process", Status: "pending", Inputs: badInputs, Attempts: append([]workflows.DocumentAttempt(nil), current.Attempts...), Quality: &workflows.DocumentUnitQuality{Status: "pending"}, Error: "incomplete paragraph output"}
								units = append(units, retry)
								changed = append(changed, retry)
							}
						} else {
							current.Status = "pending"
							current.Error = "incomplete paragraph output"
						}
						// Discard outputs for spans moved to a retry checkpoint.
						kept := map[string]string{}
						for _, in := range current.Inputs {
							if v, ok := current.Output[in.ID]; ok {
								kept[in.ID] = v
							}
						}
						current.Output = kept
						units[i] = current
						changed = append(changed, current)
					}
				}
				proposed = documentEntityCandidates(request, result, entities)
			} else {
				if stage == "repair" {
					unit.Quality.RepairFailures++
				} else {
					unit.Attempts = append(unit.Attempts, attempt.Measurement)
					unit.Status = "pending"
				}
				if requestErr != nil {
					unit.Error = requestErr.Error()
				}
				changed = append(changed, *unit)
			}
		}
		if requestErr != nil {
			attempt.FailureKind = "format"
			attempt.Measurement.Error = requestErr.Error()
		}
		if stage == "translate" && len(indices) == 1 && len(units[index].Output) == 0 && requestErr != nil && (errors.Is(requestErr, llm.ErrOutputLimit) || isContextOverflow(requestErr)) {
			if other, ok := splitDocumentUnit(&units[index]); ok {
				units[index].RetryBase = len(units[index].Attempts)
				other.Ordinal = len(units)
				other.Attempts = append([]workflows.DocumentAttempt(nil), units[index].Attempts...)
				other.RetryBase = len(other.Attempts)
				units = append(units, other)
				changed = append(changed, units[index], other)
			}
		}
		if stage == "review" && review != nil {
			proposed = documentEntityReviewUpdates(entities, units)
		}
		for _, i := range documentInvalidateEntityDependents(units, proposed) {
			changed = append(changed, units[i])
		}
		updateTime()
		documentQualityCounters(&run, units)
		if e := s.db.SaveDocumentReviewProgress(&run, attempt, review, proposed, changed...); e != nil {
			fail("failed", e)
			return
		}
		// Refresh versions after a successful atomic checkpoint.
		if len(proposed) > 0 {
			entities, err = s.db.DocumentEntities(run.SessionID, run.ID)
			if err != nil {
				fail("failed", err)
				return
			}
		}
		if ctx.Err() != nil {
			fail("paused", ctx.Err())
			return
		}
		if requestErr != nil && documentModelUnavailable(requestErr) {
			fail("waiting_model", requestErr)
			return
		}
		if requestErr != nil {
			if e := documentRetryPause(ctx, documentRetryDelay(requestErr, 1)); e != nil {
				fail("paused", e)
				return
			}
		}
	}
	documentQualityCounters(&run, units)
	alreadyExported := run.Current >= 2 && run.Document.Phase == "convert"
	if alreadyExported {
		for _, name := range []string{"output.txt", "output.md", "output.html", "output.zip"} {
			raw, _, e := s.db.DocumentOutput(run.SessionID, run.ID, name)
			if e != nil || len(raw) == 0 {
				alreadyExported = false
				break
			}
		}
	}
	run.Current = 1
	run.Document.Phase = "export"
	run.Steps[0].Status = "completed"
	run.Steps[1].Status = "running"
	if err = s.db.SaveDocumentProgress(&run); err != nil {
		fail("failed", err)
		return
	}
	if !alreadyExported {
		err = s.exportDocument(ctx, run, units)
	}
	if err != nil {
		fail("failed", err)
		return
	}
	run.Current = 2
	if run.Document.RequestedFormat != "" {
		run.Document.Phase = "convert"
		if err = s.db.SaveDocumentProgress(&run); err != nil {
			fail("failed", err)
			return
		}
		if err = s.convertDocumentTask(ctx, run, run.Document.RequestedFormat); err != nil {
			if ctx.Err() == nil {
				if raw, _, e := s.db.DocumentOutput(run.SessionID, run.ID, "output.txt"); e == nil {
					if e = s.db.PublishDocumentDraft(&run, []string{"output.txt", "output.md", "output.html", "output.zip"}, documentTextHash(string(raw))); e != nil {
						fail("failed", e)
						return
					}
				}
			}
			fail("conversion_failed", err)
			return
		}
		latest, e := s.db.DocumentRun(run.SessionID, run.ID)
		if e != nil {
			fail("failed", e)
			return
		}
		run.Document.Formats = latest.Document.Formats
		run.Document.Warnings = latest.Document.Warnings
	}
	names := []string{"output.txt", "output.md", "output.html", "output.zip"}
	for _, format := range run.Document.Formats {
		if format != "txt" && format != "md" && format != "html" && format != "zip" {
			names = append(names, "output."+format)
		}
	}
	raw, _, err := s.db.DocumentOutput(run.SessionID, run.ID, "output.txt")
	if err != nil {
		fail("failed", err)
		return
	}
	updateTime()
	run.Status = "completed"
	run.Document.Phase = "completed"
	run.Document.Terminal = true
	run.Document.FinishedAt = time.Now().UTC().Format(time.RFC3339Nano)
	run.Document.Error = ""
	run.Document.ActiveUnits = nil
	run.Steps[1].Status = "completed"
	if run.Document.QualityStatus != "reviewed" || len(run.Document.Warnings) > 0 {
		run.Status = "needs_review"
	}
	if err = s.db.PublishDocumentRevision(&run, names, documentTextHash(string(raw))); err != nil {
		fail("failed", err)
	}
}

var errDocumentModelChanged = errors.New("task model differs from active model")

func (s *Server) requestReviewedStage(ctx context.Context, run *workflows.Run, requestRun workflows.Run, u workflows.DocumentUnit, p workflows.DocumentProfile, stage string, started time.Time, baseElapsed int) (llm.StreamResult, *workflows.DocumentRequestAttempt, error) {
	if err := s.documentSlot(ctx); err != nil {
		return llm.StreamResult{}, nil, err
	}
	defer s.docInferMu.Unlock()
	cfg, client := s.snapshot()
	if cfg.Model.DefaultModel != p.Model || cfg.Model.ModelType != p.ModelType || cfg.Model.Endpoint != p.Endpoint {
		return llm.StreamResult{}, nil, errDocumentModelChanged
	}
	p.WindowTokens = minPositive(p.WindowTokens, cfg.Context.WindowTokens)
	p.WindowSource = "configured"
	if actual, e := client.ContextWindow(ctx); e == nil && actual > 0 {
		p.WindowTokens = minPositive(p.WindowTokens, actual)
		p.WindowSource = "server_metadata"
	}
	run.Document.Performance = nil
	run.Document.Activity = "waiting"
	run.Document.ReasoningChars = 0
	run.Document.ContentChars = 0
	if err := s.db.SaveDocumentProgress(run); err != nil {
		return llm.StreamResult{}, nil, err
	}
	inferCtx, release, err := s.trackGeneration(ctx)
	if err != nil {
		return llm.StreamResult{}, nil, err
	}
	defer release()
	remaining := max(1, run.Document.MaxSeconds-baseElapsed-int(time.Since(started).Seconds()))
	inferCtx, cancel := context.WithTimeout(inferCtx, time.Duration(remaining)*time.Second)
	defer cancel()
	client = client.WithBackendCancellation(inferCtx)
	var metrics performance.Accumulator
	inferCtx = llm.WithPerformanceObserver(llm.WithTemperature(inferCtx, p.Temperature), metrics.Start)
	requestHash := ""
	inferCtx = llm.WithRequestObserver(inferCtx, func(raw []byte) { requestHash = documentTextHash(string(raw)) })
	requestStart, lastProgress := time.Now(), time.Now()
	emit := func(kind, text string) error {
		if inferCtx.Err() != nil {
			return inferCtx.Err()
		}
		if kind == "reasoning" {
			run.Document.Activity = "reasoning"
			run.Document.ReasoningChars += utf8.RuneCountInString(text)
		}
		if kind == "delta" {
			run.Document.Activity = "answer"
			run.Document.ContentChars += utf8.RuneCountInString(text)
		}
		if time.Since(lastProgress) >= 2*time.Second {
			run.Document.Performance = metrics.Summary(true)
			run.Document.ElapsedSeconds = baseElapsed + int(time.Since(started).Seconds())
			lastProgress = time.Now()
			return s.db.SaveDocumentProgress(run)
		}
		return nil
	}
	var result llm.StreamResult
	if stage == "review" {
		result, err = requestDocumentReview(inferCtx, client, requestRun, u, minPositive(p.WindowTokens, cfg.Context.WindowTokens), emit)
	} else {
		result, err = requestDocumentUnit(inferCtx, client, requestRun, u, minPositive(p.WindowTokens, cfg.Context.WindowTokens), emit)
	}
	outputTokens := max(0, result.Usage.CompletionTokens)
	estimated := outputTokens == 0 && (result.Content != "" || result.Reasoning != "")
	if estimated {
		outputTokens = estimateTextTokens(result.Content) + estimateTextTokens(result.Reasoning)
	}
	protocol := documentOutputProtocol
	if stage == "review" {
		protocol = documentReviewProtocol
		if requestRun.Document.Pipeline != "" {
			protocol = documentReviewProtocolV2
		}
	}
	measurement := workflows.DocumentAttempt{ReasoningBudgetTokens: p.ReasoningBudget, OutputTokensEstimated: estimated, Protocol: protocol, ReasoningChars: utf8.RuneCountInString(result.Reasoning), ContentChars: utf8.RuneCountInString(result.Content), Revision: run.Document.Revision, Model: p.Model, FinishReason: result.FinishReason, PromptTokens: result.Usage.PromptTokens, OutputTokens: outputTokens, Seconds: time.Since(requestStart).Seconds(), Performance: metrics.Summary(false)}
	if err != nil {
		measurement.Error = err.Error()
	}
	input, _ := json.Marshal(struct {
		Unit workflows.DocumentUnit
		Task *workflows.DocumentTask
	}{u, requestRun.Document})
	if requestHash == "" {
		requestHash = documentTextHash(string(input))
	}
	attempt := &workflows.DocumentRequestAttempt{ID: rand.Text(), Stage: stage, StartedAt: requestStart.UTC().Format(time.RFC3339Nano), FinishedAt: time.Now().UTC().Format(time.RFC3339Nano), Profile: p, Units: append([]int(nil), run.Document.ActiveUnits...), InputHash: requestHash, OutputHash: documentTextHash(result.Content), RawContent: result.Content, Measurement: measurement}
	wireIDs := map[string]string{}
	if stage == "review" {
		for _, pair := range documentReviewPairs(u) {
			for _, id := range pair.InputIDs {
				wireIDs[id] = pair.ID
			}
		}
	} else {
		for _, entry := range documentWireEntries(u) {
			for _, index := range entry.Indexes {
				wireIDs[u.Inputs[index].ID] = entry.ID
			}
		}
	}
	for _, in := range u.Inputs {
		attempt.SourceSpans = append(attempt.SourceSpans, workflows.DocumentRequestSpan{ID: in.ID, WireID: wireIDs[in.ID], Source: in.Source, Block: in.Block, Start: in.Part, Length: utf8.RuneCountInString(in.Text), Annotation: in.Annotation})
	}
	run.Document.Performance = nil
	run.Document.LastPerformance = measurement.Performance
	return result, attempt, err
}

func minPositive(a, b int) int {
	if a <= 0 {
		return b
	}
	if b <= 0 {
		return a
	}
	return min(a, b)
}
