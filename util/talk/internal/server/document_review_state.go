package server

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"unicode/utf8"

	"sparktalk/internal/config"
	"sparktalk/internal/llm"
	"sparktalk/internal/workflows"
)

func documentPrepareProfiles(cfg config.Config, profiles workflows.DocumentProfiles) (*workflows.DocumentProfiles, error) {
	for _, p := range []*workflows.DocumentProfile{&profiles.Translate, &profiles.Review, &profiles.Repair} {
		if p.Model != cfg.Model.DefaultModel || p.ModelType != cfg.Model.ModelType || p.Endpoint != cfg.Model.Endpoint {
			return nil, fmt.Errorf("document profile differs from active model")
		}
		if math.IsNaN(p.Temperature) || math.IsInf(p.Temperature, 0) || p.Temperature < 0 || p.Temperature > 2 || p.ReasoningBudget < -1 || p.ReasoningBudget > 16384 {
			return nil, fmt.Errorf("invalid document sampling profile")
		}
		if p.Sampling == nil {
			p.Sampling = documentModelSampling(p.ModelType)
		}
		if p.Sampling != nil {
			copy := *p.Sampling
			if err := copy.Validate(); err != nil {
				return nil, err
			}
			p.Sampling = &copy
		}
		if p.WindowTokens <= 0 {
			p.WindowTokens = cfg.Context.WindowTokens
		}
		if cfg.Context.WindowTokens > 0 && p.WindowTokens > cfg.Context.WindowTokens {
			return nil, fmt.Errorf("profile exceeds active context window")
		}
	}
	return &profiles, nil
}

func documentProfiles(cfg config.Config, effort string) *workflows.DocumentProfiles {
	p := workflows.DocumentProfile{Model: cfg.Model.DefaultModel, ModelType: cfg.Model.ModelType, Endpoint: cfg.Model.Endpoint, Effort: effort, Temperature: 0.2, Sampling: documentModelSampling(cfg.Model.ModelType), WindowTokens: cfg.Context.WindowTokens}
	p.ReasoningBudget = documentReasoningBudget(workflows.Run{Document: &workflows.DocumentTask{Model: p.Model, ModelType: p.ModelType, Effort: p.Effort}})
	review := p
	review.Effort = "medium"
	review.ReasoningBudget = documentReasoningBudget(workflows.Run{Document: &workflows.DocumentTask{Model: p.Model, ModelType: p.ModelType, Effort: review.Effort}})
	return &workflows.DocumentProfiles{Translate: p, Review: review, Repair: review}
}

func documentStageProfile(run workflows.Run, stage string) workflows.DocumentProfile {
	p := run.Document.Profiles.Translate
	if stage == "review" {
		p = run.Document.Profiles.Review
	}
	if stage == "repair" {
		p = run.Document.Profiles.Repair
	}
	return p
}

func documentProfileRun(run workflows.Run, p workflows.DocumentProfile) workflows.Run {
	d := *run.Document
	run.Document = &d
	d.Model, d.ModelType, d.Endpoint, d.Effort = p.Model, p.ModelType, p.Endpoint, p.Effort
	d.ReasoningBudgetTokens = p.ReasoningBudget
	d.Sampling = p.Sampling
	return run
}

func documentUnitOutputHash(u workflows.DocumentUnit) string {
	raw, _ := json.Marshal(u.Output)
	return documentTextHash(string(raw))
}

func documentReviewCurrent(u workflows.DocumentUnit) bool {
	return u.Status == "completed" && u.Quality != nil && u.Quality.OutputHash == documentUnitOutputHash(u) && (u.Quality.Status == "reviewed" || u.Quality.Status == "needs_review")
}

// Only a review of the exact current target can supply preceding target context.
func documentReviewedContext(units []workflows.DocumentUnit, next workflows.DocumentUnit) string {
	if len(next.Inputs) == 0 {
		return ""
	}
	boundary := next.Inputs[0]
	for _, in := range next.Inputs {
		if in.Source == boundary.Source && (in.Block < boundary.Block || in.Block == boundary.Block && in.Part < boundary.Part) {
			boundary = in
		}
	}
	type contextParagraph struct {
		SourceIDs []string `json:"source_ids"`
		Target    string   `json:"target"`
		last      workflows.DocumentInput
	}
	var candidates []contextParagraph
	for _, u := range units {
		if u.Status != "completed" || u.Quality == nil {
			continue
		}
		inputs := map[string]workflows.DocumentInput{}
		for _, in := range u.Inputs {
			inputs[in.ID] = in
		}
		reviews := map[string]workflows.DocumentReviewItem{}
		for _, item := range u.Quality.Reviews {
			if len(item.InputIDs) > 0 {
				reviews[item.InputIDs[0]] = item
			}
		}
		for _, pair := range documentReviewPairs(u) {
			item, ok := reviews[pair.InputIDs[0]]
			if !ok || item.SourceHash != documentTextHash(pair.Source) || item.OutputHash != documentTextHash(pair.Target) {
				continue
			}
			accepted := true
			for _, finding := range item.Findings {
				accepted = accepted && finding.Severity == "minor" && !finding.Uncertain
			}
			last := inputs[pair.InputIDs[0]]
			for _, id := range pair.InputIDs {
				in, exists := inputs[id]
				// A checkpoint can straddle a repaired gap. Its first position is
				// not evidence that every paragraph precedes the next request.
				if !exists || in.Source != boundary.Source || in.Block > boundary.Block || in.Block == boundary.Block && in.Part+utf8.RuneCountInString(in.Text) > boundary.Part {
					accepted = false
				}
				if in.Block > last.Block || in.Block == last.Block && in.Part > last.Part {
					last = in
				}
			}
			if accepted {
				candidates = append(candidates, contextParagraph{SourceIDs: pair.InputIDs, Target: pair.Target, last: last})
			}
		}
	}
	sort.SliceStable(candidates, func(i, j int) bool {
		if candidates[i].last.Block != candidates[j].last.Block {
			return candidates[i].last.Block < candidates[j].last.Block
		}
		return candidates[i].last.Part < candidates[j].last.Part
	})
	selected := []contextParagraph{}
	for i := len(candidates) - 1; i >= 0 && len(selected) < 2; i-- {
		candidate := append([]contextParagraph{candidates[i]}, selected...)
		raw, _ := json.Marshal(candidate)
		if estimateTextTokens(string(raw)) > 512 {
			break
		}
		selected = candidate
	}
	if len(selected) == 0 {
		return ""
	}
	raw, _ := json.Marshal(selected)
	return string(raw)
}

func documentQualityCounters(run *workflows.Run, units []workflows.DocumentUnit) {
	d := run.Document
	d.Completed, d.Reviewed, d.ReviewTotal, d.Unresolved = 0, 0, 0, 0
	d.Total = len(units)
	allCurrent := true
	for _, u := range units {
		pairs := documentReviewPairs(u)
		d.ReviewTotal += len(pairs)
		if u.Status == "completed" {
			d.Completed++
		}
		if !documentReviewCurrent(u) {
			allCurrent = false
		}
		if u.Status != "completed" || u.Quality == nil {
			continue
		}
		// A repaired paragraph invalidates its own review, not the counts of
		// already verified siblings while their shared checkpoint is pending.
		bySource := map[string]workflows.DocumentReviewItem{}
		for _, item := range u.Quality.Reviews {
			if len(item.InputIDs) > 0 {
				bySource[item.InputIDs[0]] = item
			}
		}
		valid := 0
		for _, pair := range pairs {
			item, ok := bySource[pair.InputIDs[0]]
			if !ok || item.SourceHash != documentTextHash(pair.Source) || item.OutputHash != documentTextHash(pair.Target) {
				continue
			}
			valid++
			for _, finding := range item.Findings {
				if finding.Severity != "minor" || finding.Uncertain {
					d.Unresolved++
				}
			}
		}
		d.Reviewed += valid
		if u.Quality.Status == "needs_review" && valid < len(pairs) {
			d.Unresolved += len(pairs) - valid
		}

	}
	d.QualityStatus = "pending"
	if allCurrent && d.Completed == len(units) && d.Reviewed == d.ReviewTotal {
		d.QualityStatus = "reviewed"
		if d.Unresolved > 0 {
			d.QualityStatus = "needs_review"
		}
	}
}

// Review local IDs are transient. Persist and select by original source IDs.
func documentReviewSubset(u workflows.DocumentUnit, predicate func(workflows.DocumentReviewItem) bool) workflows.DocumentUnit {
	out := u
	out.Inputs = nil
	out.Output = map[string]string{}
	want := map[string]bool{}
	if u.Quality != nil {
		for _, item := range u.Quality.Reviews {
			if predicate(item) {
				for _, id := range item.InputIDs {
					want[id] = true
				}
			}
		}
	}
	for _, in := range u.Inputs {
		if want[in.ID] {
			out.Inputs = append(out.Inputs, in)
			out.Output[in.ID] = u.Output[in.ID]
		}
	}
	return out
}

func documentNeedsRepair(item workflows.DocumentReviewItem) bool {
	for _, f := range item.Findings {
		if !f.Uncertain && (f.Severity == "critical" || f.Severity == "major") {
			return true
		}
	}
	return false
}

func documentRepairEvidence(u workflows.DocumentUnit) string {
	type repairParagraph struct {
		Paragraph string                      `json:"paragraph"`
		Source    string                      `json:"source"`
		Target    string                      `json:"target"`
		Findings  []workflows.DocumentFinding `json:"findings"`
	}
	data := []repairParagraph{}
	paragraphs := map[string]string{}
	for _, entry := range documentWireEntries(u) {
		for _, index := range entry.Indexes {
			paragraphs[u.Inputs[index].ID] = entry.Paragraph
		}
	}
	for _, pair := range documentReviewPairs(u) {
		item := repairParagraph{Source: pair.Source, Target: pair.Target}
		for _, id := range pair.InputIDs {
			if label := paragraphs[id]; label != "" {
				item.Paragraph = label
				break
			}
		}
		if u.Quality != nil {
			for _, review := range u.Quality.Reviews {
				if len(review.InputIDs) > 0 && review.InputIDs[0] == pair.InputIDs[0] {
					item.Findings = review.Findings
					break
				}
			}
		}
		data = append(data, item)
	}
	raw, _ := json.Marshal(data)
	return string(raw)
}

// A partial review replaces only matching paragraphs; siblings remain intact.
func documentMergeReview(u *workflows.DocumentUnit, review workflows.DocumentReview) {
	if u.Quality == nil {
		u.Quality = &workflows.DocumentUnitQuality{}
	}
	bySource := map[string]workflows.DocumentReviewItem{}
	for _, item := range u.Quality.Reviews {
		if len(item.InputIDs) > 0 {
			bySource[item.InputIDs[0]] = item
		}
	}
	for _, item := range review.Items {
		if len(item.InputIDs) > 0 {
			bySource[item.InputIDs[0]] = item
		}
	}
	u.Quality.Reviews = nil
	u.Quality.Status = "reviewed"
	for _, pair := range documentReviewPairs(*u) {
		item, ok := bySource[pair.InputIDs[0]]
		if !ok || item.OutputHash != documentTextHash(pair.Target) || item.SourceHash != documentTextHash(pair.Source) {
			u.Quality.Status = "pending"
			continue
		}
		item.ID = pair.ID
		u.Quality.Reviews = append(u.Quality.Reviews, item)
	}
	if len(u.Quality.Reviews) == len(documentReviewPairs(*u)) {
		for _, item := range u.Quality.Reviews {
			for _, f := range item.Findings {
				if f.Severity != "minor" || f.Uncertain {
					u.Quality.Status = "needs_review"
				}
			}
		}
	}
	u.Quality.OutputHash = documentUnitOutputHash(*u)
	if review.ID != "" {
		u.Quality.ReviewIDs = append(u.Quality.ReviewIDs, review.ID)
	}
}

func documentPendingReview(u workflows.DocumentUnit) workflows.DocumentUnit {
	valid := map[string]bool{}
	if u.Quality != nil {
		for _, pair := range documentReviewPairs(u) {
			for _, item := range u.Quality.Reviews {
				if len(item.InputIDs) > 0 && item.InputIDs[0] == pair.InputIDs[0] && item.SourceHash == documentTextHash(pair.Source) && item.OutputHash == documentTextHash(pair.Target) {
					for _, id := range item.InputIDs {
						valid[id] = true
					}
				}
			}
		}
	}
	out := u
	out.Inputs = nil
	for _, in := range u.Inputs {
		if !valid[in.ID] {
			out.Inputs = append(out.Inputs, in)
		}
	}
	return out
}

// Stable source order also works for appended split checkpoints.
func documentOrderedIndices(units []workflows.DocumentUnit) []int {
	indices := make([]int, len(units))
	for i := range indices {
		indices[i] = i
	}
	sort.SliceStable(indices, func(i, j int) bool { return documentUnitBefore(units[indices[i]], units[indices[j]]) })
	return indices
}

func documentModelSampling(modelType string) *llm.SamplingParameters {
	if modelType != "qwen38fn_exl3" && modelType != "qwen3.8" && modelType != "qwen3.8-vllm" && modelType != "qwen3.8-tensorfold" {
		return nil
	}
	return &llm.SamplingParameters{TopP: 0.95, TopK: 20, MinP: 0, RepetitionPenalty: 1, PresencePenalty: 0, FrequencyPenalty: 0}
}
func documentSampling(run workflows.Run) *llm.SamplingParameters {
	if run.Document.Pipeline == "" {
		return nil
	}
	if run.Document.Sampling != nil {
		return run.Document.Sampling
	}
	return documentModelSampling(run.Document.ModelType)
}
