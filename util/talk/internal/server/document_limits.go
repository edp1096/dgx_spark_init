package server

import (
	"context"
	"errors"
	"fmt"
	"net"
	"time"
	"unicode"
	"unicode/utf8"

	"sparktalk/internal/llm"
	"sparktalk/internal/workflows"
)

var errDocumentIdle = errors.New("문서 구간의 모델 응답이 일정 시간 동안 진행되지 않았습니다")
var errDocumentReasoning = errors.New("문서 구간의 추론이 예산을 초과했지만 본문 출력으로 전환되지 않았습니다")

type documentLimitsKey struct{}
type documentStreamLimits struct {
	idle           time.Duration
	reasoningChars int
}

func documentReasoningBudget(run workflows.Run) int {
	// Explicit unbounded comparison profile; never selected by ordinary tasks.
	if run.Document.ReasoningBudgetTokens < 0 {
		return 0
	}
	if !llm.SupportsStructuredReasoningBudget(run.Document.Model, run.Document.ModelType) {
		return 0
	}
	if run.Document.Profiles != nil {
		return max(0, min(run.Document.ReasoningBudgetTokens, 16384))
	}
	if run.Document.Effort != "none" && run.Document.Effort != "off" && run.Document.Effort != "" && run.Document.ReasoningBudgetTokens > 0 {
		return min(run.Document.ReasoningBudgetTokens, 16384)
	}
	switch run.Document.Effort {
	case "", "none", "off":
		return 0
	case "low":
		return 2048
	case "medium":
		return 4096
	default:
		return 8192
	}
}

// Output reservation is not a claim that the backend can enforce a separate
// thinking cap. Engines without that API still consume completion tokens while
// thinking; reserve space for both thinking and the final JSON.
func documentReviewReasoningReserve(run workflows.Run) int {
	if native := documentReasoningBudget(run); native > 0 {
		return native
	}
	switch run.Document.Effort {
	case "", "none", "off":
		return 0
	case "low":
		// A soft low-effort instruction does not enforce a 4K thinking cap.
		// Leave room for the review JSON after an uncapped reasoning phase.
		return 8192
	case "medium":
		return 8192
	default:
		return 16384
	}
}

// An active stream is governed by its output budget and the remaining task
// budget, not an arbitrary ten-minute cutoff. Lack of progress has its own limit.
// The character backstop catches servers which ignore a native reasoning budget;
// it is deliberately conservative and is not represented as an exact token count.
func streamDocumentUnit(ctx context.Context, client *llm.Client, run workflows.Run, messages []llm.Message, limit int, progress func(string, string) error) (llm.StreamResult, error) {
	if sampling := documentSampling(run); sampling != nil {
		ctx = llm.WithSamplingDefaults(ctx, *sampling)
	}
	budget := documentReasoningBudget(run)
	limits := documentStreamLimits{idle: 2 * time.Minute}
	if budget > 0 {
		limits.reasoningChars = max(8192, budget*8)
	}
	if override, ok := ctx.Value(documentLimitsKey{}).(documentStreamLimits); ok {
		limits = override
	}
	streamCtx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	timer := time.AfterFunc(limits.idle, func() { cancel(errDocumentIdle) })
	defer timer.Stop()
	chars := 0
	var whitespace documentJSONWhitespace
	emit := func(kind, text string) error {
		if err := streamCtx.Err(); err != nil {
			return err
		}
		if text != "" {
			timer.Reset(limits.idle)
		}
		if kind == "reasoning" {
			chars += utf8.RuneCountInString(text)
			if limits.reasoningChars > 0 && chars > limits.reasoningChars {
				return errDocumentReasoning
			}
		}
		if kind == "delta" && whitespace.push(text) {
			return fmt.Errorf("%w: structured response stalled on more than 2048 whitespace characters outside JSON strings", llm.ErrOutputLimit)
		}
		return progress(kind, text)
	}
	streamCtx = llm.WithReasoningBudget(streamCtx, budget)
	result, err := client.Stream(llm.WithOutputLimit(streamCtx, limit), messages, run.Document.Model, run.Document.Effort, nil, emit)
	if cause := context.Cause(streamCtx); cause != nil && ctx.Err() == nil {
		return result, cause
	}
	return result, err
}

// JSON grammar permits unlimited whitespace between values. Stop a model stuck
// in that state while preserving arbitrary authored whitespace inside strings.
type documentJSONWhitespace struct {
	quoted, escaped bool
	run             int
}

func (s *documentJSONWhitespace) push(text string) bool {
	for _, r := range text {
		if s.quoted {
			if s.escaped {
				s.escaped = false
			} else if r == '\\' {
				s.escaped = true
			} else if r == '"' {
				s.quoted = false
			}
			continue
		}
		if r == '"' {
			s.quoted = true
			s.run = 0
		} else if unicode.IsSpace(r) {
			s.run++
			if s.run > 2048 {
				return true
			}
		} else {
			s.run = 0
		}
	}
	return false
}

func documentModelUnavailable(err error) bool {
	var httpErr *llm.HTTPError
	if errors.As(err, &httpErr) {
		return httpErr.StatusCode == 502 || httpErr.StatusCode == 503
	}
	var op *net.OpError
	return errors.As(err, &op) && op.Op == "dial"
}

func documentRetryDelay(err error, attempts int) time.Duration {
	var netErr net.Error
	if errors.Is(err, llm.ErrStreamInterrupted) || errors.Is(err, errDocumentIdle) || errors.Is(err, context.DeadlineExceeded) || documentModelUnavailable(err) || errors.As(err, &netErr) {
		return time.Duration(min(attempts, 3)) * time.Second
	}
	return 0
}

func documentRetryPause(ctx context.Context, delay time.Duration) error {
	if delay <= 0 {
		return ctx.Err()
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func documentFailure(err error, ordinal, attempts int) error {
	return fmt.Errorf("구간 %d 처리에 %d회 실패했습니다. 완료된 구간은 보존되어 있으며 재개할 수 있습니다: %w", ordinal+1, attempts, err)
}
