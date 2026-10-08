package server

import (
	"context"
	"encoding/json"
	"mediaapp/internal/jobs"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"
)

// runtimePhase is the small engine-neutral telemetry contract persisted with a
// job. Engines remain responsible for reporting facts about their own memory
// lifecycle; Spark Media only observes and presents those facts.
type runtimePhase struct {
	OperationID   string  `json:"operation_id"`
	Phase         string  `json:"phase"`
	Component     string  `json:"component,omitempty"`
	Detail        string  `json:"detail,omitempty"`
	Progress      float64 `json:"progress,omitempty"`
	MemoryAction  string  `json:"memory_action,omitempty"`
	ResidentAfter *bool   `json:"resident_after,omitempty"`
	StartedAt     string  `json:"started_at,omitempty"`
	UpdatedAt     string  `json:"updated_at,omitempty"`
}

var validRuntimePhases = map[string]bool{
	"preparing": true, "model_loading": true, "conditioning": true,
	"sampling": true, "decoding": true, "model_unloading": true,
	"cache_retaining": true, "finalizing": true, "completed": true,
}

func (phase runtimePhase) validFor(operationID string) bool {
	return phase.OperationID == operationID && validRuntimePhases[phase.Phase]
}

type runtimeObserver struct {
	cancel context.CancelFunc
	done   chan struct{}
	once   sync.Once
}

func (observer *runtimeObserver) Stop() {
	if observer == nil {
		return
	}
	observer.once.Do(func() {
		observer.cancel()
		<-observer.done
	})
}

func (s *Server) startRuntimeObserver(parent context.Context, jobID, endpoint string) *runtimeObserver {
	endpoint = strings.TrimRight(endpoint, "/")
	if endpoint == "" || jobID == "" {
		return nil
	}
	s.runtimeCapabilityMu.RLock()
	supported := s.runtimeCapabilities[endpoint]
	s.runtimeCapabilityMu.RUnlock()
	if !supported {
		return nil
	}
	ctx, cancel := context.WithCancel(parent)
	observer := &runtimeObserver{cancel: cancel, done: make(chan struct{})}
	go func() {
		defer close(observer.done)
		ticker := time.NewTicker(350 * time.Millisecond)
		defer ticker.Stop()
		// The first request may precede engine state publication. Subsequent
		// polls are correlated by operation_id, so stale global state is ignored.
		for {
			s.observeRuntimePhase(ctx, jobID, endpoint)
			select {
			case <-ctx.Done():
				// One final read captures short cleanup/completed phases that were
				// published immediately before the synchronous response returned.
				finalCtx, finalCancel := context.WithTimeout(context.Background(), 750*time.Millisecond)
				s.observeRuntimePhase(finalCtx, jobID, endpoint)
				finalCancel()
				return
			case <-ticker.C:
			}
		}
	}()
	return observer
}

func (s *Server) observeRuntimePhase(ctx context.Context, jobID, endpoint string) {
	statusURL := endpoint + "/v1/models/runtime/status?operation_id=" + url.QueryEscape(jobID)
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, statusURL, nil)
	if err != nil {
		return
	}
	response, err := s.client.Do(request)
	if err != nil {
		return
	}
	defer response.Body.Close()
	if response.StatusCode/100 != 2 {
		return
	}
	var envelope struct {
		Operation        *runtimePhase  `json:"operation"`
		OperationHistory []runtimePhase `json:"operation_history"`
	}
	if err := json.NewDecoder(response.Body).Decode(&envelope); err != nil {
		return
	}
	for _, phase := range envelope.OperationHistory {
		if phase.validFor(jobID) {
			s.persistRuntimePhase(jobID, phase)
		}
	}
	if envelope.Operation != nil && envelope.Operation.validFor(jobID) {
		s.persistRuntimePhase(jobID, *envelope.Operation)
	}
}

func (s *Server) persistRuntimePhase(jobID string, phase runtimePhase) {
	s.generationStateMu.Lock()
	defer s.generationStateMu.Unlock()
	job, ok := s.jobs.Get(jobID)
	if !ok || job.Status != "running" {
		return
	}
	ensureJobParams(&job)
	history, _ := job.Params["runtime_phase_history"].([]any)
	for _, item := range history {
		entry, _ := item.(map[string]any)
		if entry != nil && stringParam(entry, "operation_id", "") == phase.OperationID &&
			stringParam(entry, "phase", "") == phase.Phase &&
			stringParam(entry, "component", "") == phase.Component &&
			stringParam(entry, "updated_at", "") == phase.UpdatedAt {
			return
		}
	}
	current, _ := job.Params["runtime_phase"].(map[string]any)
	if current != nil && stringParam(current, "phase", "") == phase.Phase &&
		stringParam(current, "component", "") == phase.Component &&
		stringParam(current, "detail", "") == phase.Detail &&
		floatParam(current, "progress", -1) == phase.Progress {
		return
	}
	data, _ := json.Marshal(phase)
	value := map[string]any{}
	_ = json.Unmarshal(data, &value)
	job.Params["runtime_phase"] = value
	job.Params["runtime_observed_at"] = time.Now().Format(time.RFC3339Nano)
	history = append(history, value)
	if len(history) > 24 {
		history = history[len(history)-24:]
	}
	job.Params["runtime_phase_history"] = history
	_ = s.jobs.Save(job)
}

func mergeObservedRuntime(target *map[string]any, source map[string]any) {
	if *target == nil {
		*target = map[string]any{}
	}
	for _, key := range []string{"runtime_phase", "runtime_phase_history", "runtime_observed_at"} {
		if value, ok := source[key]; ok {
			(*target)[key] = value
		}
	}
}

func runtimeBoolPointer(value bool) *bool { return &value }

func (s *Server) publishLocalRuntimePhase(jobID, phase, component, detail string, progress float64, action string, residentAfter *bool) {
	now := time.Now().Format(time.RFC3339Nano)
	s.persistRuntimePhase(jobID, runtimePhase{
		OperationID: jobID, Phase: phase, Component: component, Detail: detail,
		Progress: progress, MemoryAction: action, ResidentAfter: residentAfter,
		StartedAt: now, UpdatedAt: now,
	})
}

func (s *Server) saveJobPreservingRuntime(job jobs.Job) error {
	s.generationStateMu.Lock()
	defer s.generationStateMu.Unlock()
	if current, ok := s.jobs.Get(job.ID); ok {
		mergeObservedRuntime(&job.Params, current.Params)
	}
	return s.jobs.Save(job)
}
