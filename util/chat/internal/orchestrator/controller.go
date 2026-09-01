package orchestrator

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

type SystemMemory struct {
	TotalGiB     float64 `json:"total_gib"`
	UsedGiB      float64 `json:"used_gib"`
	AvailableGiB float64 `json:"available_gib"`
	FreeGiB      float64 `json:"free_gib"`
}

type ComponentStatus struct {
	Component
	Status       string  `json:"status"`
	Health       string  `json:"health"`
	GPUMemoryGiB float64 `json:"gpu_memory_gib"`
	Progress     float64 `json:"progress,omitempty"`
	Phase        string  `json:"phase,omitempty"`
	ETA          string  `json:"eta,omitempty"`
	Error        string  `json:"error,omitempty"`
}

type OperationStep struct {
	Key         string    `json:"-"`
	ComponentID string    `json:"component_id,omitempty"`
	Phase       string    `json:"phase"`
	Detail      string    `json:"detail,omitempty"`
	State       string    `json:"state"`
	StartedAt   time.Time `json:"started_at"`
	FinishedAt  time.Time `json:"finished_at,omitempty"`
}

type Operation struct {
	Action      string          `json:"action,omitempty"`
	BundleID    string          `json:"bundle_id,omitempty"`
	ComponentID string          `json:"component_id,omitempty"`
	State       string          `json:"state,omitempty"`
	Phase       string          `json:"phase,omitempty"`
	Detail      string          `json:"detail,omitempty"`
	Progress    float64         `json:"progress,omitempty"`
	ETA         string          `json:"eta,omitempty"`
	Error       string          `json:"error,omitempty"`
	StartedAt   time.Time       `json:"started_at,omitempty"`
	FinishedAt  time.Time       `json:"finished_at,omitempty"`
	Steps       []OperationStep `json:"steps,omitempty"`
}

type Snapshot struct {
	SelectedBundle string            `json:"selected_bundle"`
	Bundles        []Bundle          `json:"bundles"`
	Components     []ComponentStatus `json:"components"`
	Memory         SystemMemory      `json:"memory"`
	Operation      Operation         `json:"operation"`
	Docker         string            `json:"docker"`
}

type Controller struct {
	catalog    Catalog
	client     *http.Client
	mu         sync.RWMutex
	op         Operation
	dataDir    string
	modelCache string
}

const minimumCUDAImmediateFreeGiB = 4.0

type memoryPlan struct {
	NeededGiB         float64
	FreedGiB          float64
	RequiresCUDAStart bool
}

func NewController() (*Controller, error) {
	catalog, err := LoadCatalog()
	if err != nil {
		return nil, err
	}
	return &Controller{
		catalog: catalog,
		client:  &http.Client{Timeout: 3 * time.Second},
	}, nil
}

func (c *Controller) Catalog() Catalog { return c.catalog }

// ConfigurePaths supplies host paths to embedded Compose recipes without
// exposing internal engine endpoints to the browser. Existing containers are
// adopted as-is; the paths are used when SparkTalk has to create one.
func (c *Controller) ConfigurePaths(dataDir, modelCache string) {
	c.mu.Lock()
	c.dataDir = strings.TrimSpace(dataDir)
	c.modelCache = strings.TrimSpace(modelCache)
	c.mu.Unlock()
}

func (c *Controller) Snapshot(ctx context.Context, selectedBundle string) Snapshot {
	gpuByPID := gpuMemoryByPID(ctx)
	statuses := make([]ComponentStatus, 0, len(c.catalog.Components))
	for _, component := range c.catalog.Components {
		statuses = append(statuses, c.componentStatus(ctx, component, gpuByPID))
	}
	c.mu.RLock()
	op := c.op
	op.Steps = append([]OperationStep(nil), c.op.Steps...)
	c.mu.RUnlock()
	if active := activeBundleFromStatuses(c.catalog, statuses); active != "" {
		selectedBundle = active
	}
	if op.Action == "start" && op.State == "running" && op.BundleID != "" {
		selectedBundle = op.BundleID
	}
	dockerState := "online"
	if err := commandOK(ctx, "docker", "info", "--format", "{{.ServerVersion}}"); err != nil {
		dockerState = "offline"
	}
	return Snapshot{
		SelectedBundle: selectedBundle,
		Bundles:        append([]Bundle(nil), c.catalog.Bundles...),
		Components:     statuses,
		Memory:         readSystemMemory(),
		Operation:      op,
		Docker:         dockerState,
	}
}

// ActiveBundle returns the managed set currently serving, or the target set
// while a switch is in progress. It deliberately does not use the configured
// startup default.
func (c *Controller) ActiveBundle(ctx context.Context) string {
	c.mu.RLock()
	op := c.op
	c.mu.RUnlock()
	if op.Action == "start" && op.State == "running" && op.BundleID != "" {
		return op.BundleID
	}
	for _, bundle := range c.catalog.Bundles {
		for _, id := range bundle.Components {
			component, _ := c.catalog.Component(id)
			if component.Role == "llm" && c.isRunning(ctx, component.Container) {
				return bundle.ID
			}
		}
	}
	return ""
}

func activeBundleFromStatuses(catalog Catalog, statuses []ComponentStatus) string {
	byID := make(map[string]ComponentStatus, len(statuses))
	for _, status := range statuses {
		byID[status.ID] = status
	}
	for _, bundle := range catalog.Bundles {
		for _, id := range bundle.Components {
			component, _ := catalog.Component(id)
			if component.Role == "llm" && byID[id].Status == "running" {
				return bundle.ID
			}
		}
	}
	return ""
}

func (c *Controller) StartBundle(ctx context.Context, bundleID string, reserveGiB float64) error {
	bundle, ok := c.catalog.Bundle(bundleID)
	if !ok {
		return fmt.Errorf("unknown bundle %q", bundleID)
	}
	if err := c.begin(Operation{Action: "start", BundleID: bundleID, State: "running", Phase: "기동 계획 준비", StartedAt: time.Now()}); err != nil {
		return err
	}
	if err := c.checkBundleStart(ctx, bundle, reserveGiB); err != nil {
		c.failCurrentStep(err.Error())
		c.finishOperation("failed", err.Error())
		return err
	}
	go c.runBundleStart(bundle, normalizedMemoryReserve(reserveGiB))
	return nil
}

// CheckBundleStart estimates unified-memory headroom after replacing another
// managed LLM and starting missing members of the requested set.
func (c *Controller) CheckBundleStart(ctx context.Context, bundleID string, reserveGiB float64) error {
	bundle, ok := c.catalog.Bundle(bundleID)
	if !ok {
		return fmt.Errorf("unknown bundle %q", bundleID)
	}
	return c.checkBundleStart(ctx, bundle, reserveGiB)
}

func (c *Controller) checkBundleStart(ctx context.Context, bundle Bundle, reserveGiB float64) error {
	plan := c.bundleMemoryPlan(ctx, bundle)
	return validateMemoryHeadroom(readSystemMemory(), plan, normalizedMemoryReserve(reserveGiB))
}

func normalizedMemoryReserve(reserveGiB float64) float64 {
	if reserveGiB <= 0 {
		return 8
	}
	return reserveGiB
}

func immediateFreeReserve(reserveGiB float64) float64 {
	if reserveGiB < minimumCUDAImmediateFreeGiB {
		return reserveGiB
	}
	return minimumCUDAImmediateFreeGiB
}

func isCUDAComponent(component Component) bool {
	return component.Role == "llm" || component.Role == "image" || component.Role == "asr" || component.Role == "tts"
}

func (c *Controller) bundleMemoryPlan(ctx context.Context, bundle Bundle) memoryPlan {
	desired := make(map[string]struct{}, len(bundle.Components))
	for _, id := range bundle.Components {
		desired[id] = struct{}{}
	}
	gpuByPID := gpuMemoryByPID(ctx)
	plan := memoryPlan{}
	for _, component := range c.catalog.Components {
		running := c.isRunning(ctx, component.Container)
		gpuMemory := 0.0
		if running {
			for _, pid := range containerPIDs(ctx, component.Container) {
				gpuMemory += gpuByPID[pid]
			}
		}
		if _, wanted := desired[component.ID]; wanted {
			if !running {
				plan.NeededGiB += component.MemoryGiB
				plan.RequiresCUDAStart = plan.RequiresCUDAStart || isCUDAComponent(component)
				continue
			}
			healthy := c.isHealthy(ctx, component)
			if !healthy {
				// Restarting releases the current allocation before rebuilding it.
				plan.NeededGiB += max(0, component.MemoryGiB-gpuMemory)
				plan.RequiresCUDAStart = plan.RequiresCUDAStart || isCUDAComponent(component)
				continue
			}
			plan.NeededGiB += healthyComponentRemainingMemory(component, gpuMemory)
			continue
		}
		if component.Role == "llm" && running {
			if gpuMemory <= 0 {
				gpuMemory = component.MemoryGiB
			}
			plan.FreedGiB += gpuMemory
		}
	}
	return plan
}

// Healthy LLM, ASR and TTS services have already loaded their steady-state
// weights. Their host-side allocations are included in MemAvailable but are
// not reported by nvidia-smi, so subtracting GPU usage from the catalog peak
// would count that memory twice. FLUX is different: its API becomes healthy
// before the generation model is loaded and still needs its remaining peak.
func healthyComponentRemainingMemory(component Component, gpuMemory float64) float64 {
	if component.Role != "image" {
		return 0
	}
	if gpuMemory <= 0 {
		return component.MemoryGiB
	}
	return max(0, component.MemoryGiB-gpuMemory)
}

func validateMemoryHeadroom(memory SystemMemory, plan memoryPlan, reserveGiB float64) error {
	if plan.NeededGiB <= 0 {
		return nil
	}
	projected := memory.AvailableGiB + plan.FreedGiB - plan.NeededGiB
	if projected < reserveGiB {
		return fmt.Errorf(
			"통합메모리 부족 예상: 시스템 가용 %.1f GiB, 반환 예정 %.1f GiB, 추가 예상 %.1f GiB, 기동 후 약 %.1f GiB (최소 여유 %.1f GiB)",
			memory.AvailableGiB, plan.FreedGiB, plan.NeededGiB, projected, reserveGiB,
		)
	}
	if plan.RequiresCUDAStart {
		immediate := memory.FreeGiB + plan.FreedGiB
		minimum := immediateFreeReserve(reserveGiB)
		if immediate < minimum {
			return fmt.Errorf(
				"CUDA 기동용 즉시 여유 메모리 부족: 현재 %.1f GiB, 반환 예정 포함 %.1f GiB (최소 %.1f GiB, 시스템 가용 %.1f GiB)",
				memory.FreeGiB, immediate, minimum, memory.AvailableGiB,
			)
		}
	}
	return nil
}

func (c *Controller) StopBundle(bundleID string) error {
	bundle, ok := c.catalog.Bundle(bundleID)
	if !ok {
		return fmt.Errorf("unknown bundle %q", bundleID)
	}
	if err := c.begin(Operation{Action: "stop", BundleID: bundleID, State: "running", Phase: "세트 중지", StartedAt: time.Now()}); err != nil {
		return err
	}
	go func() {
		var failures []string
		for index := len(bundle.Components) - 1; index >= 0; index-- {
			component, _ := c.catalog.Component(bundle.Components[index])
			c.updateOperation(component.ID, progressInfo{
				Key: "stop:" + component.ID, Phase: component.Name + " 중지", Detail: "컨테이너를 안전하게 종료하고 있습니다.",
				Progress: float64(len(bundle.Components)-1-index) / float64(len(bundle.Components)),
			})
			if err := stopContainer(context.Background(), component.Container); err != nil && !isMissingContainer(err) {
				failures = append(failures, component.Name+": "+err.Error())
			}
		}
		if len(failures) > 0 {
			c.finishOperation("failed", strings.Join(failures, "; "))
			return
		}
		c.finishOperation("complete", "")
	}()
	return nil
}

func (c *Controller) ComponentAction(componentID, action string) error {
	component, ok := c.catalog.Component(componentID)
	if !ok {
		return fmt.Errorf("unknown component %q", componentID)
	}
	if action != "start" && action != "stop" && action != "restart" {
		return errors.New("action must be start, stop, or restart")
	}
	if err := c.begin(Operation{Action: action, ComponentID: componentID, State: "running", Phase: component.Name, StartedAt: time.Now()}); err != nil {
		return err
	}
	go func() {
		var err error
		switch action {
		case "stop":
			err = stopContainer(context.Background(), component.Container)
		case "restart":
			err = stopContainer(context.Background(), component.Container)
			if err == nil || isMissingContainer(err) {
				err = c.startAndWait(component)
			}
		default:
			err = c.startAndWait(component)
		}
		if err != nil {
			c.finishOperation("failed", err.Error())
			return
		}
		c.finishOperation("complete", "")
	}()
	return nil
}

func (c *Controller) begin(op Operation) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.op.State == "running" {
		return errors.New("another runtime operation is already running")
	}
	if op.Phase != "" {
		op.Steps = []OperationStep{{Key: "plan", Phase: op.Phase, State: "current", StartedAt: time.Now()}}
	}
	c.op = op
	return nil
}

func (c *Controller) updateOperation(componentID string, info progressInfo) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if info.Key == "" {
		info.Key = info.Phase
	}
	now := time.Now()
	last := len(c.op.Steps) - 1
	if last < 0 || c.op.Steps[last].Key != info.Key || c.op.Steps[last].ComponentID != componentID {
		if last >= 0 && c.op.Steps[last].State == "current" {
			c.op.Steps[last].State = "complete"
			c.op.Steps[last].FinishedAt = now
		}
		c.op.Steps = append(c.op.Steps, OperationStep{
			Key: info.Key, ComponentID: componentID, Phase: info.Phase, Detail: info.Detail,
			State: "current", StartedAt: now,
		})
		if len(c.op.Steps) > 16 {
			c.op.Steps = append([]OperationStep(nil), c.op.Steps[len(c.op.Steps)-16:]...)
		}
	} else {
		c.op.Steps[last].Phase = info.Phase
		c.op.Steps[last].Detail = info.Detail
	}
	c.op.ComponentID = componentID
	c.op.Phase = info.Phase
	c.op.Detail = info.Detail
	c.op.Progress = info.Progress
	c.op.ETA = info.ETA
}

func (c *Controller) failCurrentStep(detail string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if last := len(c.op.Steps) - 1; last >= 0 && c.op.Steps[last].State == "current" {
		c.op.Steps[last].State = "failed"
		c.op.Steps[last].Detail = detail
		c.op.Steps[last].FinishedAt = time.Now()
	}
}

func (c *Controller) completeCurrentStep() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if last := len(c.op.Steps) - 1; last >= 0 && c.op.Steps[last].State == "current" {
		c.op.Steps[last].State = "complete"
		c.op.Steps[last].FinishedAt = time.Now()
	}
}

func (c *Controller) finishOperation(state, message string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.op.State = state
	c.op.Error = message
	c.op.Progress = 1
	c.op.ETA = ""
	c.op.FinishedAt = time.Now()
	if last := len(c.op.Steps) - 1; last >= 0 && c.op.Steps[last].State == "current" {
		if state == "complete" {
			c.op.Steps[last].State = "complete"
		} else {
			c.op.Steps[last].State = "failed"
		}
		c.op.Steps[last].FinishedAt = c.op.FinishedAt
	}
	if state == "complete" {
		if c.op.Action == "stop" {
			c.op.Phase = "중지 완료"
		} else {
			c.op.Phase = "준비 완료"
		}
		c.op.Detail = "선택한 구성을 사용할 수 있습니다."
	}
}

func (c *Controller) runBundleStart(bundle Bundle, reserveGiB float64) {
	ctx := context.Background()
	var llm Component
	for _, id := range bundle.Components {
		component, _ := c.catalog.Component(id)
		if component.Role == "llm" {
			llm = component
		}
	}
	for _, component := range c.catalog.Components {
		if component.Role == "llm" && component.ID != llm.ID {
			if !c.isRunning(ctx, component.Container) {
				continue
			}
			c.updateOperation(component.ID, progressInfo{
				Key: "replace:" + component.ID, Phase: component.Name + " 중지", Detail: "기존 언어 모델의 메모리를 반환하고 있습니다.", Progress: .02,
			})
			if err := stopContainer(ctx, component.Container); err != nil && !isMissingContainer(err) {
				c.failCurrentStep(err.Error())
				c.finishOperation("failed", component.Name+": "+err.Error())
				return
			}
		}
	}
	if err := c.waitForBundleHeadroom(bundle, reserveGiB, "세트 기동 전 메모리 재확인", 15*time.Second); err != nil {
		c.failCurrentStep(err.Error())
		c.finishOperation("failed", err.Error())
		return
	}

	ordered := append([]string(nil), bundle.Components...)
	// GB10 shares physical memory between CPU and GPU. Start small CUDA
	// services first so their contexts exist before the LLM consumes most of
	// the immediately free pages. Image weights remain lazy until generation.
	sort.SliceStable(ordered, func(i, j int) bool {
		a, _ := c.catalog.Component(ordered[i])
		b, _ := c.catalog.Component(ordered[j])
		return a.Role != "llm" && b.Role == "llm"
	})
	var failures []string
	for index, id := range ordered {
		component, _ := c.catalog.Component(id)
		if component.Role == "llm" && c.componentNeedsStart(ctx, component) {
			if err := c.waitForBundleHeadroom(bundle, reserveGiB, "언어 모델 기동 직전 메모리 확인", 15*time.Second); err != nil {
				c.failCurrentStep(err.Error())
				c.finishOperation("failed", err.Error())
				return
			}
		}
		c.updateOperation(component.ID, progressInfo{
			Key: "start:" + component.ID, Phase: component.Name + " 시작", Detail: "컨테이너와 실행 설정을 확인하고 있습니다.",
			Progress: float64(index) / float64(len(ordered)),
		})
		if err := c.startAndWait(component); err != nil {
			message := component.Name + ": " + err.Error()
			c.failCurrentStep(err.Error())
			failures = append(failures, message)
			if component.Role == "llm" {
				c.finishOperation("failed", strings.Join(failures, "; "))
				return
			}
		}
	}
	if len(failures) > 0 {
		c.completeCurrentStep()
		c.finishOperation("failed", strings.Join(failures, "; "))
		return
	}
	c.finishOperation("complete", "")
}

func (c *Controller) componentNeedsStart(ctx context.Context, component Component) bool {
	return !c.isRunning(ctx, component.Container) || !c.isHealthy(ctx, component)
}

func (c *Controller) waitForBundleHeadroom(bundle Bundle, reserveGiB float64, phase string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for {
		memory := readSystemMemory()
		plan := c.bundleMemoryPlan(context.Background(), bundle)
		lastErr = validateMemoryHeadroom(memory, plan, reserveGiB)
		c.updateOperation("", progressInfo{
			Key:      "memory:" + phase,
			Phase:    phase,
			Detail:   fmt.Sprintf("시스템 가용 %.1f GiB · 즉시 여유 %.1f GiB · 추가 예상 %.1f GiB · 최소 확보 %.1f GiB", memory.AvailableGiB, memory.FreeGiB, plan.NeededGiB, reserveGiB),
			Progress: .04,
		})
		if lastErr == nil {
			return nil
		}
		if time.Now().After(deadline) {
			return lastErr
		}
		time.Sleep(time.Second)
	}
}

func (c *Controller) startAndWait(component Component) error {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	if c.isRunning(ctx, component.Container) {
		if c.isHealthy(ctx, component) {
			return nil
		}
		// A wrapper process can remain alive after its model child has failed.
		// Recovery must restart that unhealthy container instead of polling the
		// same dead child until the startup timeout expires again.
		if err := runCommand(ctx, nil, "docker", "restart", component.Container); err != nil {
			return err
		}
	} else {
		if containerExists(ctx, component.Container) {
			if err := runCommand(ctx, nil, "docker", "start", component.Container); err != nil {
				return err
			}
		} else {
			compose, err := composeAsset(component.ComposeAsset)
			if err != nil {
				return err
			}
			project := "sparktalk-" + strings.ReplaceAll(component.ID, "_", "-")
			if err := c.ensureRuntimePaths(); err != nil {
				return err
			}
			if err := runCommandEnv(ctx, compose, c.composeEnvironment(), "docker", "compose", "-p", project, "-f", "-", "up", "-d"); err != nil {
				return err
			}
		}
	}
	timeout := time.Duration(component.StartupTimeoutSeconds) * time.Second
	if timeout <= 0 {
		timeout = 5 * time.Minute
	}
	startedAt := time.Now()
	deadline := startedAt.Add(timeout)
	for time.Now().Before(deadline) {
		if c.isHealthy(context.Background(), component) {
			c.updateOperation(component.ID, progressInfo{Key: "ready:" + component.ID, Phase: component.Name + " API 응답 확인", Detail: "서비스가 요청을 받을 준비를 마쳤습니다.", Progress: 1})
			return nil
		}
		logs := containerLogs(context.Background(), component.Container)
		if failure := startupFailure(logs); failure != "" {
			return errors.New(failure)
		}
		info := inferProgress(component, logs)
		if component.ID == "flash-next" {
			info = estimateFlashNextWeightProgress(info, time.Since(startedAt))
		}
		if info.Phase == "" {
			info = progressInfo{Key: "init", Phase: component.Name + " 준비 중", Detail: "컨테이너 로그를 기다리고 있습니다.", Progress: .05}
		}
		c.updateOperation(component.ID, info)
		if !c.isRunning(context.Background(), component.Container) {
			return fmt.Errorf("container stopped during startup: %s", lastLogLine(logs))
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("startup timed out after %s", timeout)
}

func estimateFlashNextWeightProgress(info progressInfo, elapsed time.Duration) progressInfo {
	if info.Key != "main-weights" || info.ETA != "" {
		return info
	}
	const expected = 9 * time.Minute
	if elapsed < 0 {
		elapsed = 0
	}
	ratio := min(1, float64(elapsed)/float64(expected))
	info.Progress = .16 + ratio*.32
	remaining := max(0, expected-elapsed)
	seconds := int(remaining.Round(time.Second).Seconds())
	info.ETA = fmt.Sprintf("%02d:%02d", seconds/60, seconds%60)
	return info
}

func (c *Controller) ensureRuntimePaths() error {
	c.mu.RLock()
	paths := []string{c.dataDir, c.modelCache}
	c.mu.RUnlock()
	for _, runtimePath := range paths {
		if runtimePath == "" {
			continue
		}
		if err := os.MkdirAll(runtimePath, 0750); err != nil {
			return fmt.Errorf("create runtime path %s: %w", runtimePath, err)
		}
	}
	return nil
}

func (c *Controller) composeEnvironment() []string {
	c.mu.RLock()
	dataDir, modelCache := c.dataDir, c.modelCache
	c.mu.RUnlock()
	environment := os.Environ()
	if dataDir != "" {
		environment = append(environment, "SPARKTALK_DATA_DIR="+dataDir)
	}
	if modelCache != "" {
		environment = append(environment, "SPARKTALK_HF_CACHE="+modelCache)
	}
	return environment
}

func (c *Controller) componentStatus(ctx context.Context, component Component, gpuByPID map[int]float64) ComponentStatus {
	status := ComponentStatus{Component: component, Status: "missing", Health: "offline"}
	state, _, err := inspectContainer(ctx, component.Container)
	if err != nil {
		return status
	}
	status.Status = state
	if state == "running" {
		if c.isHealthy(ctx, component) {
			status.Health = "online"
		} else {
			logs := containerLogs(ctx, component.Container)
			if failure := startupFailure(logs); failure != "" {
				status.Health = "failed"
				status.Phase = failure
			} else {
				status.Health = "starting"
				info := inferProgress(component, logs)
				status.Phase, status.Progress, status.ETA = info.Phase, info.Progress, info.ETA
			}
		}
		for _, pid := range containerPIDs(ctx, component.Container) {
			status.GPUMemoryGiB += gpuByPID[pid]
		}
	}
	return status
}

func (c *Controller) isHealthy(ctx context.Context, component Component) bool {
	if !c.isRunning(ctx, component.Container) {
		return false
	}
	if component.HealthURL == "" {
		return true
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, component.HealthURL, nil)
	if err != nil {
		return false
	}
	response, err := c.client.Do(req)
	if err != nil {
		return false
	}
	_ = response.Body.Close()
	return response.StatusCode >= 200 && response.StatusCode < 300
}

func (c *Controller) isRunning(ctx context.Context, container string) bool {
	state, _, err := inspectContainer(ctx, container)
	return err == nil && state == "running"
}

func inspectContainer(ctx context.Context, container string) (string, int, error) {
	command := exec.CommandContext(ctx, "docker", "inspect", "-f", "{{.State.Status}}|{{.State.Pid}}", container)
	output, err := command.Output()
	if err != nil {
		return "", 0, err
	}
	parts := strings.Split(strings.TrimSpace(string(output)), "|")
	if len(parts) != 2 {
		return "", 0, errors.New("unexpected docker inspect response")
	}
	pid, _ := strconv.Atoi(parts[1])
	return parts[0], pid, nil
}

func containerExists(ctx context.Context, container string) bool {
	_, _, err := inspectContainer(ctx, container)
	return err == nil
}

func stopContainer(ctx context.Context, container string) error {
	if !containerExists(ctx, container) {
		return fmt.Errorf("No such container: %s", container)
	}
	state, _, err := inspectContainer(ctx, container)
	if err != nil || state != "running" {
		return err
	}
	stopCtx, cancel := context.WithTimeout(ctx, 45*time.Second)
	defer cancel()
	return runCommand(stopCtx, nil, "docker", "stop", "-t", "30", container)
}

func isMissingContainer(err error) bool {
	return err != nil && strings.Contains(err.Error(), "No such container")
}

func runCommand(ctx context.Context, stdin []byte, name string, args ...string) error {
	return runCommandEnv(ctx, stdin, nil, name, args...)
}

func runCommandEnv(ctx context.Context, stdin []byte, environment []string, name string, args ...string) error {
	command := exec.CommandContext(ctx, name, args...)
	if environment != nil {
		command.Env = environment
	}
	if stdin != nil {
		command.Stdin = bytes.NewReader(stdin)
	}
	output, err := command.CombinedOutput()
	if err != nil {
		message := strings.TrimSpace(string(output))
		if message == "" {
			message = err.Error()
		}
		return errors.New(message)
	}
	return nil
}

func commandOK(ctx context.Context, name string, args ...string) error {
	commandCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	return runCommand(commandCtx, nil, name, args...)
}

func containerLogs(ctx context.Context, container string) string {
	commandCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	startedAt, _ := exec.CommandContext(commandCtx, "docker", "inspect", "-f", "{{.State.StartedAt}}", container).Output()
	args := []string{"logs", "--tail", "240"}
	if value := strings.TrimSpace(string(startedAt)); value != "" {
		args = append(args, "--since", value)
	}
	args = append(args, container)
	output, _ := exec.CommandContext(commandCtx, "docker", args...).CombinedOutput()
	return strings.ReplaceAll(string(output), "\r", "\n")
}

func startupFailure(logs string) string {
	lower := strings.ToLower(logs)
	if strings.Contains(lower, "cuda error: out of memory") ||
		strings.Contains(lower, "cuda out of memory") ||
		strings.Contains(lower, "nv_err_no_memory") {
		return "CUDA 메모리 부족: 새 GPU 컨텍스트 또는 모델 메모리를 할당하지 못했습니다."
	}
	return ""
}

func lastLogLine(logs string) string {
	lines := strings.Split(strings.TrimSpace(logs), "\n")
	if len(lines) == 0 {
		return "no logs"
	}
	return lines[len(lines)-1]
}

func readSystemMemory() SystemMemory {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return SystemMemory{}
	}
	values := map[string]uint64{}
	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 2 {
			continue
		}
		value, _ := strconv.ParseUint(fields[1], 10, 64)
		values[strings.TrimSuffix(fields[0], ":")] = value * 1024
	}
	total, available := values["MemTotal"], values["MemAvailable"]
	return SystemMemory{
		TotalGiB:     bytesToGiB(total),
		UsedGiB:      bytesToGiB(total - available),
		AvailableGiB: bytesToGiB(available),
		FreeGiB:      bytesToGiB(values["MemFree"]),
	}
}

func bytesToGiB(value uint64) float64 {
	return float64(value) / float64(uint64(1)<<30)
}

func gpuMemoryByPID(ctx context.Context) map[int]float64 {
	commandCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	output, err := exec.CommandContext(commandCtx, "nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil
	}
	result := make(map[int]float64)
	for _, line := range strings.Split(strings.TrimSpace(string(output)), "\n") {
		parts := strings.Split(line, ",")
		if len(parts) != 2 {
			continue
		}
		pid, _ := strconv.Atoi(strings.TrimSpace(parts[0]))
		mib, _ := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
		result[pid] = mib / 1024
	}
	return result
}

func containerPIDs(ctx context.Context, container string) []int {
	commandCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	output, err := exec.CommandContext(commandCtx, "docker", "top", container, "-eo", "pid").Output()
	if err != nil {
		return nil
	}
	var pids []int
	for index, line := range strings.Split(strings.TrimSpace(string(output)), "\n") {
		if index == 0 {
			continue
		}
		pid, err := strconv.Atoi(strings.TrimSpace(line))
		if err == nil {
			pids = append(pids, pid)
		}
	}
	return pids
}
