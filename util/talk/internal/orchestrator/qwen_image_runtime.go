package orchestrator

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"strings"
	"sync"
	"time"
)

const managedWorkloadIdleGrace = 2 * time.Minute

// MemAvailable can be ample while clean model/library pages leave too few
// immediate pages for a new GB10 CUDA context beside the resident QAD engine.
const managedAuxiliaryCUDAFreeGiB = 10.0

func isManagedTTS(x Component) bool {
	return x.ComposeAsset == "compose.qwen3-tts.yaml" && x.Model == "qwen3-tts-0.6b-q8"
}

func isQwenImage21(x Component) bool {
	return x.ComposeAsset == "compose.qwen-image21.yaml" || x.ComposeAsset == "compose.qwim-mmh3.yaml"
}

type idleWorkloadState struct {
	Status    string   `json:"status"`
	Busy      *bool    `json:"busy"`
	Queued    int      `json:"queued"`
	Quiescing bool     `json:"quiescing"`
	IdleFor   *float64 `json:"idle_for_seconds"`
}

type idleWorkloadLease struct {
	component Component
	host      Host
	users     int
	lastUsed  time.Time
	timer     *time.Timer
	epoch     uint64
}

func (c *Controller) workloadIdleState(ctx context.Context, x Component) (idleWorkloadState, error) {
	ctx, cancel := context.WithTimeout(ctx, 2500*time.Millisecond)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(x.Endpoint, "/")+"/v1/runtime/memory", nil)
	if err != nil {
		return idleWorkloadState{}, err
	}
	response, err := c.client.Do(req)
	if err != nil {
		return idleWorkloadState{}, err
	}
	defer response.Body.Close()
	var state idleWorkloadState
	if response.StatusCode != http.StatusOK || json.NewDecoder(io.LimitReader(response.Body, 4096)).Decode(&state) != nil || state.Status != "ok" || state.Busy == nil {
		return state, fmt.Errorf("%s 사용 상태를 확인할 수 없습니다", x.Name)
	}
	return state, nil
}

func (c *Controller) idleWorkloadAction(ctx context.Context, x Component, action string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(x.Endpoint, "/")+"/v1/runtime/"+action, nil)
	if err != nil {
		return err
	}
	response, err := c.client.Do(req)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("%s %s HTTP %d", x.Name, action, response.StatusCode)
	}
	var state struct {
		Status string `json:"status"`
	}
	if json.NewDecoder(io.LimitReader(response.Body, 4096)).Decode(&state) != nil || state.Status != "ok" {
		return fmt.Errorf("%s %s 응답 확인 실패", x.Name, action)
	}
	return nil
}

// Quiesce atomically refuses new API jobs before Docker stops the process.
// Both active inference and requests queued behind it protect the process.
func (c *Controller) stopIdleWorkload(ctx context.Context, x Component, ownLease ...int) error {
	allowed := 0
	if len(ownLease) > 0 {
		allowed = ownLease[0]
	}
	c.idleMu.Lock()
	lease := c.idleLeases[x.DeploymentKey()]
	protected := lease != nil && lease.users > allowed
	c.idleMu.Unlock()
	if protected {
		if allowed > 0 {
			return nil
		} // The following image lease will recheck headroom.
		return fmt.Errorf("%s has queued Talk work", x.Name)
	}
	if !c.componentRunning(ctx, x) {
		return nil
	}
	if err := c.idleWorkloadAction(ctx, x, "quiesce"); err != nil {
		return err
	}
	if err := c.stopComponent(ctx, x); err != nil {
		_ = c.idleWorkloadAction(ctx, x, "resume")
		return err
	}
	state, err := c.inspectComponent(ctx, x)
	if err != nil || (state != "exited" && state != "dead") {
		return fmt.Errorf("%s 종료 확인 실패: %s %v", x.Name, state, err)
	}
	return nil
}

func (c *Controller) workloadActivity(x Component) func() {
	key := x.DeploymentKey()
	c.idleMu.Lock()
	if c.idleClosed {
		c.idleMu.Unlock()
		return func() {}
	}
	if c.idleLeases == nil {
		c.idleLeases = map[string]*idleWorkloadLease{}
	}
	lease := c.idleLeases[key]
	if lease == nil {
		lease = &idleWorkloadLease{component: x, host: c.host(x.Host)}
		c.idleLeases[key] = lease
	}
	lease.component = x
	lease.host = c.host(x.Host)
	lease.users++
	lease.epoch++
	if lease.timer != nil {
		lease.timer.Stop()
		lease.timer = nil
	}
	c.idleMu.Unlock()
	var once sync.Once
	return func() {
		once.Do(func() {
			c.idleMu.Lock()
			defer c.idleMu.Unlock()
			lease.users--
			lease.lastUsed = time.Now()
			if lease.users == 0 && !c.idleClosed && !lease.component.KeepResident {
				c.scheduleWorkloadIdleLocked(key, lease, c.workloadGrace())
			}
		})
	}
}

func (c *Controller) workloadGrace() time.Duration {
	if c.idleDuration > 0 {
		return c.idleDuration
	}
	return managedWorkloadIdleGrace
}

func (c *Controller) scheduleWorkloadIdleLocked(key string, lease *idleWorkloadLease, delay time.Duration) {
	lease.epoch++
	epoch := lease.epoch
	lease.timer = time.AfterFunc(delay, func() { c.reapIdleWorkload(key, epoch) })
}

func (c *Controller) reapIdleWorkload(key string, epoch uint64) {
	c.idleMu.Lock()
	lease := c.idleLeases[key]
	if c.idleClosed || lease == nil || lease.component.KeepResident || lease.epoch != epoch || lease.users != 0 {
		c.idleMu.Unlock()
		return
	}
	x, host := lease.component, lease.host
	c.idleMu.Unlock()
	c.workloadOnce.Do(func() { c.workloadQueue = make(chan struct{}, 1) })
	select {
	case c.workloadQueue <- struct{}{}:
		defer func() { <-c.workloadQueue }()
	default:
		c.retryWorkloadIdle(key, epoch, 5*time.Second)
		return
	}
	c.idleMu.Lock()
	protected := c.idleClosed || lease.epoch != epoch || lease.users != 0
	c.idleMu.Unlock()
	if protected || !reflect.DeepEqual(c.host(x.Host), host) || !c.componentRunning(context.Background(), x) {
		return
	}
	ctx, cancel := context.WithTimeout(c.idleLifecycle, time.Minute)
	defer cancel()
	state, err := c.workloadIdleState(ctx, x)
	if err != nil || *state.Busy || state.Queued > 0 {
		c.retryWorkloadIdle(key, epoch, 5*time.Second)
		return
	}
	if state.IdleFor != nil && *state.IdleFor >= 0 && time.Duration(*state.IdleFor*float64(time.Second)) < c.workloadGrace() {
		c.retryWorkloadIdle(key, epoch, c.workloadGrace()-time.Duration(*state.IdleFor*float64(time.Second)))
		return
	}
	c.idleMu.Lock()
	protected = c.idleClosed || lease.epoch != epoch || lease.users != 0
	if !protected {
		lease.epoch++
	}
	c.idleMu.Unlock()
	if protected {
		return
	}
	epoch++
	if err := c.begin(Operation{Action: "idle-reclaim", ComponentID: x.ID, State: "running", Phase: x.Name + " 유휴 메모리 반환", StartedAt: time.Now()}); err != nil {
		c.retryWorkloadIdle(key, epoch, 5*time.Second)
		return
	}
	if err := c.stopIdleWorkload(ctx, x); err != nil {
		c.finishOperation("failed", err.Error())
		c.retryWorkloadIdle(key, epoch, 5*time.Second)
		return
	}
	c.finishOperation("complete", "")
}

func (c *Controller) retryWorkloadIdle(key string, epoch uint64, delay time.Duration) {
	c.idleMu.Lock()
	defer c.idleMu.Unlock()
	lease := c.idleLeases[key]
	if !c.idleClosed && lease != nil && lease.users == 0 && lease.epoch == epoch {
		c.scheduleWorkloadIdleLocked(key, lease, delay)
	}
}

// Close cancels only automatic background reclamation, not a user's operation.
func (c *Controller) Close() {
	c.idleMu.Lock()
	defer c.idleMu.Unlock()
	c.idleClosed = true
	for _, lease := range c.idleLeases {
		if lease.timer != nil {
			lease.timer.Stop()
		}
	}
	if c.idleCancel != nil {
		c.idleCancel()
	}
}

func (c *Controller) acquireManagedWorkload(ctx context.Context, b Bundle, x Component, reserve float64, requested ...float64) (func() error, error) {
	requestedAt := time.Now()
	endActivity := c.workloadActivity(x) // Includes callers waiting for the lease.
	c.workloadOnce.Do(func() { c.workloadQueue = make(chan struct{}, 1) })
	select {
	case c.workloadQueue <- struct{}{}:
	case <-ctx.Done():
		endActivity()
		return nil, ctx.Err()
	}
	if err := ctx.Err(); err != nil {
		<-c.workloadQueue
		endActivity()
		return nil, err
	}
	if err := c.begin(Operation{Action: "workload", requestedAt: requestedAt, BundleID: b.ID, ComponentID: x.ID, State: "running", Phase: x.Name + " 작업 준비", StartedAt: time.Now()}); err != nil {
		<-c.workloadQueue
		endActivity()
		return nil, err
	}
	var once sync.Once
	var failure, cleanupErr error
	release := func() error {
		once.Do(func() {
			cleanupCtx, cancel := context.WithTimeout(context.Background(), time.Minute)
			defer cancel()
			if failure == nil {
				failure = ctx.Err()
			}
			memory := readSystemMemory()
			if c.memoryProbe != nil {
				memory = c.memoryProbe()
			}
			if !x.KeepResident && c.componentRunning(cleanupCtx, x) && (failure != nil || memory.AvailableGiB < normalizedMemoryReserve(reserve)) {
				cleanupErr = c.stopIdleWorkload(cleanupCtx, x, 1)
			}
			if cleanupErr != nil {
				c.finishOperation("failed", cleanupErr.Error())
			} else if failure != nil {
				c.finishOperation("failed", failure.Error())
			} else {
				c.finishOperation("complete", "")
			}
			<-c.workloadQueue
			endActivity()
		})
		return cleanupErr
	}
	fail := func(err error) (func() error, error) {
		failure = err
		if cleanup := release(); cleanup != nil {
			err = errors.Join(err, cleanup)
		}
		return nil, err
	}
	if len(requested) > 0 && requested[0] > 0 {
		x.MemoryGiB = max(x.MemoryGiB, requested[0])
	}
	cacheReclaimed := false
	for {
		if err := ctx.Err(); err != nil {
			return fail(err)
		}
		running := c.componentRunning(ctx, x)
		resident := 0.0
		if running {
			state, err := c.workloadIdleState(ctx, x)
			if err != nil {
				return fail(err)
			}
			if *state.Busy || state.Queued > 0 {
				c.updateOperation(x.ID, progressInfo{Key: "workload:wait-image", Phase: x.Name + " 기존 작업 완료 대기"})
				select {
				case <-ctx.Done():
					return fail(ctx.Err())
				case <-time.After(time.Second):
				}
				continue
			}
			gpu := gpuMemoryByPID(ctx)
			for _, pid := range containerPIDs(ctx, x.Container) {
				resident += gpu[pid]
			}
			resident += containerHostResidentMemoryGiB(ctx, x.Container)
		}
		memory := readSystemMemory()
		if c.memoryProbe != nil {
			memory = c.memoryProbe()
		}
		needed := workloadAdditionalMemory(x, resident)
		err := validateMemoryHeadroom(memory, memoryPlan{NeededGiB: needed, RequiresCUDAStart: !running, MinimumCUDAFreeGiB: managedAuxiliaryCUDAFreeGiB}, normalizedMemoryReserve(reserve))
		if err == nil {
			break
		}
		reclaimed, reclaimErr := c.reclaimIdleAuxiliary(ctx, b, x.ID, true)
		if reclaimErr != nil {
			return fail(reclaimErr)
		}
		if reclaimed {
			continue
		}
		if x.ID != "flux2" {
			if image, ok := c.Catalog().ResolveComponent(b.ID, "flux2"); ok && !image.KeepResident && isQwenImage21(image) && c.componentRunning(ctx, image) {
				if e := c.stopIdleWorkload(ctx, image); e == nil {
					continue
				}
			}
		}
		var cold *cudaStartMemoryError
		if !running && !cacheReclaimed && errors.As(err, &cold) {
			cacheReclaimed = true
			if e := c.reclaimAuxiliaryFileCache(ctx, b); e != nil {
				return fail(e)
			}
			continue
		}
		return fail(fmt.Errorf("%s 작업 전 메모리 확인: %w", x.Name, err))
	}
	if err := c.startAndWaitContext(ctx, x); err != nil {
		return fail(err)
	}
	c.updateOperation(x.ID, progressInfo{Key: "workload:execute", Phase: x.Name + " 처리 중", Detail: "연속 요청은 같은 프로세스를 이어서 사용합니다."})
	return release, nil
}

// Register already running native image services after a Talk restart. The API
// idle age protects direct calls that completed more recently than this timer.
func (c *Controller) AdoptIdleWorkloads(bundleID string) {
	b, ok := c.Catalog().Bundle(bundleID)
	if !ok {
		return
	}
	for _, id := range b.Components {
		x, found := c.Catalog().ResolveComponent(bundleID, id)
		if found && (isQwenImage21(x) || (isManagedTTS(x) && c.local(x))) {
			c.workloadActivity(x)()
		}
	}
}
