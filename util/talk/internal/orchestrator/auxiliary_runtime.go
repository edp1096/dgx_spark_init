package orchestrator

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// The lease excludes other Talk work. Also protect requests submitted directly
// to a shared service. An unknown health response is not proof of idleness.
func (c *Controller) supportIdle(ctx context.Context, x Component) (bool, error) {
	ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, x.HealthURL, nil)
	if err != nil {
		return false, err
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return false, fmt.Errorf("%s 사용 상태 확인: %w", x.Name, err)
	}
	defer resp.Body.Close()
	var state struct {
		Status string `json:"status"`
		Busy   *bool  `json:"busy"`
		Active *int   `json:"active"`
	}
	if resp.StatusCode != 200 || json.NewDecoder(io.LimitReader(resp.Body, 4096)).Decode(&state) != nil || state.Status != "ok" {
		return false, fmt.Errorf("%s 사용 상태를 확인할 수 없어 회수하지 않습니다", x.Name)
	}
	if state.Busy == nil && state.Active == nil {
		return false, fmt.Errorf("%s 작업 수를 확인할 수 없어 회수하지 않습니다", x.Name)
	}
	return (state.Busy == nil || !*state.Busy) && (state.Active == nil || *state.Active == 0), nil
}

type fluxMemoryState struct {
	CoreReady *bool `json:"core_ready"`
	Busy      *bool `json:"busy"`
}

func (c *Controller) fluxMemoryState(ctx context.Context, x Component) (fluxMemoryState, error) {
	ctx, cancel := context.WithTimeout(ctx, time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(x.Endpoint, "/")+"/v1/runtime/memory", nil)
	if err != nil {
		return fluxMemoryState{}, err
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return fluxMemoryState{}, err
	}
	defer resp.Body.Close()
	var state fluxMemoryState
	if resp.StatusCode != 200 || json.NewDecoder(io.LimitReader(resp.Body, 4096)).Decode(&state) != nil {
		return state, fmt.Errorf("Klein 사용 상태 확인 실패")
	}
	return state, nil
}

func (c *Controller) fluxRuntimeAction(ctx context.Context, x Component, action string) error {
	if x.ComposeAsset != "compose.flux2.yaml" {
		return fmt.Errorf("%s는 Klein 부품별 관리 API를 지원하지 않습니다", x.Name)
	}
	ctx, cancel := context.WithTimeout(ctx, 3*time.Minute)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(x.Endpoint, "/")+"/v1/runtime/"+action, nil)
	if err != nil {
		return err
	}
	client := *c.client
	client.Timeout = 3 * time.Minute
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("Klein %s: %w", action, err)
	}
	defer resp.Body.Close()
	var state struct {
		Status    string `json:"status"`
		CoreReady bool   `json:"core_ready"`
	}
	if resp.StatusCode != 200 || json.NewDecoder(io.LimitReader(resp.Body, 8192)).Decode(&state) != nil || state.Status != "ok" {
		return fmt.Errorf("Klein %s 실패 (HTTP %d); 본체는 중지하지 않았습니다", action, resp.StatusCode)
	}
	if action == "prepare" && !state.CoreReady {
		return fmt.Errorf("Klein 본체 사전 적재를 확인하지 못했습니다")
	}
	return nil
}

func (c *Controller) prepareResidentImage(ctx context.Context, b Bundle, reserve float64) error {
	x, ok := c.Catalog().ResolveComponent(b.ID, "flux2")
	if !ok || x.ComposeAsset != "compose.flux2.yaml" {
		return nil
	}
	for {
		memory := readSystemMemory()
		if c.memoryProbe != nil {
			memory = c.memoryProbe()
		}
		needed := x.startupMemoryGiB()
		running := c.componentRunning(ctx, x)
		if running {
			needed = max(0, needed-containerHostResidentMemoryGiB(ctx, x.Container))
			gpu := gpuMemoryByPID(ctx)
			for _, pid := range containerPIDs(ctx, x.Container) {
				needed = max(0, needed-gpu[pid])
			}
		}
		err := validateMemoryHeadroom(memory, memoryPlan{NeededGiB: needed, RequiresCUDAStart: !running}, normalizedMemoryReserve(reserve))
		if err == nil {
			break
		}
		freed, reclaimErr := c.reclaimIdleAuxiliary(ctx, b, "flux2", true)
		if reclaimErr != nil {
			return reclaimErr
		}
		if !freed {
			return err
		}
	}
	c.updateOperation(x.ID, progressInfo{Key: "preload:flux2", Phase: "Klein 본체·VAE 사전 적재"})
	if err := c.startAndWaitContext(ctx, x); err != nil {
		return err
	}
	return c.fluxRuntimeAction(ctx, x, "prepare")
}
