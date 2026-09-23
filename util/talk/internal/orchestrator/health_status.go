package orchestrator

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"
)

const healthProbeInterval = 2 * time.Second
const healthGrace = 45 * time.Second
const healthFailureThreshold = 3

type containerObservation struct {
	identity string
	state    string
	started  time.Time
	oom      bool
}

type healthHistory struct {
	identity string
	checked  time.Time
	lastOK   time.Time
	latency  time.Duration
	failures int
	detail   string
}

type statusMonitor struct {
	gate    chan struct{}
	history healthHistory
}

func (h *healthHistory) reset(identity string) {
	if h.identity != identity {
		*h = healthHistory{identity: identity}
	}
}
func (h *healthHistory) record(now time.Time, latency time.Duration, err error) {
	h.checked = now
	h.latency = latency
	if err == nil {
		h.lastOK = now
		h.failures = 0
		h.detail = ""
	} else {
		h.failures++
		h.detail = err.Error()
	}
}
func (h healthHistory) health(now time.Time, instance containerObservation, startup time.Duration) string {
	if instance.oom {
		return "failed"
	}
	if instance.state != "running" && instance.state != "external" {
		return "offline"
	}
	if !h.checked.IsZero() && h.failures == 0 {
		return "online"
	}
	if !h.lastOK.IsZero() {
		if h.failures < healthFailureThreshold && now.Sub(h.lastOK) < healthGrace {
			return "online"
		}
		return "unresponsive"
	}
	// A process that was started long ago is not "loading" merely because an
	// observer just attached. Only a recent, known start can be warming up.
	if instance.state == "running" && !instance.started.IsZero() && now.Sub(instance.started) >= 0 && now.Sub(instance.started) < startup {
		return "starting"
	}
	return "unresponsive"
}

func (c *Controller) statusMonitorFor(component Component) *statusMonitor {
	// Include resolved host identities and URL so edits cannot inherit another
	// deployment's success history. Keys are internal, never emitted to logs.
	key := fmt.Sprintf("%s|%s|%s|%s|%v|%s|%v", component.ID, component.Controller, component.Container, component.HealthURL, c.host(component.Host), component.WorkerContainer, c.host(component.WorkerHost))
	c.statusMu.Lock()
	defer c.statusMu.Unlock()
	if c.statusMonitors == nil {
		c.statusMonitors = map[string]*statusMonitor{}
	}
	m := c.statusMonitors[key]
	if m == nil {
		m = &statusMonitor{gate: make(chan struct{}, 1)}
		c.statusMonitors[key] = m
	}
	return m
}
func (c *Controller) observeContainer(ctx context.Context, component Component) (containerObservation, error) {
	probe, cancel := context.WithTimeout(ctx, 6*time.Second)
	defer cancel()
	raw, err := executeHost(probe, c.host(component.Host), nil, "docker", "inspect", "-f", "{{.Id}}|{{.State.Status}}|{{.State.StartedAt}}|{{.RestartCount}}|{{.State.OOMKilled}}", component.Container)
	if err != nil {
		return containerObservation{}, err
	}
	fields := strings.Split(strings.TrimSpace(string(raw)), "|")
	if len(fields) != 5 {
		return containerObservation{}, fmt.Errorf("invalid container lifecycle response")
	}
	started, err := time.Parse(time.RFC3339Nano, fields[2])
	if err != nil {
		return containerObservation{}, fmt.Errorf("invalid container start time: %w", err)
	}
	return containerObservation{identity: fields[0] + "|" + fields[2] + "|" + fields[3], state: fields[1], started: started, oom: fields[4] == "true"}, nil
}
func (c *Controller) probeHealth(ctx context.Context, component Component) (time.Duration, error) {
	begin := time.Now()
	if component.HealthURL == "" {
		return 0, nil
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, component.HealthURL, nil)
	if err != nil {
		return time.Since(begin), fmt.Errorf("invalid health URL")
	}
	resp, err := c.client.Do(req)
	if err != nil {
		// Do not retain a URL that may contain credentials/query secrets.
		if ctx.Err() != nil {
			return time.Since(begin), ctx.Err()
		}
		if timeout, ok := err.(interface{ Timeout() bool }); ok && timeout.Timeout() {
			return time.Since(begin), fmt.Errorf("health request timed out")
		}
		return time.Since(begin), fmt.Errorf("health connection failed")
	}
	resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return time.Since(begin), fmt.Errorf("health HTTP %d", resp.StatusCode)
	}
	return time.Since(begin), nil
}
func (c *Controller) observedStatus(ctx context.Context, component Component) ComponentStatus {
	result := ComponentStatus{Component: component, Status: "unknown", Health: "unresponsive"}
	monitor := c.statusMonitorFor(component)
	select {
	case monitor.gate <- struct{}{}:
		defer func() { <-monitor.gate }()
	case <-ctx.Done():
		result.Error = ctx.Err().Error()
		return result
	}
	instance := containerObservation{identity: "external", state: "external"}
	if component.Controller != "external" {
		var err error
		instance, err = c.observeContainer(ctx, component)
		if err != nil {
			result.Error = err.Error()
			if strings.Contains(result.Error, "No such object") || strings.Contains(result.Error, "No such container") {
				result.Status = "missing"
				result.Health = "offline"
				monitor.history = healthHistory{}
			}
			return result
		}
	}
	if component.isCluster() && instance.state == "running" && !instance.oom {
		worker, err := c.observeContainer(ctx, Component{Host: component.WorkerHost, Container: component.WorkerContainer})
		if err != nil {
			result.Error = "worker lifecycle check failed: " + err.Error()
			return result
		}
		instance.identity += "|" + worker.identity
		if worker.started.After(instance.started) {
			instance.started = worker.started
		}
		if worker.state != "running" {
			instance.state = worker.state
		}
		instance.oom = instance.oom || worker.oom
	}
	result.Status = instance.state
	monitor.history.reset(instance.identity)
	if instance.state != "running" && instance.state != "external" || instance.oom {
		monitor.history = healthHistory{identity: instance.identity}
		result.Health = "offline"
		if instance.oom {
			result.Health = "failed"
			result.Phase = "OOM"
		}
		return result
	}
	if time.Since(monitor.history.checked) >= healthProbeInterval {
		latency, err := c.probeHealth(ctx, component)
		if ctx.Err() != nil {
			result.Error = ctx.Err().Error()
			return result
		} // caller cancellation is not a server failure
		monitor.history.record(time.Now(), latency, err)
	}
	h := monitor.history
	startup := time.Duration(component.StartupTimeoutSeconds) * time.Second
	if startup <= 0 {
		startup = 10 * time.Minute
	}
	result.Health = h.health(time.Now(), instance, startup)
	if !h.checked.IsZero() {
		checked := h.checked
		result.HealthCheckedAt = &checked
	}
	if !h.lastOK.IsZero() {
		last := h.lastOK
		result.LastHealthyAt = &last
	}
	result.HealthLatencyMS = h.latency.Milliseconds()
	result.HealthFailures = h.failures
	result.HealthError = h.detail
	if result.Health == "unresponsive" {
		result.Phase = "연결 이상"
		if strings.Contains(h.detail, "timed out") {
			result.Phase = "응답 지연"
		}
	}
	if result.Health == "starting" {
		probe, cancel := context.WithTimeout(ctx, 6*time.Second)
		defer cancel()
		raw, _ := executeHost(probe, c.host(component.Host), nil, "docker", "logs", "--since", instance.started.Format(time.RFC3339Nano), "--tail", "240", component.Container)
		logs := string(raw)
		if failure := startupFailure(logs); failure != "" {
			result.Health = "failed"
			result.Phase = failure
		} else {
			info := inferProgress(component, logs)
			result.Phase, result.Progress, result.ETA = info.Phase, info.Progress, info.ETA
		}
	}
	return result
}
