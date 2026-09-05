package orchestrator

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// UpdateCatalog serializes persistence with runtime operations. Readers see an
// immutable catalog; a failed save leaves the previous catalog in place.
func (c *Controller) UpdateCatalog(next Catalog, persist func() error) error {
	validated, err := ValidateCatalog(next)
	if err != nil {
		return err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if !reflect.DeepEqual(c.catalog, validated) && c.op.State == "running" {
		return fmt.Errorf("세트 작업이 끝난 뒤 구성을 저장하세요")
	}
	// Compare effective deployments, including both cluster nodes. A binding edit
	// must not redirect stop/status operations away from a running container.
	checked := map[string]bool{}
	for _, oldBundle := range c.catalog.Bundles {
		newBundle, bundleExists := validated.Bundle(oldBundle.ID)
		profileChanged := !bundleExists || oldBundle.ModelID != newBundle.ModelID || oldBundle.ModelType != newBundle.ModelType || oldBundle.ContextTokens != newBundle.ContextTokens
		for _, old := range c.catalog.BundleComponents(oldBundle.ID) {
			nextComponent, exists := validated.ResolveComponent(oldBundle.ID, old.ID)
			if !profileChanged && exists && reflect.DeepEqual(old, nextComponent) && reflect.DeepEqual(c.catalog.Hosts[old.Host], validated.Hosts[nextComponent.Host]) && reflect.DeepEqual(c.catalog.Hosts[old.WorkerHost], validated.Hosts[nextComponent.WorkerHost]) {
				continue
			}
			if old.Controller == "external" {
				continue
			}
			nodes := []struct{ host, container string }{{old.Host, old.Container}}
			if old.isCluster() {
				nodes = append(nodes, struct{ host, container string }{old.WorkerHost, old.WorkerContainer})
			}
			for _, node := range nodes {
				key := node.host + "/" + node.container
				if checked[key] {
					continue
				}
				checked[key] = true
				ctx, cancel := context.WithTimeout(context.Background(), 6*time.Second)
				out, err := executeHost(ctx, c.catalog.Hosts[node.host], nil, "docker", "inspect", "-f", "{{.State.Running}}", node.container)
				cancel()
				if err == nil && strings.TrimSpace(string(out)) == "true" {
					return fmt.Errorf("%s의 %s (%s) 중지 후 배치·세트 구성을 변경하세요", oldBundle.Name, old.Name, node.host)
				}
			}
		}
	}
	if persist != nil {
		if err := persist(); err != nil {
			return err
		}
	}
	c.catalog = validated
	return nil
}

func shellQuote(value string) string { return "'" + strings.ReplaceAll(value, "'", "'\"'\"'") + "'" }

func hostCommand(ctx context.Context, host Host, args ...string) *exec.Cmd {
	if host.Address == "" {
		return exec.CommandContext(ctx, args[0], args[1:]...)
	}
	sshArgs := []string{"-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=5", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2"}
	if host.Port != 0 {
		sshArgs = append(sshArgs, "-p", strconv.Itoa(host.Port))
	}
	if host.IdentityFile != "" {
		sshArgs = append(sshArgs, "-i", host.IdentityFile)
	}
	if host.User != "" {
		sshArgs = append(sshArgs, "-l", host.User)
	}
	quoted := make([]string, len(args))
	for i, arg := range args {
		quoted[i] = shellQuote(arg)
	}
	sshArgs = append(sshArgs, "--", host.Address, strings.Join(quoted, " "))
	return exec.CommandContext(ctx, "ssh", sshArgs...)
}

func executeHost(ctx context.Context, host Host, stdin []byte, args ...string) ([]byte, error) {
	command := hostCommand(ctx, host, args...)
	if stdin != nil {
		command.Stdin = bytes.NewReader(stdin)
	}
	output, err := command.CombinedOutput()
	if err != nil {
		return output, fmt.Errorf("%s: %w", strings.TrimSpace(string(output)), err)
	}
	return output, nil
}

func (c *Controller) host(id string) Host            { return c.Catalog().Hosts[id] }
func (c *Controller) local(component Component) bool { return c.host(component.Host).Address == "" }

func (c *Controller) inspectComponent(ctx context.Context, component Component) (string, error) {
	probeCtx, cancel := context.WithTimeout(ctx, 6*time.Second)
	defer cancel()
	output, err := executeHost(probeCtx, c.host(component.Host), nil, "docker", "inspect", "-f", "{{.State.Status}}", component.Container)
	return strings.TrimSpace(string(output)), err
}

func (c *Controller) componentRunning(ctx context.Context, component Component) bool {
	if component.Controller == "external" {
		return c.httpHealthy(ctx, component)
	}
	state, err := c.inspectComponent(ctx, component)
	return err == nil && state == "running"
}

func (c *Controller) componentLogs(ctx context.Context, component Component) string {
	if component.Controller == "external" {
		return ""
	}
	if c.local(component) {
		return containerLogs(ctx, component.Container)
	}
	probeCtx, cancel := context.WithTimeout(ctx, 6*time.Second)
	defer cancel()
	output, _ := executeHost(probeCtx, c.host(component.Host), nil, "docker", "logs", "--tail", "240", component.Container)
	return strings.ReplaceAll(string(output), "\r", "\n")
}

func (c *Controller) stopComponent(ctx context.Context, component Component) error {
	if component.Controller == "external" {
		return nil
	}
	stopCtx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()
	if component.isCluster() {
		// Attempt both sides even when one node is offline. This avoids leaving the
		// reachable head running when manage.sh's SSH preflight cannot reach worker.
		var failures []string
		for _, node := range []struct{ host, container string }{{component.Host, component.Container}, {component.WorkerHost, component.WorkerContainer}} {
			out, err := executeHost(stopCtx, c.host(node.host), nil, "docker", "stop", "-t", "30", node.container)
			if err != nil && !strings.Contains(string(out), "No such container") {
				failures = append(failures, node.host+": "+err.Error())
			}
		}
		if len(failures) > 0 {
			return fmt.Errorf("%s", strings.Join(failures, "; "))
		}
		return nil
	}
	if c.local(component) {
		return stopContainer(ctx, component.Container)
	}
	_, err := executeHost(stopCtx, c.host(component.Host), nil, "docker", "stop", "-t", "30", component.Container)
	return err
}

func (c *Controller) startComponent(ctx context.Context, component Component) error {
	if component.Controller == "external" {
		return nil
	}
	host := c.host(component.Host)
	if component.isCluster() {

		workerComponent := Component{Host: component.WorkerHost, Container: component.WorkerContainer}
		if c.componentRunning(ctx, component) || c.componentRunning(ctx, workerComponent) {
			if err := c.stopComponent(ctx, component); err != nil {
				return err
			}
		}
		return c.runEmbeddedRecipe(ctx, component, "start", "")
	}
	data, err := composeAsset(component.ComposeAsset)
	if err != nil {
		return err
	}
	var recipe map[string]any
	if err := yaml.Unmarshal(data, &recipe); err != nil {
		return err
	}
	service := recipe["services"].(map[string]any)["runtime"].(map[string]any)
	if image, ok := service["image"].(string); ok {
		if err := ensureLocalServiceImage(ctx, host, image); err != nil {
			return err
		}
	}
	service["container_name"] = component.Container
	if recipeID(component) == "flash-next-exl3" && component.RuntimeOptions["MODEL_VARIANT"] == "official" {
		if environment, ok := service["environment"].(map[string]any); ok {
			environment["EXL3_ABLIT_LAMBDA"] = "0"
		}
	}
	// Use Compose interpolation only for host-side paths and declared published
	// ports. The API endpoint can independently refer to a proxy or SSH tunnel.
	data, err = yaml.Marshal(recipe)
	if err != nil {
		return err
	}
	env := []string{"env"}
	if host.DataDir != "" {
		env = append(env, "SPARKTALK_DATA_DIR="+host.DataDir)
	} else if host.Address == "" {
		c.mu.RLock()
		path := c.dataDir
		c.mu.RUnlock()
		if path != "" {
			env = append(env, "SPARKTALK_DATA_DIR="+path)
		}
	}
	if host.ModelCache != "" {
		env = append(env, "SPARKTALK_HF_CACHE="+host.ModelCache)
	} else if host.Address == "" {
		c.mu.RLock()
		path := c.modelCache
		c.mu.RUnlock()
		if path != "" {
			env = append(env, "SPARKTALK_HF_CACHE="+path)
		}
	}
	if component.BindAddress != "" {
		env = append(env, "SPARKTALK_BIND_ADDR="+component.BindAddress)
	}
	if component.Port != 0 {
		env = append(env, "SPARKTALK_PORT="+strconv.Itoa(component.Port))
	}
	// Persist the rendered recipe on its execution host, so docker compose down
	// can use a real config file rather than the former stdin-only recipe.
	directory := host.DataDir
	if directory == "" && host.Address == "" {
		c.mu.RLock()
		directory = c.dataDir
		c.mu.RUnlock()
	}
	if directory == "" {
		return fmt.Errorf("host %s requires data_dir", component.Host)
	}
	if component.ServiceRole() == "ssh" {
		for _, subdir := range []string{"keys", "state"} {
			if _, err := executeHost(ctx, host, nil, "mkdir", "-p", filepath.Join(directory, "extra", "ssh", subdir)); err != nil {
				return err
			}
		}
	}
	directory = filepath.Join(directory, "runtime", component.ID)
	if _, err := executeHost(ctx, host, nil, "mkdir", "-p", directory); err != nil {
		return err
	}
	configPath := filepath.Join(directory, "compose.yaml")
	renderArgs := append(append([]string{}, env...), "docker", "compose", "-p", "sparktalk-"+component.ID, "-f", "-", "config")
	command := hostCommand(ctx, host, renderArgs...)
	command.Stdin = bytes.NewReader(data)
	var stderr bytes.Buffer
	command.Stderr = &stderr
	rendered, err := command.Output()
	if err != nil {
		return fmt.Errorf("render compose: %s: %w", stderr.String(), err)
	}
	if _, err := executeHost(ctx, host, rendered, "sh", "-c", "umask 077; cat > \"$1\"", "sh", configPath); err != nil {
		return err
	}
	// Recreate a stopped/unhealthy named container using the edited recipe,
	// including one originally created under another Compose project.
	state, inspectErr := c.inspectComponent(ctx, component)
	if inspectErr == nil {
		if state == "running" {
			if _, err := executeHost(ctx, host, nil, "docker", "stop", "-t", "30", component.Container); err != nil {
				return err
			}
		}
		if _, err := executeHost(ctx, host, nil, "docker", "rm", component.Container); err != nil {
			return err
		}
	}
	_, err = executeHost(ctx, host, nil, "docker", "compose", "-p", "sparktalk-"+component.ID, "-f", configPath, "up", "-d")
	return err
}

// Extra and speech images are built locally, not published to Docker Hub. Stream missing
// images to the execution host without buffering the archive in unified memory.
func ensureLocalServiceImage(ctx context.Context, host Host, image string) error {
	switch image {
	case "sparktalk-extra-media:latest", "sparktalk-extra-ssh:latest", "sparktalk-extra-collector:latest",
		"sparktalk-nemotron-asr:0.6b-q8", "sparktalk-magpie-tts:v2607-longform1":
	default:
		return nil
	}
	if _, err := executeHost(ctx, host, nil, "docker", "image", "inspect", image); err == nil {
		return nil
	}
	if _, err := executeHost(ctx, Host{}, nil, "docker", "image", "inspect", image); err != nil {
		return fmt.Errorf("서비스 이미지 %s를 앱 실행 머신에서 먼저 빌드하세요: %w", image, err)
	}
	transferCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	save := hostCommand(transferCtx, Host{}, "docker", "image", "save", image)
	var saveError bytes.Buffer
	save.Stderr = &saveError
	stream, err := save.StdoutPipe()
	if err != nil {
		return err
	}
	defer stream.Close()
	if err := save.Start(); err != nil {
		return err
	}
	loader := hostCommand(transferCtx, host, "docker", "image", "load")
	loader.Stdin = stream
	output, loadErr := loader.CombinedOutput()
	if loadErr != nil {
		cancel()
	}
	saveErr := save.Wait()
	if loadErr != nil {
		return fmt.Errorf("서비스 이미지 %s 전달 실패: %s: %w", image, output, loadErr)
	}
	if saveErr != nil {
		return fmt.Errorf("서비스 이미지 %s 내보내기 실패: %s: %w", image, saveError.String(), saveErr)
	}
	return nil
}

func (c *Controller) hostMemory(ctx context.Context, hostID string) (SystemMemory, error) {
	host := c.host(hostID)
	if host.Address == "" {
		return readSystemMemory(), nil
	}
	probeCtx, cancel := context.WithTimeout(ctx, 6*time.Second)
	defer cancel()
	data, err := executeHost(probeCtx, host, nil, "cat", "/proc/meminfo")
	if err != nil {
		return SystemMemory{}, err
	}
	memory := parseSystemMemory(data)
	if memory.TotalGiB == 0 {
		return memory, fmt.Errorf("invalid memory response from %s", hostID)
	}
	return memory, nil
}

// ExportCatalogJSON makes an independent copy suitable for configuration/UI
// editing; private lookup indexes are rebuilt only after validation.
func ExportCatalogJSON(catalog Catalog) []byte {
	data, _ := json.MarshalIndent(catalog, "", "  ")
	return data
}

func activeBundleFromStatusesPreferred(catalog Catalog, statuses []ComponentStatus, preferred string) string {
	if bundle, ok := catalog.Bundle(preferred); ok {
		for _, id := range bundle.Components {
			for _, status := range statuses {
				if status.ID == id && status.Role == "llm" && (status.Health == "online" || status.Status == "running") {
					return preferred
				}
			}
		}
	}
	return activeBundleFromStatuses(catalog, statuses)
}

func (c *Controller) checkRemoteMemory(ctx context.Context, bundle Bundle, defaultReserve float64) error {
	catalog := c.Catalog()
	plans := map[string]memoryPlan{}
	for _, id := range bundle.Components {
		component, _ := catalog.ResolveComponent(bundle.ID, id)
		if component.Controller == "external" {
			continue
		}
		if !c.local(component) {
			plan := plans[component.Host]
			if !c.componentRunning(ctx, component) {
				plan.NeededGiB += component.MemoryGiB
				plan.RequiresCUDAStart = plan.RequiresCUDAStart || isCUDAComponent(component)
			}
			plans[component.Host] = plan
		}
		if component.isCluster() {
			worker := Component{Host: component.WorkerHost, Container: component.WorkerContainer}
			plan := plans[component.WorkerHost]
			if !c.componentRunning(ctx, worker) {
				plan.NeededGiB += component.WorkerMemoryGiB
				plan.RequiresCUDAStart = true
			}
			plans[component.WorkerHost] = plan
		}
	}
	for hostID, plan := range plans {
		memory, err := c.hostMemory(ctx, hostID)
		if err != nil {
			return fmt.Errorf("호스트 %s 연결·메모리 확인 실패: %w", hostID, err)
		}
		reserve := catalog.Hosts[hostID].MemoryReserveGiB
		if reserve <= 0 {
			reserve = normalizedMemoryReserve(defaultReserve)
		}
		if err := validateMemoryHeadroom(memory, plan, reserve); err != nil {
			return fmt.Errorf("호스트 %s: %w", hostID, err)
		}
	}
	return nil
}

type HostStatus struct {
	Memory SystemMemory `json:"memory"`
	Error  string       `json:"error,omitempty"`
}

func (c *Controller) ActiveBundlePreferred(ctx context.Context, preferred string) string {
	c.mu.RLock()
	op := c.op
	c.mu.RUnlock()
	if op.Action == "start" && op.State == "running" && op.BundleID != "" {
		return op.BundleID
	}
	if bundle, ok := c.Catalog().Bundle(preferred); ok {
		for _, id := range bundle.Components {
			component, _ := c.Catalog().ResolveComponent(bundle.ID, id)
			if component.Role == "llm" && c.componentRunning(ctx, component) {
				return preferred
			}
		}
	}
	return c.ActiveBundle(ctx)
}

func (c *Controller) localMemoryReserve(bundle Bundle, fallback float64) float64 {
	catalog := c.Catalog()
	for _, id := range bundle.Components {
		component, _ := catalog.ResolveComponent(bundle.ID, id)
		host := catalog.Hosts[component.Host]
		if component.Controller != "external" && host.Address == "" && host.MemoryReserveGiB > 0 {
			return host.MemoryReserveGiB
		}
	}
	return normalizedMemoryReserve(fallback)
}
