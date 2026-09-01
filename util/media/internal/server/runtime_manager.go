package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"mediaapp/internal/jobs"
)

// runtimeDefinition is deliberately code-owned. HTTP clients can start and
// stop these names, but can never supply a compose path, service or command.
type runtimeDefinition struct {
	Compose     string
	Service     string
	Container   string
	HealthPath  string
	EstimatedGB float64
	AutoReclaim bool
}

var managedRuntimes = map[string]runtimeDefinition{
	"image":       {"krea2_turbo_nvfp4/compose.yaml", "api", "krea2-turbo-nvfp4-api", "/health", 42, true},
	"video":       {"ltx-2.5_api/compose.yaml", "api", "ltx2-5-nvfp4-api", "/health", 40, true},
	"speech":      {"qwen3_tts/compose.yaml", "custom", "qwen3-tts-custom-api", "/health", 12, true},
	"recognition": {"qwen3_asr/compose.yaml", "api", "qwen3-asr-api", "/health", 12, true},
	"prompt":      {"llama.cpp/compose.yaml", "llama-server", "llama-cpp-spark", "/v1/models", 16, true},
	"media":       {"media_access_api/compose.yaml", "api", "media-access-api", "/health", 2, false},
	"upscale":     {"seedvr2_upscaler/compose.yaml", "api", "seedvr2-upscaler-api", "/health", 32, true},
	"garment":     {"garment_extractor/compose.yaml", "api", "garment-extractor-api", "/health", 12, true},
	"faceswap":    {"reactor_faceswap/compose.yaml", "api", "reactor-faceswap-api", "/health", 10, true},
	"character":   {"minimax_h3/compose.yaml", "comfyui", "minimax-h3-comfyui", "/health", 45, true},
}

type runtimeState struct {
	Name        string  `json:"name"`
	Status      string  `json:"status"`
	Endpoint    string  `json:"endpoint"`
	Compose     string  `json:"compose"`
	Service     string  `json:"service"`
	EstimatedGB float64 `json:"estimated_peak_gb"`
	Active      bool    `json:"active"`
}

func (s *Server) runtimeEndpoint(name string) string {
	cfg := s.config()
	if name == "image" {
		backend := cfg.Image.Backends[cfg.Image.DefaultMode]
		return strings.TrimRight(backend.Endpoint, "/")
	}
	return strings.TrimRight(cfg.Engines[name].Endpoint, "/")
}

func composeRoot() (string, error) {
	current, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for directory := current; ; directory = filepath.Dir(directory) {
		candidate := filepath.Join(directory, "runtimes")
		if _, goModErr := os.Stat(filepath.Join(directory, "go.mod")); goModErr == nil {
			if info, statErr := os.Stat(candidate); statErr == nil && info.IsDir() {
				return candidate, nil
			}
		}
		parent := filepath.Dir(directory)
		if parent == directory {
			break
		}
	}
	return "", fmt.Errorf("SparkMedia runtimes directory was not found")
}

func composeFile(definition runtimeDefinition) (string, error) {
	root, err := composeRoot()
	if err != nil {
		return "", err
	}
	path := filepath.Clean(filepath.Join(root, definition.Compose))
	relative, err := filepath.Rel(root, path)
	if err != nil || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("runtime compose path escapes allowlisted root")
	}
	if _, err := os.Stat(path); err != nil {
		return "", fmt.Errorf("runtime compose file: %w", err)
	}
	return path, nil
}

func (s *Server) runtimeHealthy(name string) bool {
	definition, ok := managedRuntimes[name]
	if !ok {
		return false
	}
	endpoint := s.runtimeEndpoint(name)
	if endpoint == "" {
		return false
	}
	response, err := s.health.Get(endpoint + definition.HealthPath)
	if err != nil {
		return false
	}
	defer response.Body.Close()
	return response.StatusCode/100 == 2
}

func runtimeContainerRunning(ctx context.Context, definition runtimeDefinition) bool {
	command := exec.CommandContext(ctx, "docker", "inspect", "-f", "{{.State.Running}}", definition.Container)
	output, err := command.Output()
	return err == nil && strings.TrimSpace(string(output)) == "true"
}

func (s *Server) runtimeBusy(name string) bool {
	for _, job := range s.jobs.List() {
		if job.Status == "running" && runtimeForJob(job) == name {
			return true
		}
	}
	definition, ok := managedRuntimes[name]
	if !ok || !s.runtimeHealthy(name) {
		return false
	}
	response, err := s.health.Get(s.runtimeEndpoint(name) + definition.HealthPath)
	if err != nil {
		return false
	}
	defer response.Body.Close()
	var state struct {
		Busy bool `json:"busy"`
	}
	return json.NewDecoder(io.LimitReader(response.Body, 64<<10)).Decode(&state) == nil && state.Busy
}

func runtimeForJob(job jobs.Job) string {
	switch job.Kind {
	case "speech":
		return "speech"
	case "recognition":
		return "recognition"
	case "video":
		if stringParam(job.Params, "mode", "") == "upscale" {
			return "upscale"
		}
		return "video"
	case "image":
		switch stringParam(job.Params, "mode", "") {
		case "upscale":
			return "upscale"
		case "garment_extract":
			return "garment"
		case "face_swap":
			return "faceswap"
		default:
			return "image"
		}
	}
	return ""
}

func hostAvailableGB() float64 {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) >= 2 && fields[0] == "MemAvailable:" {
			kilobytes, _ := strconv.ParseFloat(fields[1], 64)
			return kilobytes / 1024 / 1024
		}
	}
	return 0
}

func (s *Server) reclaimForRuntime(ctx context.Context, required string) error {
	definition := managedRuntimes[required]
	const reserveGB = 8.0
	if available := hostAvailableGB(); available == 0 || available >= definition.EstimatedGB+reserveGB {
		return nil
	}
	type candidate struct {
		name string
		gb   float64
	}
	var candidates []candidate
	for name, item := range managedRuntimes {
		if name == required || !item.AutoReclaim || !runtimeContainerRunning(ctx, item) || s.runtimeBusy(name) {
			continue
		}
		candidates = append(candidates, candidate{name, item.EstimatedGB})
	}
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].gb > candidates[j].gb })
	for _, item := range candidates {
		if err := s.stopRuntimeLocked(ctx, item.name, false); err != nil {
			continue
		}
		if hostAvailableGB() >= definition.EstimatedGB+reserveGB {
			return nil
		}
	}
	if hostAvailableGB() < definition.EstimatedGB+reserveGB {
		return fmt.Errorf("%s runtime needs about %.0f GB but safe memory could not be reclaimed", required, definition.EstimatedGB)
	}
	return nil
}

func (s *Server) startRuntime(ctx context.Context, name string) error {
	definition, ok := managedRuntimes[name]
	if !ok {
		return fmt.Errorf("runtime %q is not managed", name)
	}
	s.runtimeControlMu.Lock()
	defer s.runtimeControlMu.Unlock()
	if s.runtimeHealthy(name) {
		return nil
	}
	if err := s.reclaimForRuntime(ctx, name); err != nil {
		return err
	}
	compose, err := composeFile(definition)
	if err != nil {
		return err
	}
	command := exec.CommandContext(ctx, "docker", "compose", "-f", compose, "up", "-d", definition.Service)
	if output, err := command.CombinedOutput(); err != nil {
		return fmt.Errorf("start %s runtime: %w: %s", name, err, strings.TrimSpace(string(output)))
	}
	deadline := time.NewTimer(10 * time.Minute)
	defer deadline.Stop()
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()
	for {
		if s.runtimeHealthy(name) {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-deadline.C:
			return fmt.Errorf("%s runtime did not become healthy", name)
		case <-ticker.C:
		}
	}
}

func (s *Server) stopRuntimeLocked(ctx context.Context, name string, protect bool) error {
	definition, ok := managedRuntimes[name]
	if !ok {
		return fmt.Errorf("runtime %q is not managed", name)
	}
	if protect && s.runtimeBusy(name) {
		return fmt.Errorf("%s runtime has an active operation", name)
	}
	compose, err := composeFile(definition)
	if err != nil {
		return err
	}
	command := exec.CommandContext(ctx, "docker", "compose", "-f", compose, "stop", definition.Service)
	if output, err := command.CombinedOutput(); err != nil {
		return fmt.Errorf("stop %s runtime: %w: %s", name, err, strings.TrimSpace(string(output)))
	}
	return nil
}

func (s *Server) stopRuntime(ctx context.Context, name string) error {
	s.runtimeControlMu.Lock()
	defer s.runtimeControlMu.Unlock()
	return s.stopRuntimeLocked(ctx, name, true)
}

func (s *Server) ensureGenerationRuntime(job jobs.Job) error {
	if !s.runtimeControlEnabled {
		return nil
	}
	name := runtimeForJob(job)
	if name == "" || s.runtimeHealthy(name) {
		return nil
	}
	if current, ok := s.jobs.Get(job.ID); ok && current.Status == "queued" {
		ensureJobParams(&current)
		current.Params["stage"] = "starting_" + name + "_runtime"
		_ = s.jobs.Save(current)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	if err := s.startRuntime(ctx, name); err != nil {
		return fmt.Errorf("prepare %s runtime: %w", name, err)
	}
	return nil
}

func (s *Server) runtimeStates(w http.ResponseWriter, _ *http.Request) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	names := make([]string, 0, len(managedRuntimes))
	for name := range managedRuntimes {
		names = append(names, name)
	}
	sort.Strings(names)
	states := make([]runtimeState, 0, len(names))
	for _, name := range names {
		definition := managedRuntimes[name]
		status := "stopped"
		if runtimeContainerRunning(ctx, definition) {
			status = "starting"
		}
		if s.runtimeHealthy(name) {
			status = "online"
		}
		states = append(states, runtimeState{Name: name, Status: status, Endpoint: s.runtimeEndpoint(name), Compose: definition.Compose, Service: definition.Service, EstimatedGB: definition.EstimatedGB, Active: s.runtimeBusy(name)})
	}
	writeJSON(w, http.StatusOK, states)
}

func (s *Server) startRuntimeHTTP(w http.ResponseWriter, r *http.Request) {
	if !s.runtimeControlEnabled {
		http.Error(w, "runtime control is disabled", http.StatusNotImplemented)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 10*time.Minute)
	defer cancel()
	if err := s.startRuntime(ctx, r.PathValue("name")); err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"name": r.PathValue("name"), "status": "online"})
}

func (s *Server) stopRuntimeHTTP(w http.ResponseWriter, r *http.Request) {
	if !s.runtimeControlEnabled {
		http.Error(w, "runtime control is disabled", http.StatusNotImplemented)
		return
	}
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
	defer cancel()
	err := s.stopRuntime(ctx, r.PathValue("name"))
	if err != nil {
		status := http.StatusBadGateway
		if strings.Contains(err.Error(), "active operation") {
			status = http.StatusConflict
		}
		http.Error(w, err.Error(), status)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"name": r.PathValue("name"), "status": "stopped"})
}
