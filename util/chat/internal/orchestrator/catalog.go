package orchestrator

import (
	"embed"
	"encoding/json"
	"fmt"
	"net/url"
	"regexp"
	"strings"
)

//go:embed assets/*
var assets embed.FS

type Host struct {
	Address          string  `json:"address,omitempty" yaml:"address,omitempty"`
	User             string  `json:"user,omitempty" yaml:"user,omitempty"`
	Port             int     `json:"port,omitempty" yaml:"port,omitempty"`
	IdentityFile     string  `json:"identity_file,omitempty" yaml:"identity_file,omitempty"`
	DataDir          string  `json:"data_dir,omitempty" yaml:"data_dir,omitempty"`
	ModelCache       string  `json:"model_cache,omitempty" yaml:"model_cache,omitempty"`
	MemoryReserveGiB float64 `json:"memory_reserve_gib,omitempty" yaml:"memory_reserve_gib,omitempty"`
}

type Component struct {
	RuntimeOptions  map[string]string `json:"runtime_options,omitempty" yaml:"runtime_options,omitempty"`
	Host            string            `json:"host" yaml:"host"`
	Controller      string            `json:"controller" yaml:"controller"`
	Endpoint        string            `json:"endpoint" yaml:"endpoint"`
	BindAddress     string            `json:"bind_address,omitempty" yaml:"bind_address,omitempty"`
	Port            int               `json:"port,omitempty" yaml:"port,omitempty"`
	WorkerHost      string            `json:"worker_host,omitempty" yaml:"worker_host,omitempty"`
	WorkerContainer string            `json:"worker_container,omitempty" yaml:"worker_container,omitempty"`
	WorkerMemoryGiB float64           `json:"worker_memory_gib,omitempty" yaml:"worker_memory_gib,omitempty"`
	ManagePath      string            `json:"manage_path,omitempty" yaml:"manage_path,omitempty"`

	ID                    string  `json:"id" yaml:"id"`
	Name                  string  `json:"name" yaml:"name"`
	Role                  string  `json:"role" yaml:"role"`
	Container             string  `json:"container" yaml:"container"`
	HealthURL             string  `json:"health_url" yaml:"health_url"`
	Model                 string  `json:"model,omitempty" yaml:"model,omitempty"`
	MemoryGiB             float64 `json:"memory_gib" yaml:"memory_gib"`
	StartupTimeoutSeconds int     `json:"startup_timeout_seconds" yaml:"startup_timeout_seconds"`
	ComposeAsset          string  `json:"compose_asset" yaml:"compose_asset"`
	ProgressKind          string  `json:"progress_kind" yaml:"progress_kind"`
}

type Bundle struct {
	Bindings      map[string]Deployment `json:"bindings,omitempty" yaml:"bindings,omitempty"`
	ID            string                `json:"id" yaml:"id"`
	Name          string                `json:"name" yaml:"name"`
	Description   string                `json:"description" yaml:"description"`
	ModelType     string                `json:"model_type" yaml:"model_type"`
	ModelID       string                `json:"model_id" yaml:"model_id"`
	ContextTokens int                   `json:"context_tokens" yaml:"context_tokens"`
	Components    []string              `json:"components" yaml:"components"`
	MemoryGiB     float64               `json:"memory_gib" yaml:"memory_gib"`
}

type Catalog struct {
	Hosts       map[string]Host `json:"hosts" yaml:"hosts"`
	Components  []Component     `json:"components" yaml:"components"`
	Bundles     []Bundle        `json:"bundles" yaml:"bundles"`
	byComponent map[string]Component
	byBundle    map[string]Bundle
}

func LoadCatalog() (Catalog, error) {
	data, err := assets.ReadFile("assets/catalog.json")
	if err != nil {
		return Catalog{}, err
	}
	var catalog Catalog
	if err := json.Unmarshal(data, &catalog); err != nil {
		return Catalog{}, fmt.Errorf("parse embedded runtime catalog: %w", err)
	}
	return ValidateCatalog(catalog)
}

var identifier = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]*$`)

func ValidateCatalog(catalog Catalog) (Catalog, error) {
	// Do not mutate configuration snapshots or a controller catalog while rebuilding indexes.
	data, err := json.Marshal(catalog)
	if err != nil {
		return Catalog{}, err
	}
	var copy Catalog
	if err := json.Unmarshal(data, &copy); err != nil {
		return Catalog{}, err
	}
	catalog = copy
	migrateExtraBindings(&catalog)

	if len(catalog.Bundles) == 0 {
		return Catalog{}, fmt.Errorf("at least one bundle is required")
	}
	if catalog.Hosts == nil {
		catalog.Hosts = map[string]Host{"local": {}}
	}
	for id, host := range catalog.Hosts {
		if !identifier.MatchString(id) || host.Port < 0 || host.Port > 65535 || host.MemoryReserveGiB < 0 {
			return Catalog{}, fmt.Errorf("invalid host %q", id)
		}
		if host.Address != "" && (strings.HasPrefix(host.Address, "-") || strings.ContainsAny(host.Address, " \t\r\n@/")) {
			return Catalog{}, fmt.Errorf("invalid host address %q", id)
		}
		if host.User != "" && !regexp.MustCompile(`^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$`).MatchString(host.User) {
			return Catalog{}, fmt.Errorf("invalid SSH user %q", id)
		}
	}
	catalog.byComponent = make(map[string]Component, len(catalog.Components))
	physical := map[string]string{}
	for i, component := range catalog.Components {
		component.ManagePath = "" // legacy input is accepted but never used at runtime
		component = componentDefaults(component)
		if !identifier.MatchString(component.ID) || component.Name == "" || component.MemoryGiB < 0 || component.WorkerMemoryGiB < 0 {
			return Catalog{}, fmt.Errorf("invalid runtime component %q", component.ID)
		}
		if _, exists := catalog.byComponent[component.ID]; exists {
			return Catalog{}, fmt.Errorf("duplicate runtime component %q", component.ID)
		}
		if err := validateDeployment(catalog, component); err != nil {
			return Catalog{}, err
		}
		if component.Controller != "external" {
			key := component.Host + "/" + component.Container
			if other, ok := physical[key]; ok {
				return Catalog{}, fmt.Errorf("components %q and %q share a container; reuse one component ID", other, component.ID)
			}
			physical[key] = component.ID
		}
		catalog.Components[i] = component
		catalog.byComponent[component.ID] = component
	}
	catalog.byBundle = make(map[string]Bundle, len(catalog.Bundles))
	for i, bundle := range catalog.Bundles {
		switch bundle.Name {
		case "Qwen 27B 세트":
			bundle.Name = "Qwen 27B"
		case "Qwen 27B EXL3 세트":
			bundle.Name = "Qwen 27B EXL3"
		case "Flash-Next 세트":
			bundle.Name = "Flash-Next"
		case "Flash-Next EXL3 세트":
			bundle.Name = "Flash-Next EXL3"
		case "Gemma 세트":
			bundle.Name = "Gemma"
		case "GLM 5.3 Flash EXL3 + 워커 Extra":
			bundle.Name = "GLM 5.3 Flash EXL3"
		case "DeepSeek V4 Flash Vision Exp + 워커 Extra·ASR":
			bundle.Name = "DeepSeek V4 Flash Vision Exp"
		}

		for id, binding := range bundle.Bindings {
			binding.ManagePath = nil
			bundle.Bindings[id] = binding
		}
		if !identifier.MatchString(bundle.ID) || bundle.Name == "" || len(bundle.Components) == 0 {
			return Catalog{}, fmt.Errorf("invalid runtime bundle %q", bundle.ID)
		}
		if _, exists := catalog.byBundle[bundle.ID]; exists {
			return Catalog{}, fmt.Errorf("duplicate bundle %q", bundle.ID)
		}
		bundle.MemoryGiB = 0
		seen := map[string]bool{}
		roles := map[string]bool{}
		deployments := map[string]string{}
		for _, id := range bundle.Components {
			component, exists := catalog.byComponent[id]
			if !exists || seen[id] {
				return Catalog{}, fmt.Errorf("bundle %q: unknown or duplicate component %q", bundle.ID, id)
			}
			component = bundle.Bindings[id].Apply(component)
			if err := validateDeployment(catalog, component); err != nil {
				return Catalog{}, fmt.Errorf("bundle %q: %w", bundle.ID, err)
			}
			if component.Controller != "external" {
				key := component.DeploymentKey()
				if other, ok := deployments[key]; ok {
					return Catalog{}, fmt.Errorf("bundle %q: %s and %s share a container", bundle.ID, other, id)
				}
				deployments[key] = id
			}
			seen[id] = true
			role := component.ServiceRole()
			if roles[role] {
				return Catalog{}, fmt.Errorf("bundle %q: duplicate role %q", bundle.ID, role)
			}
			roles[role] = true
			if component.Controller != "external" {
				bundle.MemoryGiB += component.MemoryGiB + component.WorkerMemoryGiB
			}
		}
		for id := range bundle.Bindings {
			if !seen[id] {
				return Catalog{}, fmt.Errorf("bundle %q: binding references non-member %q", bundle.ID, id)
			}
		}
		if !roles["llm"] || bundle.ModelID == "" || bundle.ModelType == "" {
			return Catalog{}, fmt.Errorf("bundle %q: model profile is required", bundle.ID)
		}
		catalog.Bundles[i] = bundle
		catalog.byBundle[bundle.ID] = bundle
	}
	return catalog, nil
}

func (c Component) ServiceRole() string {
	if c.Role != "tool" {
		return c.Role
	}
	switch c.ComposeAsset {
	case "compose.extra-media.yaml":
		return "media"
	case "compose.extra-ssh.yaml":
		return "ssh"
	case "compose.extra-collector.yaml":
		return "collector"
	}
	return c.ID
}

func (c Catalog) Component(id string) (Component, bool) {
	component, ok := c.byComponent[id]
	return component, ok
}

func (c Catalog) Bundle(id string) (Bundle, bool) {
	bundle, ok := c.byBundle[id]
	return bundle, ok
}

func composeAsset(name string) ([]byte, error) {
	return assets.ReadFile("assets/" + name)
}

func validateDeployment(catalog Catalog, component Component) error {
	if err := validateRecipeOptions(component); err != nil {
		return err
	}
	if component.MemoryGiB < 0 || component.WorkerMemoryGiB < 0 || component.StartupTimeoutSeconds < 0 {
		return fmt.Errorf("component %q: invalid memory/timeout", component.ID)
	}

	if _, exists := catalog.Hosts[component.Host]; !exists {
		return fmt.Errorf("component %q: unknown host %q", component.ID, component.Host)
	}
	for _, endpoint := range []string{component.Endpoint, component.HealthURL} {
		u, err := url.Parse(endpoint)
		if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Hostname() == "" || u.User != nil || u.Fragment != "" {
			return fmt.Errorf("component %q: invalid API/health URL", component.ID)
		}
	}
	switch component.Role {
	case "llm", "image", "asr", "tts", "tool", "media", "ssh", "collector":
	default:
		return fmt.Errorf("component %q: invalid role", component.ID)
	}
	switch component.Controller {
	case "compose":
		if _, err := composeAsset(component.ComposeAsset); err != nil {
			return fmt.Errorf("component %q: unknown compose recipe", component.ID)
		}
	case "glm53-cluster", "dspark-cluster":
		if component.WorkerContainer == "" || component.WorkerHost == component.Host {
			return fmt.Errorf("component %q: cluster requires a distinct worker", component.ID)
		}
		if _, ok := catalog.Hosts[component.WorkerHost]; !ok {
			return fmt.Errorf("component %q: unknown worker", component.ID)
		}
	case "external":
	default:
		return fmt.Errorf("component %q: unknown controller", component.ID)
	}
	if component.Controller != "external" {
		if !regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9_.-]*$`).MatchString(component.Container) {
			return fmt.Errorf("component %q: invalid container", component.ID)
		}

	}
	if component.Port < 0 || component.Port > 65535 || strings.ContainsAny(component.BindAddress, "\r\n$") {
		return fmt.Errorf("component %q: invalid published address/port", component.ID)
	}

	return nil
}

func (c Component) isCluster() bool {
	return c.Controller == "glm53-cluster" || c.Controller == "dspark-cluster"
}
