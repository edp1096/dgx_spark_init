package orchestrator

import (
	"embed"
	"encoding/json"
	"fmt"
)

//go:embed assets/*
var assets embed.FS

type Component struct {
	ID                    string  `json:"id"`
	Name                  string  `json:"name"`
	Role                  string  `json:"role"`
	Container             string  `json:"container"`
	HealthURL             string  `json:"health_url"`
	Model                 string  `json:"model,omitempty"`
	MemoryGiB             float64 `json:"memory_gib"`
	StartupTimeoutSeconds int     `json:"startup_timeout_seconds"`
	ComposeAsset          string  `json:"compose_asset"`
	ProgressKind          string  `json:"progress_kind"`
}

type Bundle struct {
	ID            string   `json:"id"`
	Name          string   `json:"name"`
	Description   string   `json:"description"`
	ModelType     string   `json:"model_type"`
	ModelID       string   `json:"model_id"`
	ContextTokens int      `json:"context_tokens"`
	Components    []string `json:"components"`
	MemoryGiB     float64  `json:"memory_gib"`
}

type Catalog struct {
	Components  []Component `json:"components"`
	Bundles     []Bundle    `json:"bundles"`
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
	catalog.byComponent = make(map[string]Component, len(catalog.Components))
	for _, component := range catalog.Components {
		if component.ID == "" || component.Container == "" || component.ComposeAsset == "" {
			return Catalog{}, fmt.Errorf("invalid runtime component %q", component.ID)
		}
		if _, exists := catalog.byComponent[component.ID]; exists {
			return Catalog{}, fmt.Errorf("duplicate runtime component %q", component.ID)
		}
		catalog.byComponent[component.ID] = component
	}
	catalog.byBundle = make(map[string]Bundle, len(catalog.Bundles))
	for index := range catalog.Bundles {
		bundle := catalog.Bundles[index]
		if bundle.ID == "" || len(bundle.Components) == 0 {
			return Catalog{}, fmt.Errorf("invalid runtime bundle %q", bundle.ID)
		}
		for _, id := range bundle.Components {
			component, exists := catalog.byComponent[id]
			if !exists {
				return Catalog{}, fmt.Errorf("bundle %q references unknown component %q", bundle.ID, id)
			}
			bundle.MemoryGiB += component.MemoryGiB
		}
		catalog.Bundles[index] = bundle
		catalog.byBundle[bundle.ID] = bundle
	}
	return catalog, nil
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
