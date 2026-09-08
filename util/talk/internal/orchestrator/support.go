package orchestrator

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"
)

type SupportSpec struct {
	Key          string `json:"key"`
	ID           string `json:"id"`
	Name         string `json:"name"`
	Description  string `json:"description"`
	Port         int    `json:"port"`
	InternalPort int    `json:"internal_port"`
	Image        string `json:"image"`
	Version      string `json:"version"`
	BuildAsset   string `json:"build_asset"`
	Dockerfile   string `json:"dockerfile"`
}

func SupportSpecs() []SupportSpec {
	var specs []SupportSpec
	b, err := assets.ReadFile("assets/support-services.json")
	if err != nil {
		panic(err)
	}
	if err = json.Unmarshal(b, &specs); err != nil {
		panic(err)
	}
	return specs
}
func (c Component) SupportSpec() (SupportSpec, bool) {
	for _, s := range SupportSpecs() {
		if c.ComposeAsset == "compose."+s.ID+".yaml" {
			return s, true
		}
	}
	return SupportSpec{}, false
}
func (c Component) IsSupport() bool { _, ok := c.SupportSpec(); return ok }

// Resolve support outside a model set as a shared service; set bindings still win.
func (c Catalog) ResolveSupport(bundleID, id string) (Component, bool) {
	if x, ok := c.ResolveComponent(bundleID, id); ok {
		return x, true
	}
	x, ok := c.Component(id)
	return x, ok && x.IsSupport()
}
func (c Catalog) SupportComponents(bundleID string) []Component {
	var out []Component
	for _, x := range c.Components {
		if x.IsSupport() {
			resolved, _ := c.ResolveSupport(bundleID, x.ID)
			out = append(out, resolved)
		}
	}
	return out
}
func (c Catalog) ModelBundle(bundle Bundle) Bundle {
	ids := make([]string, 0, len(bundle.Components))
	for _, id := range bundle.Components {
		x, ok := c.ResolveComponent(bundle.ID, id)
		if ok && !x.IsSupport() {
			ids = append(ids, id)
		}
	}
	bundle.Components = ids
	return bundle
}
func (c Catalog) ModelComponents(bundleID string) []Component {
	out := []Component{}
	for _, x := range c.BundleComponents(bundleID) {
		if !x.IsSupport() {
			out = append(out, x)
		}
	}
	return out
}

type SupportStatus struct {
	ComponentStatus
	Key               string `json:"key"`
	Description       string `json:"description"`
	Image             string `json:"image"`
	Version           string `json:"version"`
	Installed         string `json:"installed"`
	RunningImage      string `json:"running_image,omitempty"`
	InstallationError string `json:"installation_error,omitempty"`
}

func (c *Controller) SupportSnapshot(ctx context.Context, bundle string) []SupportStatus {
	components := c.Catalog().SupportComponents(bundle)
	out := make([]SupportStatus, len(components))
	var wg sync.WaitGroup
	for i, component := range components {
		wg.Add(1)
		go func(i int, component Component) {
			defer wg.Done()
			ctx, cancel := context.WithTimeout(ctx, 6*time.Second)
			defer cancel()
			spec, _ := component.SupportSpec()
			status := c.componentStatus(ctx, component, nil)
			installed := "external"
			detail := ""
			if component.Controller != "external" {
				raw, err := executeHost(ctx, c.host(component.Host), nil, "docker", "image", "inspect", spec.Image)
				installed = "ready"
				if err != nil {
					installed = "unknown"
					detail = fmt.Sprint(err)
					if strings.Contains(string(raw), "No such image") || strings.Contains(string(raw), "No such object") {
						installed = "missing"
						detail = ""
					}
				}
			}
			runningImage := ""
			if status.Status == "running" {
				if raw, e := executeHost(ctx, c.host(component.Host), nil, "docker", "inspect", "-f", "{{.Config.Image}}", component.Container); e == nil {
					runningImage = strings.TrimSpace(string(raw))
				}
			}
			out[i] = SupportStatus{RunningImage: runningImage, ComponentStatus: status, Key: spec.Key, Description: spec.Description, Image: spec.Image, Version: spec.Version, Installed: installed, InstallationError: detail}
		}(i, component)
	}
	wg.Wait()
	return out
}

func (c *Controller) Operation() Operation {
	c.mu.RLock()
	defer c.mu.RUnlock()
	op := c.op
	op.Steps = append([]OperationStep(nil), op.Steps...)
	return op
}
