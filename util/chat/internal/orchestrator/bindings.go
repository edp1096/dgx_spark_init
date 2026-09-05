package orchestrator

import (
	"reflect"
	"strings"
)

// Deployment contains only set-specific values. Nil inherits the shared service
// definition; a pointer to zero/empty explicitly resets a value.
type Deployment struct {
	RuntimeOptions        *map[string]string `json:"runtime_options,omitempty" yaml:"runtime_options,omitempty"`
	Host                  *string            `json:"host,omitempty" yaml:"host,omitempty"`
	Endpoint              *string            `json:"endpoint,omitempty" yaml:"endpoint,omitempty"`
	HealthURL             *string            `json:"health_url,omitempty" yaml:"health_url,omitempty"`
	Controller            *string            `json:"controller,omitempty" yaml:"controller,omitempty"`
	BindAddress           *string            `json:"bind_address,omitempty" yaml:"bind_address,omitempty"`
	Port                  *int               `json:"port,omitempty" yaml:"port,omitempty"`
	Container             *string            `json:"container,omitempty" yaml:"container,omitempty"`
	WorkerHost            *string            `json:"worker_host,omitempty" yaml:"worker_host,omitempty"`
	WorkerContainer       *string            `json:"worker_container,omitempty" yaml:"worker_container,omitempty"`
	WorkerMemoryGiB       *float64           `json:"worker_memory_gib,omitempty" yaml:"worker_memory_gib,omitempty"`
	ManagePath            *string            `json:"manage_path,omitempty" yaml:"manage_path,omitempty"`
	MemoryGiB             *float64           `json:"memory_gib,omitempty" yaml:"memory_gib,omitempty"`
	StartupTimeoutSeconds *int               `json:"startup_timeout_seconds,omitempty" yaml:"startup_timeout_seconds,omitempty"`
	Model                 *string            `json:"model,omitempty" yaml:"model,omitempty"`
	Name                  *string            `json:"name,omitempty" yaml:"name,omitempty"`
}

func (d Deployment) Apply(base Component) Component {
	source, target := reflect.ValueOf(d), reflect.ValueOf(&base).Elem()
	for i := 0; i < source.NumField(); i++ {
		if value := source.Field(i); !value.IsNil() {
			target.FieldByName(source.Type().Field(i).Name).Set(value.Elem())
		}
	}
	return base
}

func deploymentDifference(base, value Component) Deployment {
	var result Deployment
	target, source, defaults := reflect.ValueOf(&result).Elem(), reflect.ValueOf(value), reflect.ValueOf(base)
	for i := 0; i < target.NumField(); i++ {
		name := target.Type().Field(i).Name
		v := source.FieldByName(name)
		if reflect.DeepEqual(v.Interface(), defaults.FieldByName(name).Interface()) {
			continue
		}
		field := reflect.New(v.Type())
		field.Elem().Set(v)
		target.Field(i).Set(field)
	}
	return result
}

func (c Catalog) ResolveComponent(bundleID, componentID string) (Component, bool) {
	bundle, ok := c.Bundle(bundleID)
	if !ok {
		return Component{}, false
	}
	for _, id := range bundle.Components {
		if id == componentID {
			base, exists := c.Component(id)
			return bundle.Bindings[id].Apply(base), exists
		}
	}
	return Component{}, false
}

func (c Catalog) BundleComponents(bundleID string) []Component {
	bundle, ok := c.Bundle(bundleID)
	if !ok {
		return nil
	}
	components := make([]Component, 0, len(bundle.Components))
	for _, id := range bundle.Components {
		component, _ := c.ResolveComponent(bundleID, id)
		components = append(components, component)
	}
	return components
}

func (component Component) DeploymentKey() string {
	if component.Controller == "external" {
		return "external/" + component.Endpoint
	}
	return component.Host + "/" + component.Container
}

// Deployments enumerates actual execution targets, not shared recipes. Place the
// requested set first so it wins when the same container is shared by sets.
func (c Catalog) Deployments(preferred string) []Component {
	var out []Component
	seen := map[string]bool{}
	appendBundle := func(id string) {
		for _, component := range c.BundleComponents(id) {
			key := component.DeploymentKey()
			if !seen[key] {
				seen[key] = true
				out = append(out, component)
			}
		}
	}
	appendBundle(preferred)
	for _, bundle := range c.Bundles {
		if bundle.ID != preferred {
			appendBundle(bundle.ID)
		}
	}
	return out
}

// Migrate only the three legacy duplicates introduced by SparkTalk. Preserve
// edited endpoints and deployment values. Unrelated custom recipes are retained.
func migrateExtraBindings(catalog *Catalog) {
	for _, kind := range []string{"media", "ssh", "collector"} {
		oldID, newID := "worker-extra-"+kind, "extra-"+kind
		oldIndex, baseIndex := -1, -1
		for i, component := range catalog.Components {
			if component.ID == oldID {
				oldIndex = i
			}
			if component.ID == newID {
				baseIndex = i
			}
		}
		if oldIndex < 0 {
			continue
		}
		legacy := componentDefaults(catalog.Components[oldIndex])
		if legacy.ComposeAsset != "compose.extra-"+kind+".yaml" {
			continue
		}
		base := legacy
		if baseIndex >= 0 {
			base = componentDefaults(catalog.Components[baseIndex])
			if base.ComposeAsset != legacy.ComposeAsset || base.ServiceRole() != legacy.ServiceRole() || base.ProgressKind != legacy.ProgressKind {
				continue
			}
		} else {
			base.ID = newID
			base.Name = "Extra " + strings.ToUpper(kind[:1]) + kind[1:]
			if kind == "ssh" {
				base.Name = "Extra SSH"
			}
		}
		// The generated Worker prefix described placement, not a distinct service.
		if strings.EqualFold(legacy.Name, "Worker Extra "+kind) {
			legacy.Name = base.Name
		}
		conflict := false
		for _, bundle := range catalog.Bundles {
			hasOld, hasNew := false, false
			for _, id := range bundle.Components {
				hasOld = hasOld || id == oldID
				hasNew = hasNew || id == newID
			}
			if hasOld && hasNew {
				conflict = true
			}
		}
		if conflict {
			continue
		} // Validation reports duplicate roles; never discard either config.
		for i := range catalog.Bundles {
			bundle := &catalog.Bundles[i]
			for j, id := range bundle.Components {
				if id != oldID {
					continue
				}
				effective := bundle.Bindings[oldID].Apply(legacy)
				if bundle.Bindings == nil {
					bundle.Bindings = map[string]Deployment{}
				}
				bundle.Bindings[newID] = deploymentDifference(base, effective)
				delete(bundle.Bindings, oldID)
				bundle.Components[j] = newID
			}
		}
		if baseIndex < 0 {
			catalog.Components = append(catalog.Components, base)
		}
		catalog.Components = append(catalog.Components[:oldIndex], catalog.Components[oldIndex+1:]...)
	}
}

func componentDefaults(component Component) Component {
	if component.Host == "" {
		component.Host = "local"
	}
	if component.Controller == "" {
		component.Controller = "compose"
	}
	if component.Endpoint == "" {
		component.Endpoint = strings.TrimSuffix(strings.TrimSuffix(component.HealthURL, "/health"), "/ready")
	}
	return component
}
