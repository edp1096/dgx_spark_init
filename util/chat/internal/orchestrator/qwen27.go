package orchestrator

import (
	"context"
	"encoding/json"
	"path/filepath"
)

// Use the configured cache for new installations; retain an existing local
// abliterated checkpoint mount so upgrading the app does not duplicate weights.
func (c *Controller) qwen27ModelPath(ctx context.Context, component Component, variant string) string {
	host := c.host(component.Host)
	c.mu.RLock()
	cache := c.modelCache
	c.mu.RUnlock()
	if host.ModelCache != "" {
		cache = host.ModelCache
	}
	target := filepath.Join(cache, "qwen27-"+variant)
	if variant != "abliterated" {
		return target
	}
	if _, err := executeHost(ctx, host, nil, "test", "-f", filepath.Join(target, "config.json")); err == nil {
		return target
	}
	output, err := executeHost(ctx, host, nil, "docker", "inspect", component.Container)
	if err != nil {
		return target
	}
	var containers []struct {
		Mounts []struct{ Type, Source, Destination string }
	}
	if json.Unmarshal(output, &containers) != nil || len(containers) != 1 {
		return target
	}
	for _, mount := range containers[0].Mounts {
		if mount.Type != "bind" || filepath.Base(mount.Source) != "Huihui-RadixArk-Qwen3.8-27B-abliterated-NVFP4" {
			continue
		}
		if mount.Destination != "/models/qwen27" && mount.Destination != "/models/Huihui-RadixArk-Qwen3.8-27B-abliterated-NVFP4" {
			continue
		}
		if _, err := executeHost(ctx, host, nil, "test", "-f", filepath.Join(mount.Source, "config.json")); err == nil {
			return mount.Source
		}
	}
	return target
}
