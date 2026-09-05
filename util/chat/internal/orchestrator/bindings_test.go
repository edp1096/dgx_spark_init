package orchestrator

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"gopkg.in/yaml.v3"
)

func pointer[T any](value T) *T { return &value }

func TestSharedExtraResolvesPerSet(t *testing.T) {
	catalog, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	extras := 0
	for _, component := range catalog.Components {
		if strings.HasPrefix(component.ID, "worker-extra-") {
			t.Fatal("duplicated Extra definition")
		}
		if strings.HasPrefix(component.ID, "extra-") {
			extras++
		}
	}
	if extras != 3 {
		t.Fatalf("want 3 definitions, got %d", extras)
	}
	local, _ := catalog.ResolveComponent("qwen27", "extra-collector")
	remote, _ := catalog.ResolveComponent("glm53-worker-extra", "extra-collector")
	if local.Host != "local" || remote.Host != "worker" || local.Endpoint == remote.Endpoint {
		t.Fatalf("wrong bindings: %+v %+v", local, remote)
	}
	if local.ComposeAsset != remote.ComposeAsset || local.ID != remote.ID {
		t.Fatal("not sharing a recipe")
	}
	for _, bundle := range catalog.Bundles {
		for _, component := range catalog.BundleComponents(bundle.ID) {
			if component.ServiceRole() == "collector" {
				want := "local"
				if bundle.ID == "glm53-worker-extra" || bundle.ID == "ds4fve" {
					want = "worker"
				}
				if component.Host != want {
					t.Fatalf("%s collector host = %s, want %s", bundle.ID, component.Host, want)
				}
			}
		}
	}
}

func TestBindingValidationAndExplicitReset(t *testing.T) {
	catalog, _ := LoadCatalog()
	for i := range catalog.Components {
		if catalog.Components[i].ID == "extra-collector" {
			catalog.Components[i].Port = 18695
			catalog.Components[i].BindAddress = "192.168.100.61"
		}
	}
	for i := range catalog.Bundles {
		if catalog.Bundles[i].ID == "qwen27" {
			catalog.Bundles[i].Bindings = map[string]Deployment{"extra-collector": {Port: pointer(0), BindAddress: pointer("")}}
		}
	}
	catalog, err := ValidateCatalog(catalog)
	if err != nil {
		t.Fatal(err)
	}
	resolved, _ := catalog.ResolveComponent("qwen27", "extra-collector")
	if resolved.Port != 0 || resolved.BindAddress != "" {
		t.Fatal("zero values did not override defaults")
	}
	for name, patch := range map[string]Deployment{"host": {Host: pointer("missing")}, "port": {Port: pointer(70000)}, "url": {HealthURL: pointer("file:///tmp/x")}, "memory": {MemoryGiB: pointer(-1.0)}} {
		t.Run(name, func(t *testing.T) {
			next, _ := ValidateCatalog(catalog)
			next.Bundles[0].Bindings = map[string]Deployment{"extra-collector": patch}
			if _, err := ValidateCatalog(next); err == nil {
				t.Fatal("invalid binding accepted")
			}
		})
	}
	next, _ := ValidateCatalog(catalog)
	next.Bundles[0].Bindings["not-a-member"] = Deployment{Host: pointer("worker")}
	if _, err := ValidateCatalog(next); err == nil {
		t.Fatal("orphan binding accepted")
	}
}

func TestLegacyExtraMigrationPreservesEditedDeployment(t *testing.T) {
	catalog, _ := LoadCatalog()
	remote, _ := catalog.ResolveComponent("glm53-worker-extra", "extra-collector")
	remote.ID = "worker-extra-collector"
	remote.Name = "Worker Extra Collector"
	remote.Endpoint = "http://proxy:18695"
	remote.Port = 18695
	catalog.Components = append(catalog.Components, remote)
	for i := range catalog.Bundles {
		if catalog.Bundles[i].ID != "glm53-worker-extra" {
			continue
		}
		for j, id := range catalog.Bundles[i].Components {
			if id == "extra-collector" {
				catalog.Bundles[i].Components[j] = remote.ID
			}
		}
		delete(catalog.Bundles[i].Bindings, "extra-collector")
	}
	migrated, err := ValidateCatalog(catalog)
	if err != nil {
		t.Fatal(err)
	}
	if _, exists := migrated.Component(remote.ID); exists {
		t.Fatal("legacy duplicate remains")
	}
	resolved, _ := migrated.ResolveComponent("glm53-worker-extra", "extra-collector")
	if resolved.Endpoint != remote.Endpoint || resolved.Port != 18695 || resolved.Host != "worker" {
		t.Fatalf("lost custom settings: %+v", resolved)
	}
	local, _ := migrated.ResolveComponent("qwen27", "extra-collector")
	if local.Endpoint == resolved.Endpoint {
		t.Fatal("migration changed local binding")
	}
	again, err := ValidateCatalog(migrated)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(migrated, again) {
		t.Fatal("migration is not idempotent")
	}
	for _, marshal := range []func(any) ([]byte, error){json.Marshal, yaml.Marshal} {
		data, err := marshal(migrated)
		if err != nil {
			t.Fatal(err)
		}
		var copy Catalog
		if err := yaml.Unmarshal(data, &copy); err != nil {
			t.Fatal(err)
		}
		copy, err = ValidateCatalog(copy)
		if err != nil {
			t.Fatal(err)
		}
		value, _ := copy.ResolveComponent("glm53-worker-extra", "extra-collector")
		if value.Endpoint != remote.Endpoint {
			t.Fatal("round trip lost endpoint")
		}
	}
}

func TestBindingStopTargetsWorkerOnlyAndRequiresContext(t *testing.T) {
	dir := t.TempDir()
	log := filepath.Join(dir, "calls")
	t.Setenv("BINDING_LOG", log)
	for _, name := range []string{"docker", "ssh"} {
		script := "#!/bin/sh\nprintf '%s %s\\n' '" + name + "' \"$*\" >> \"$BINDING_LOG\"\n"
		if err := os.WriteFile(filepath.Join(dir, name), []byte(script), 0700); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("PATH", dir+":"+os.Getenv("PATH"))
	c, _ := NewController()
	if err := c.ComponentAction("extra-collector", "stop"); err == nil {
		t.Fatal("ambiguous action accepted")
	}
	if err := c.ComponentAction("extra-collector", "stop", "glm53-worker-extra"); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(time.Second)
	for {
		c.mu.RLock()
		state := c.op.State
		c.mu.RUnlock()
		if state != "running" {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("stop timed out")
		}
		time.Sleep(time.Millisecond)
	}
	calls, _ := os.ReadFile(log)
	if !strings.Contains(string(calls), "ssh ") || strings.Contains(string(calls), "docker stop") {
		t.Fatalf("wrong execution host: %s", calls)
	}
	if !strings.Contains(string(calls), "192.168.100.60") || !strings.Contains(string(calls), "sparktalk-extra-collector") {
		t.Fatal("missing worker target")
	}
}

func TestBindingEditChecksRunningWorkerBeforeSaving(t *testing.T) {
	dir := t.TempDir()
	log := filepath.Join(dir, "calls")
	t.Setenv("BINDING_LOG", log)
	if err := os.WriteFile(filepath.Join(dir, "ssh"), []byte("#!/bin/sh\necho \"$*\" >> \"$BINDING_LOG\"\necho true\n"), 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "docker"), []byte("#!/bin/sh\nexit 1\n"), 0700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+":"+os.Getenv("PATH"))
	c, _ := NewController()
	next, _ := LoadCatalog()
	for i := range next.Bundles {
		if next.Bundles[i].ID == "glm53-worker-extra" {
			patch := next.Bundles[i].Bindings["extra-collector"]
			patch.Host = pointer("local")
			next.Bundles[i].Bindings["extra-collector"] = patch
		}
	}
	saved := false
	if err := c.UpdateCatalog(next, func() error { saved = true; return nil }); err == nil {
		t.Fatal("rerouted a running worker")
	}
	if saved {
		t.Fatal("persisted rejected binding")
	}
	old, _ := c.Catalog().ResolveComponent("glm53-worker-extra", "extra-collector")
	if old.Host != "worker" {
		t.Fatal("mutated controller on failure")
	}
	calls, _ := os.ReadFile(log)
	if !strings.Contains(string(calls), "sparktalk-extra-collector") {
		t.Fatal("did not inspect worker")
	}
}

func TestSnapshotUsesSelectedBinding(t *testing.T) {
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(200) }))
	defer api.Close()
	catalog, _ := LoadCatalog()
	for i := range catalog.Bundles {
		bundle := &catalog.Bundles[i]
		if bundle.ID != "glm53-worker-extra" {
			continue
		}
		for _, id := range bundle.Components {
			patch := bundle.Bindings[id]
			patch.Controller = pointer("external")
			patch.Endpoint = pointer(api.URL)
			patch.HealthURL = pointer(api.URL)
			bundle.Bindings[id] = patch
		}
	}
	c, err := NewControllerWithCatalog(catalog)
	if err != nil {
		t.Fatal(err)
	}
	snapshot := c.Snapshot(context.Background(), "glm53-worker-extra")
	if snapshot.SelectedBundle != "glm53-worker-extra" || len(snapshot.Components) != len(catalog.BundleComponents("glm53-worker-extra")) {
		t.Fatalf("wrong snapshot: %+v", snapshot)
	}
	for _, component := range snapshot.Components {
		if component.Health != "online" || component.Endpoint != api.URL {
			t.Fatalf("base endpoint used: %+v", component)
		}
		if component.ID == "extra-collector" && component.Host != "worker" {
			t.Fatal("local status substituted")
		}
	}
}
