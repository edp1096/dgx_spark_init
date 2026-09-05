package orchestrator

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"gopkg.in/yaml.v3"
)

func TestCatalogRoundTripAndIsolation(t *testing.T) {
	original, err := LoadCatalog()
	if err != nil {
		t.Fatal(err)
	}
	data, _ := yaml.Marshal(original)
	var imported Catalog
	if err := yaml.Unmarshal(data, &imported); err != nil {
		t.Fatal(err)
	}
	imported, err = ValidateCatalog(imported)
	if err != nil {
		t.Fatal(err)
	}
	second, err := ValidateCatalog(imported)
	if err != nil {
		t.Fatal(err)
	}
	if imported.Bundles[0].MemoryGiB != second.Bundles[0].MemoryGiB {
		t.Fatal("memory doubled on reload")
	}
	second.Components[0].Endpoint = "http://other:9000"
	if imported.Components[0].Endpoint == second.Components[0].Endpoint {
		t.Fatal("validation shared mutable components")
	}
	if _, ok := second.Bundle("glm53-worker-extra"); !ok {
		t.Fatal("cluster set missing")
	}
}

func TestCatalogRejectsBrokenReferencesAndDuplicates(t *testing.T) {
	for name, edit := range map[string]func(*Catalog){
		"host":               func(c *Catalog) { c.Components[0].Host = "missing" },
		"bundle":             func(c *Catalog) { c.Bundles = append(c.Bundles, c.Bundles[0]) },
		"service":            func(c *Catalog) { c.Bundles[0].Components = append(c.Bundles[0].Components, "missing") },
		"URL":                func(c *Catalog) { c.Components[0].Endpoint = "file:///etc/passwd" },
		"controller":         func(c *Catalog) { c.Components[0].Controller = "typo" },
		"physical duplicate": func(c *Catalog) { copy := c.Components[0]; copy.ID = "copy"; c.Components = append(c.Components, copy) },
	} {
		t.Run(name, func(t *testing.T) {
			c, _ := LoadCatalog()
			edit(&c)
			if _, err := ValidateCatalog(c); err == nil {
				t.Fatal("accepted invalid catalog")
			}
		})
	}
}

func TestRemoteCommandQuotesArguments(t *testing.T) {
	cmd := hostCommand(context.Background(), Host{Address: "worker", User: "someone", Port: 2222, IdentityFile: "/tmp/my key"}, "sh", "-c", "printf '%s' \"$1\"", "sh", "x'; touch /tmp/never; echo '")
	if !strings.Contains(strings.Join(cmd.Args, "|"), "StrictHostKeyChecking=yes") {
		t.Fatal("host verification missing")
	}
	// Exercise the exact encoded remote command locally, without executing SSH.
	encoded := cmd.Args[len(cmd.Args)-1]
	out, err := executeHost(context.Background(), Host{}, nil, "sh", "-c", encoded)
	if err != nil || string(out) != "x'; touch /tmp/never; echo '" {
		t.Fatalf("quoting lost: %q %v", out, err)
	}
}

func TestExternalHealthDoesNotRequireDocker(t *testing.T) {
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusOK) }))
	defer api.Close()
	c, _ := NewController()
	component := Component{ID: "outside", Controller: "external", Endpoint: api.URL, HealthURL: api.URL}
	if !c.isHealthy(context.Background(), component) {
		t.Fatal("external API required Docker")
	}
	if err := c.stopComponent(context.Background(), component); err != nil {
		t.Fatal(err)
	}
	if got := c.componentStatus(context.Background(), component, nil); got.Status != "external" || got.Health != "online" {
		t.Fatalf("%+v", got)
	}
}

func TestCatalogUpdateRollbackAndOperationGuard(t *testing.T) {
	c, _ := NewController()
	next, _ := LoadCatalog()
	next.Bundles[0].Name = "changed"
	if err := c.UpdateCatalog(next, func() error { return errors.New("disk full") }); err == nil {
		t.Fatal("save error discarded")
	}
	if c.Catalog().Bundles[0].Name == "changed" {
		t.Fatal("failed save changed live catalog")
	}
	if err := c.begin(Operation{State: "running"}); err != nil {
		t.Fatal(err)
	}
	if err := c.UpdateCatalog(next, nil); err == nil {
		t.Fatal("edited while starting")
	}
}

func TestRemoteStopAndMemoryUseSSH(t *testing.T) {
	dir := t.TempDir()
	log := filepath.Join(dir, "calls")
	t.Setenv("MOCK_LOG", log)
	// All tests use fake executables: never touch a real runtime or SSH host.
	script := "#!/bin/sh\nprintf '%s\\n' \"$*\" >> \"$MOCK_LOG\"\ncase \"$*\" in *meminfo*) printf 'MemTotal: 8388608 kB\\nMemAvailable: 4194304 kB\\nMemFree: 3145728 kB\\n';; esac\n"
	if err := os.WriteFile(filepath.Join(dir, "ssh"), []byte(script), 0700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+":"+os.Getenv("PATH"))
	c, _ := NewController()
	worker, _ := c.Catalog().ResolveComponent("glm53-worker-extra", "extra-collector")
	if err := c.stopComponent(context.Background(), worker); err != nil {
		t.Fatal(err)
	}
	memory, err := c.hostMemory(context.Background(), "worker")
	if err != nil || memory.AvailableGiB != 4 {
		t.Fatalf("%+v %v", memory, err)
	}
	calls, _ := os.ReadFile(log)
	if !strings.Contains(string(calls), "192.168.100.60") || !strings.Contains(string(calls), "sparktalk-extra-collector") {
		t.Fatalf("wrong remote command %s", calls)
	}
}

func TestClusterStopAttemptsWorkerAfterHeadFailure(t *testing.T) {
	dir := t.TempDir()
	log := filepath.Join(dir, "calls")
	t.Setenv("MOCK_LOG", log)
	for name, body := range map[string]string{"docker": "exit 1", "ssh": "printf 'worker stopped'"} {
		script := "#!/bin/sh\nprintf '%s %s\\n' '" + name + "' \"$*\" >> \"$MOCK_LOG\"\n" + body + "\n"
		if err := os.WriteFile(filepath.Join(dir, name), []byte(script), 0700); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("PATH", dir+":"+os.Getenv("PATH"))
	c, _ := NewController()
	component, _ := c.Catalog().Component("glm53")
	if err := c.stopComponent(context.Background(), component); err == nil {
		t.Fatal("lost head failure")
	}
	calls, _ := os.ReadFile(log)
	if !strings.Contains(string(calls), "glm53-worker") {
		t.Fatalf("worker stop skipped: %s", calls)
	}
}

func TestGLMSetMemoryDoesNotChargeWorkerExtrasToHead(t *testing.T) {
	dir := t.TempDir()
	for _, name := range []string{"docker", "nvidia-smi"} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("#!/bin/sh\nexit 1\n"), 0700); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("PATH", dir+":"+os.Getenv("PATH"))
	c, _ := NewController()
	bundle, _ := c.Catalog().Bundle("glm53-worker-extra")
	plan := c.bundleMemoryPlan(context.Background(), bundle)
	if plan.NeededGiB != 110 {
		t.Fatalf("head charged remote RAM: %+v", plan)
	}
	// Confirm JSON export includes all public host information, but no internal indexes.
	data := ExportCatalogJSON(c.Catalog())
	var value map[string]any
	if err := json.Unmarshal(data, &value); err != nil {
		t.Fatal(err)
	}
	if value["hosts"] == nil || value["byComponent"] != nil {
		t.Fatal("bad export")
	}
}

func TestComposeStartPersistsRecipeAndAppliesPorts(t *testing.T) {
	dir := t.TempDir()
	log := filepath.Join(dir, "docker-calls")
	t.Setenv("MOCK_LOG", log)
	script := `#!/bin/sh
printf '%s | port=%s bind=%s\n' "$*" "$SPARKTALK_PORT" "$SPARKTALK_BIND_ADDR" >> "$MOCK_LOG"
case "$*" in
  "image inspect "*) exit 0;;
  *inspect*) exit 1;;
  *config) cat;;
esac
`
	if err := os.WriteFile(filepath.Join(dir, "docker"), []byte(script), 0700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+":"+os.Getenv("PATH"))
	c, _ := NewController()
	c.ConfigurePaths(dir, filepath.Join(dir, "models"))
	component, _ := c.Catalog().Component("extra-ssh")
	component.Port = 18699
	component.BindAddress = "127.0.0.2"
	if err := c.startComponent(context.Background(), component); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, "runtime", "extra-ssh", "compose.yaml")
	if _, err := os.Stat(path); err != nil {
		t.Fatal("recipe not persisted", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "extra", "ssh", "keys")); err != nil {
		t.Fatal("key directory not prepared", err)
	}
	calls, _ := os.ReadFile(log)
	if !strings.Contains(string(calls), "port=18699 bind=127.0.0.2") || !strings.Contains(string(calls), path+" up -d") {
		t.Fatalf("configuration not used: %s", calls)
	}
}

func TestControllerBootUsesSavedCatalogWithoutRuntimeMutations(t *testing.T) {
	catalog, _ := LoadCatalog()
	catalog.Components[0].Endpoint = "http://saved-host:18000"
	// Startup must not compare saved routing against the built-in catalog and
	// reject it merely because the corresponding container is already running.
	dir := t.TempDir()
	log := filepath.Join(dir, "called")
	t.Setenv("MOCK_LOG", log)
	if err := os.WriteFile(filepath.Join(dir, "docker"), []byte("#!/bin/sh\necho called > \"$MOCK_LOG\"\necho true\n"), 0700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+":"+os.Getenv("PATH"))
	c, err := NewControllerWithCatalog(catalog)
	if err != nil {
		t.Fatal(err)
	}
	if c.Catalog().Components[0].Endpoint != "http://saved-host:18000" {
		t.Fatal("saved routing lost")
	}
	if _, err := os.Stat(log); !os.IsNotExist(err) {
		t.Fatal("constructor touched Docker")
	}
}

func TestExtraImageTransfer(t *testing.T) {
	for _, image := range []string{"sparktalk-extra-media:latest", "sparktalk-nemotron-asr:0.6b-q8", "sparktalk-magpie-tts:v2607-longform1"} {
		for _, fail := range []bool{false, true} {
			t.Run(image+fmt.Sprint(fail), func(t *testing.T) {
				dir := t.TempDir()
				t.Setenv("PATH", dir+":"+os.Getenv("PATH"))
				t.Setenv("TRANSFER_OUT", filepath.Join(dir, "loaded"))
				docker := "#!/bin/sh\ncase \"$*\" in *save*) printf archive;; esac\n"
				ssh := "#!/bin/sh\ncase \"$*\" in *inspect*) exit 1;; *load*) cat > \"$TRANSFER_OUT\";; esac\n"
				if fail {
					ssh = "#!/bin/sh\nexit 1\n"
				}
				for name, script := range map[string]string{"docker": docker, "ssh": ssh} {
					if err := os.WriteFile(filepath.Join(dir, name), []byte(script), 0700); err != nil {
						t.Fatal(err)
					}
				}
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer cancel()
				err := ensureLocalServiceImage(ctx, Host{Address: "worker"}, image)
				if (err != nil) != fail {
					t.Fatalf("transfer error: %v", err)
				}
				if !fail {
					data, _ := os.ReadFile(os.Getenv("TRANSFER_OUT"))
					if string(data) != "archive" {
						t.Fatalf("archive: %q", data)
					}
				}
			})
		}
	}

}
