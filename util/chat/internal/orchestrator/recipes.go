package orchestrator

import (
	"bytes"
	"context"
	"crypto/sha256"
	"fmt"
	"net"
	"os/user"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

func recipeID(c Component) string {
	switch c.Controller {
	case "glm53-cluster":
		return "glm53"
	case "dspark-cluster":
		return "ds4fve"
	}
	switch c.ComposeAsset {
	case "compose.qwen27-exl3.yaml":
		return "qwen27-exl3"
	case "compose.flash-next-exl3.yaml":
		return "flash-next-exl3"
	}
	return ""
}

var recipeOptionNames = map[string]bool{
	"MODEL_VARIANT": true, "HEAD_RAIL_IP": true, "WORKER_RAIL_IP": true,
	"HEAD_NCCL_IF": true, "WORKER_NCCL_IF": true, "HEAD_NCCL_HCA": true, "WORKER_NCCL_HCA": true,
	"NCCL_SUBNET": true, "MASTER_PORT": true, "MAX_MODEL_LEN": true, "MAX_NUM_SEQS": true,
	"GPU_MEMORY_UTILIZATION": true, "GPU_MEMORY_UTILIZATION_TEXT": true,
	"KV_CACHE_MEMORY": true, "MTP_TOKENS": true, "DFLASH_TOKENS": true,
}

func validateRecipeOptions(c Component) error {
	for k, v := range c.RuntimeOptions {
		if !recipeOptionNames[k] || strings.ContainsAny(v, "\x00\r\n") {
			return fmt.Errorf("%s: unsupported runtime option %s", c.ID, k)
		}
		if k == "MODEL_VARIANT" && v != "official" && v != "abliterated" {
			return fmt.Errorf("invalid model variant")
		}
	}
	return nil
}

// Each recipe is embedded. Runtime never reads or invokes the workspace checkout.
func (c *Controller) materializeRecipe(ctx context.Context, component Component) (string, error) {
	id := recipeID(component)
	data, err := assets.ReadFile("assets/recipes/" + id + ".tar.gz")
	if err != nil {
		return "", err
	}
	head := c.host(component.Host)
	c.mu.RLock()
	dataDir, cache := c.dataDir, c.modelCache
	c.mu.RUnlock()
	if head.DataDir != "" {
		dataDir = head.DataDir
	}
	if head.ModelCache != "" {
		cache = head.ModelCache
	}
	if !filepath.IsAbs(dataDir) || !filepath.IsAbs(cache) {
		return "", fmt.Errorf("absolute runtime data/cache directories are required")
	}
	sum := sha256.Sum256(data)
	dir := filepath.Join(dataDir, "runtime", "recipes", fmt.Sprintf("%s-%x", id, sum[:8]))
	if _, err = executeHost(ctx, head, data, "sh", "-c", `set -eu; umask 077; mkdir -p "$1"; tar -xzf - -C "$1"`, "sh", dir); err != nil {
		return "", err
	}
	values := map[string]string{
		"MODEL_KIND": id, "HF_CACHE": cache, "API_PORT": strconv.Itoa(component.Port), "VLLM_PORT": strconv.Itoa(component.Port),
		"MODEL_HOST_PATH":   filepath.Join(cache, map[string]string{"glm53": "glm53-exl3", "qwen27-exl3": "exl3-qwen38-27b-uncensored-4bpw", "flash-next-exl3": "exl3-qwen38-fn-4.05bpw"}[id]),
		"SERVED_MODEL_NAME": component.Model,
	}
	if component.Port == 0 {
		values["API_PORT"] = "8000"
		values["VLLM_PORT"] = "8888"
	}
	values["GLM53_CACHE_PATH"] = filepath.Join(filepath.Dir(cache), "glm53-vllm")
	values["DFLASH_HOST_PATH"] = filepath.Join(cache, "glm53-dflash2-mxfp8")
	values["ABLIT_HOST_PATH"] = filepath.Join(cache, "glm53-ablit-oproj")
	if id == "glm53" {
		values["ABLIT_HOST_PATH"] = filepath.Join(cache, "glm53-lovesenko-oproj")
		values["ABLIT_LAYERS"] = "0-44"
		values["ABLIT_INCLUDE_MTP"] = "0"
		values["ABLIT_DONOR"] = "lovesenko/GLM-5.3-Flash-tr3-4bpw-Abliterated"
		values["ABLIT_DONOR_REVISION"] = "c8f58e6aa9117c73607d692978b22f091d80450c"
	}
	values["EXL3_CACHE_PATH"] = filepath.Join(filepath.Dir(cache), map[string]string{"qwen27-exl3": "exl3-qwen38-27b", "flash-next-exl3": "exl3-qwen38-fn"}[id])
	values["ABLIT_OUTPUT_PATH"] = filepath.Join(values["EXL3_CACHE_PATH"], "ablit")
	if component.isCluster() {
		worker := c.host(component.WorkerHost)
		workerDir := worker.DataDir
		if workerDir == "" {
			workerDir = dataDir
		}
		workerCache := worker.ModelCache
		if workerCache == "" {
			workerCache = cache
		}
		workerUser := worker.User
		if workerUser == "" {
			u, e := user.Current()
			if e != nil {
				return "", e
			}
			workerUser = u.Username
		}
		if !regexp.MustCompile(`^[A-Za-z0-9._-]+$`).MatchString(workerUser) {
			return "", fmt.Errorf("invalid worker user")
		}
		values["WORKER_HOST"] = workerUser + "@" + worker.Address
		values["WORKER_LAN_IP"] = worker.Address
		values["WORKER_USER"] = workerUser
		remoteDir := filepath.Join(workerDir, "runtime", "recipes", fmt.Sprintf("%s-%x", id, sum[:8]))
		values["REMOTE_COMPOSE_DIR"] = remoteDir
		values["WORKER_DIR"] = filepath.Join(remoteDir, "upstream")
		values["WORKER_SCRIPT_DIR"] = values["WORKER_DIR"]
		values["WORKER_HF_CACHE"] = workerCache
		lan := head.Address
		if lan == "" {
			conn, e := net.Dial("udp", net.JoinHostPort(worker.Address, "22"))
			if e != nil {
				return "", e
			}
			lan = conn.LocalAddr().(*net.UDPAddr).IP.String()
			conn.Close()
		}
		values["HEAD_LAN_IP"] = lan
		values["HEAD_RAIL_IP"] = "10.200.0.1"
		values["WORKER_RAIL_IP"] = "10.200.0.2"
		values["HEAD_NCCL_IF"] = "enp1s0f1np1"
		values["WORKER_NCCL_IF"] = "enp1s0f1np1"
		values["HEAD_NCCL_HCA"] = "rocep1s0f1"
		values["WORKER_NCCL_HCA"] = "rocep1s0f1"
		values["NCCL_SUBNET"] = "10.200.0.0/24"
	}
	for k, v := range component.RuntimeOptions {
		values[k] = v
	}
	variant := values["MODEL_VARIANT"]
	if variant == "" {
		variant = "official"
		if id == "qwen27-exl3" || id == "flash-next-exl3" {
			variant = "abliterated"
		}
		values["MODEL_VARIANT"] = variant
	}
	values["ABLIT"] = "0"
	values["ABLITERATED"] = "0"
	values["ABLIT_LAMBDA"] = "0"
	if variant == "abliterated" {
		values["ABLIT"] = "1"
		values["ABLITERATED"] = "1"
		values["ABLIT_LAMBDA"] = "1.5"
	}
	if id == "ds4fve" {
		values["MASTER_ADDR"] = values["HEAD_RAIL_IP"]
		values["VLLM_HOST_IP"] = values["HEAD_RAIL_IP"]
		values["WORKER_VLLM_HOST_IP"] = values["WORKER_RAIL_IP"]
		values["NCCL_IB_HCA"] = "=" + values["HEAD_NCCL_HCA"]
		values["WORKER_NCCL_IB_HCA"] = "=" + values["WORKER_NCCL_HCA"]
		for _, prefix := range []string{"NCCL", "TP", "GLOO"} {
			values[prefix+"_SOCKET_IFNAME"] = values["HEAD_NCCL_IF"]
			values["WORKER_"+prefix+"_SOCKET_IFNAME"] = values["WORKER_NCCL_IF"]
		}
		values["VLLM_HOST"] = "127.0.0.1"
		if component.BindAddress != "" {
			values["VLLM_HOST"] = component.BindAddress
		}
		values["DSPARK_REVISION_ABLITERATED"] = "48095b3452a17f3e3ae8f77892399389c45de9e1"
	}
	keys := make([]string, 0, len(values))
	for k := range values {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var env strings.Builder
	for _, k := range keys {
		fmt.Fprintf(&env, "%s=%s\n", k, shellQuote(values[k]))
	}
	// App-generated env contains no credential; fixed recipe defaults plus validated overrides.
	_, err = executeHost(ctx, head, []byte(env.String()), "sh", "-c", `set -eu; umask 077; cp "$1/env.sample" "$1/.env.tmp"; cat >> "$1/.env.tmp"; mv "$1/.env.tmp" "$1/.env"`, "sh", dir)
	return dir, err
}

func (c *Controller) runEmbeddedRecipe(ctx context.Context, component Component, action string, token string) error {
	dir, err := c.materializeRecipe(ctx, component)
	if err != nil {
		return err
	}
	head := c.host(component.Host)
	// Read credentials over stdin, not argv or a remote SSH command string.
	command := `set -eu; IFS= read -r HF_TOKEN || :; export HF_TOKEN; exec bash "$1/manage.sh" "$2"`
	var output []byte
	if report := recipeReporter(ctx); report != nil {
		report("실행 패키지 준비 완료 · " + action)
		cmd := hostCommand(ctx, head, "bash", "-c", command, "bash", dir, action)
		cmd.Stdin = bytes.NewBufferString(token + "\n")
		stream := &recipeOutput{token: token, report: report}
		cmd.Stdout, cmd.Stderr = stream, stream
		err = cmd.Run()
		output = []byte(stream.finish())
	} else {
		output, err = executeHost(ctx, head, []byte(token+"\n"), "bash", "-c", command, "bash", dir, action)
	}
	if err != nil {
		detail := string(output)
		if token != "" {
			detail = strings.ReplaceAll(detail, token, "[redacted]")
		}
		return fmt.Errorf("%s %s failed: %s", component.Name, action, detail)
	}
	return nil
}

func (c *Controller) PrepareModel(ctx context.Context, component Component, variant, action, token string, progress ...func(string)) error {
	if len(progress) > 0 && progress[0] != nil {
		ctx = context.WithValue(ctx, recipeProgressKey{}, progress[0])
		progress[0]("실행 상태 확인 중")
	}
	if recipeID(component) == "" {
		return fmt.Errorf("this service has no embedded model preparation recipe")
	}
	if recipeID(component) == "qwen27-exl3" && variant == "official" {
		return fmt.Errorf("Qwen 27B EXL3 currently has only the Uncensored checkpoint configured")
	}
	options := map[string]string{}
	for k, v := range component.RuntimeOptions {
		options[k] = v
	}
	options["MODEL_VARIANT"] = variant
	component.RuntimeOptions = options
	if err := c.begin(Operation{Action: "prepare", ComponentID: component.ID, State: "running", Phase: "모델 준비", StartedAt: time.Now()}); err != nil {
		return err
	}
	defer func() {
		if c.operationRunning() {
			c.finishOperation("complete", "모델 준비 작업 종료")
		}
	}()
	// Preparing a different variant must not overwrite the active recipe env.
	workerRunning := component.isCluster() && c.componentRunning(ctx, Component{Host: component.WorkerHost, Container: component.WorkerContainer})
	if c.componentRunning(ctx, component) || workerRunning {
		c.finishOperation("failed", "실행 중인 모델은 준비할 수 없습니다")
		return fmt.Errorf("%s 중지 후 모델을 준비하세요", component.Name)
	}
	err := c.runEmbeddedRecipe(ctx, component, action, token)
	if err != nil {
		c.finishOperation("failed", err.Error())
	}
	return err
}

func (c *Controller) operationRunning() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.op.State == "running"
}
