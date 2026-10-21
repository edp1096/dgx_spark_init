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
	case "qwen38-cluster":
		return "qwen38-tp2"
	case "glm53-cluster":
		return "glm53"
	case "glm53-fp4-cluster":
		return "glm53-fp4"
	case "ds41-cluster":
		return "ds41"
	}
	return ""
}

var recipeOptionNames = map[string]bool{
	"DSV41_PRELOAD_COUNT": true, "MODEL_VARIANT": true, "HEAD_RAIL_IP": true, "WORKER_RAIL_IP": true,
	"HEAD_NCCL_IF": true, "WORKER_NCCL_IF": true, "HEAD_NCCL_HCA": true, "WORKER_NCCL_HCA": true,
	"NCCL_SUBNET": true, "MASTER_PORT": true, "MAX_MODEL_LEN": true, "MAX_NUM_SEQS": true,
	"GPU_MEMORY_UTILIZATION": true, "GPU_MEMORY_UTILIZATION_TEXT": true,
	"DRAFT_VOCAB": true, "KV_CACHE_MEMORY": true, "CACHE_RAM_MIB": true, "MTP_TOKENS": true, "DFLASH_TOKENS": true,
	"IMAGE_RESIDENCY": true,
}

func validateRecipeOptions(c Component) error {
	if c.ComposeAsset == "compose.qwen-fp4-local.yaml" {
		for key, value := range c.RuntimeOptions {
			if key != "MODEL_VARIANT" || value != "abliterated" {
				return fmt.Errorf("FP4-Local uses the fixed 1M context, MTP 4 and abliterated checkpoint; runtime option overrides are not supported")
			}
		}
	}

	if c.Controller == "glm53-fp4-cluster" {
		for key, value := range c.RuntimeOptions {
			switch key {
			case "MODEL_VARIANT", "HEAD_RAIL_IP", "WORKER_RAIL_IP", "HEAD_NCCL_IF", "WORKER_NCCL_IF", "HEAD_NCCL_HCA", "WORKER_NCCL_HCA", "NCCL_SUBNET", "MASTER_PORT":
			case "MAX_MODEL_LEN":
				if value != "1048576" {
					return fmt.Errorf("GLM FP4-Local requires the qualified 1M context")
				}
			case "MAX_NUM_SEQS":
				if value != "1" {
					return fmt.Errorf("GLM FP4-Local requires one active request")
				}
			case "MTP_TOKENS":
				if value != "1" && value != "5" {
					return fmt.Errorf("GLM FP4-Local MTP must be 1 or 5")
				}
			default:
				return fmt.Errorf("GLM FP4-Local uses a fixed qualified profile; unsupported option %s", key)
			}
		}
	}
	for k, v := range c.RuntimeOptions {
		if c.Controller == "ds41-cluster" {
			switch k {
			case "MODEL_VARIANT", "MAX_MODEL_LEN", "MASTER_PORT", "HEAD_RAIL_IP", "WORKER_RAIL_IP", "HEAD_NCCL_IF", "WORKER_NCCL_IF", "HEAD_NCCL_HCA", "WORKER_NCCL_HCA", "NCCL_SUBNET", "MTP_TOKENS":
			default:
				return fmt.Errorf("TensorFold DS41 uses a fixed qualified serving profile; unsupported override %s", k)
			}
			if k == "MTP_TOKENS" && v != "auto" && v != "1" && v != "2" && v != "3" && v != "4" && v != "5" {
				return fmt.Errorf("TensorFold MTP_TOKENS must be auto or a fixed depth 1..5")
			}
		}
		if k == "IMAGE_RESIDENCY" && (c.ComposeAsset != "compose.qwen-image21.yaml" || (v != "legacy" && v != "dit") || (v == "dit" && !c.KeepResident)) {
			return fmt.Errorf("%s: IMAGE_RESIDENCY requires the single Qwen Image service; dit requires KeepResident", c.ID)
		}
		if c.ComposeAsset == "compose.flash-next.yaml" && k == "MTP_TOKENS" && v != "0" && v != "3" {
			return fmt.Errorf("QAD MTP_TOKENS must be 0 or 3")
		}
		if k == "DRAFT_VOCAB" && ((c.ComposeAsset != "compose.flash-next.yaml") || (v != "off" && v != "ko64k")) {
			return fmt.Errorf("%s: DRAFT_VOCAB requires a supported model and off or ko64k", c.ID)
		}
		if c.Controller == "qwen38-cluster" && ((k == "MAX_MODEL_LEN" && v != "262144" && v != "524288" && v != "1048576") || (k == "MODEL_VARIANT" && v != "abliterated")) {
			return fmt.Errorf("Qwen TP2 requires the tested abliteration checkpoint and 256K/512K/1M context")
		}
		if c.Controller == "glm53-cluster" && k == "MTP_TOKENS" && v != "0" {
			return fmt.Errorf("GLM MTP is not qualified for this TP2/1M recipe; MTP_TOKENS must be 0")
		}
		if c.Controller == "glm53-cluster" && k == "DFLASH_TOKENS" && v != "0" && v != "5" {
			return fmt.Errorf("GLM DFlash2 supports DFLASH_TOKENS=0 or 5 in this profile")
		}
		if c.Controller == "glm53-cluster" && c.ProgressKind == "sglang" {
			if k == "KV_CACHE_MEMORY" || k == "GPU_MEMORY_UTILIZATION" {
				return fmt.Errorf("GLM SGLang does not use the vLLM option %s", k)
			}
			if k == "MAX_NUM_SEQS" && v != "1" {
				return fmt.Errorf("GLM SGLang requires MAX_NUM_SEQS=1")
			}
		}
		if !recipeOptionNames[k] || strings.ContainsAny(v, "\x00\r\n") {
			return fmt.Errorf("%s: unsupported runtime option %s", c.ID, k)
		}
		if k == "DSV41_PRELOAD_COUNT" {
			n, err := strconv.Atoi(v)
			if c.Controller != "ds41-cluster" || err != nil || n < 0 || n > 224 || strconv.Itoa(n) != v {
				return fmt.Errorf("%s: DSV41_PRELOAD_COUNT requires DS41 and an integer 0..224", c.ID)
			}
		}
		if k == "MODEL_VARIANT" && v != "official" && v != "abliterated" && !(c.ComposeAsset == "compose.flash-next.yaml" && (v == "huihui_lil" || v == "radixark")) {
			return fmt.Errorf("invalid model variant")
		}
		if c.Controller == "ds41-cluster" && k == "MAX_MODEL_LEN" && v != "1048576" {
			return fmt.Errorf("TensorFold DS41 requires the qualified 1M profile; legacy streaming preload options are unsupported")
		}
	}
	return nil
}

// Each recipe is embedded. Runtime never reads or invokes the workspace checkout.
func (c *Controller) materializeRecipe(ctx context.Context, component Component) (string, error) {
	if err := c.CheckAutoCluster(component); err != nil {
		return "", err
	}
	id := recipeID(component)
	data, err := assets.ReadFile("assets/recipes/" + id + ".tar.gz")
	if err != nil {
		return "", err
	}
	head := c.host(component.Host)
	dataDir, cache, err := c.runtimeHostPaths(component)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	dir := filepath.Join(dataDir, "runtime", "recipes", fmt.Sprintf("%s-%x", id, sum[:8]))
	if _, err = executeHost(ctx, head, data, "sh", "-c", `set -eu; umask 077; mkdir -p "$1"; tar -xzf - -C "$1"`, "sh", dir); err != nil {
		return "", err
	}
	values := map[string]string{
		"MODEL_KIND": id, "HF_CACHE": cache, "API_PORT": strconv.Itoa(component.Port), "VLLM_PORT": strconv.Itoa(component.Port),
		"MODEL_HOST_PATH":   cache,
		"SERVED_MODEL_NAME": component.Model,
	}
	if component.Port == 0 {
		values["API_PORT"] = "8000"
		values["VLLM_PORT"] = "8888"
	}
	if id == "glm53" {
		values["MODEL_HOST_PATH"] = filepath.Join(cache, "nvidia", "GLM-5.3-Flash-NVFP4")
		values["GLM53_CACHE_PATH"] = filepath.Join(filepath.Dir(cache), "glm53-nvfp4")
		values["VLLM_BIND"] = component.BindAddress
		if values["VLLM_BIND"] == "" {
			values["VLLM_BIND"] = "127.0.0.1"
		}
	}
	if id == "glm53-fp4" {
		values["GLM53_HEAD_CONTAINER"] = component.Container
		values["GLM53_WORKER_CONTAINER"] = component.WorkerContainer
		values["MODEL_HOST_PATH"] = filepath.Join(cache, "nvidia", "GLM-5.3-Flash-FP4-Local")
		values["GLM53_CACHE_PATH"] = filepath.Join(filepath.Dir(cache), "glm53-fp4-local")
		values["VLLM_BIND"] = component.BindAddress
		if values["VLLM_BIND"] == "" {
			values["VLLM_BIND"] = "127.0.0.1"
		}
	}
	values["CONTEXT_SIZE"] = component.RuntimeOptions["MAX_MODEL_LEN"]
	if values["CONTEXT_SIZE"] == "" {
		values["CONTEXT_SIZE"] = "131072"
	}
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
		if id == "ds41" || id == "glm53-fp4" {
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
	if id == "ds41" {
		values["HEAD_CONTAINER"] = component.Container
		values["WORKER_CONTAINER"] = component.WorkerContainer
		values["VLLM_HOST"] = component.BindAddress
		if values["VLLM_HOST"] == "" {
			values["VLLM_HOST"] = "127.0.0.1"
		}
		if values["MAX_MODEL_LEN"] == "" {
			values["MAX_MODEL_LEN"] = "1048576"
		}
	}
	if id == "qwen38-tp2" {
		values["HEAD_CONTAINER"] = component.Container
		values["WORKER_CONTAINER"] = component.WorkerContainer
		values["QWEN_TP2_HEAD_CACHE"] = filepath.Join(dataDir, "cache", "sglang-flash-next-tp2")
		values["QWEN_TP2_WORKER_CACHE"] = filepath.Join(c.host(component.WorkerHost).DataDir, "cache", "sglang-flash-next-tp2")
		values["QWEN_TP2_BIND"] = component.BindAddress
		if values["QWEN_TP2_BIND"] == "" {
			values["QWEN_TP2_BIND"] = "127.0.0.1"
		}
		if values["MAX_MODEL_LEN"] == "" {
			values["MAX_MODEL_LEN"] = "1048576"
		}
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
	if action != "model" && action != "setup" {
		return fmt.Errorf("invalid preparation action")
	}
	if err := ValidateModelPreparationVariant(component, variant); err != nil {
		return err
	}
	if len(progress) > 0 && progress[0] != nil {
		ctx = context.WithValue(ctx, recipeProgressKey{}, progress[0])
		progress[0]("실행 상태 확인 중")
	}
	if recipeID(component) == "" && embeddedBuildAsset(component.ComposeAsset) == "" {
		return fmt.Errorf("this service has no embedded model preparation recipe")
	}
	options := map[string]string{}
	for k, v := range component.RuntimeOptions {
		options[k] = v
	}
	options["MODEL_VARIANT"] = variant
	component.RuntimeOptions = options
	if err := validateRecipeOptions(component); err != nil {
		return err
	}
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
	var err error
	if recipeID(component) == "" {
		err = c.prepareSingleService(ctx, component, action, token)
	} else {
		err = c.runEmbeddedRecipe(ctx, component, action, token)
	}
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
