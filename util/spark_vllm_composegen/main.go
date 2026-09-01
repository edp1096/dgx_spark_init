package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
	"unicode"

	"gopkg.in/yaml.v3"
)

type Config struct {
	RecipeVersion string         `yaml:"recipe_version,omitempty"`
	Name          string         `yaml:"name,omitempty"`
	Description   string         `yaml:"description,omitempty"`
	Container     string         `yaml:"container,omitempty"`
	BuildArgs     []string       `yaml:"build_args,omitempty"`
	Mods          []string       `yaml:"mods,omitempty"`
	Defaults      map[string]any `yaml:"defaults,omitempty"`
	ClusterOnly   bool           `yaml:"cluster_only,omitempty"`
	SoloOnly      bool           `yaml:"solo_only,omitempty"`
	Command       string         `yaml:"command,omitempty"`
	Mode          string         `yaml:"mode"`
	Service       string         `yaml:"service"`
	ContainerName string         `yaml:"container_name"`
	Image         string         `yaml:"image"`
	Privileged    *bool          `yaml:"privileged,omitempty"`
	NonPrivileged bool           `yaml:"non_privileged,omitempty"`
	MemoryGB      int            `yaml:"memory_gb,omitempty"`
	SwapGB        int            `yaml:"swap_gb,omitempty"`
	PidsLimit     int            `yaml:"pids_limit,omitempty"`
	ShmSizeGB     int            `yaml:"shm_size_gb,omitempty"`
	HFCache       string         `yaml:"hf_cache"`
	Volumes       []string       `yaml:"volumes,omitempty"`
	Env           map[string]any `yaml:"env,omitempty"`
	ETHIF         string         `yaml:"eth_if,omitempty"`
	IBHCA         string         `yaml:"ib_hca,omitempty"`
	HeadIP        string         `yaml:"head_ip,omitempty"`
	NodeIP        string         `yaml:"node_ip,omitempty"`
	NumNodes      int            `yaml:"num_nodes,omitempty"`
	NodeRank      int            `yaml:"node_rank,omitempty"`
	MasterPort    int            `yaml:"master_port,omitempty"`
	RayPort       int            `yaml:"ray_port,omitempty"`
	Model         string         `yaml:"model,omitempty"`
	ServeArgs     []string       `yaml:"serve_args,omitempty"`
	PreCommands   []string       `yaml:"pre_commands,omitempty"`
}

func main() {
	var configPath, outPath, modeOverride, modelOverride, imageOverride string
	flag.StringVar(&configPath, "config", "", "YAML config path")
	flag.StringVar(&outPath, "out", "-", "output path, or - for stdout")
	flag.StringVar(&modeOverride, "mode", "", "override mode: single, ray-head, ray-worker, noray-head, or noray-worker")
	flag.StringVar(&modelOverride, "model", "", "override model id/path")
	flag.StringVar(&imageOverride, "image", "", "override container image")
	flag.Parse()

	if configPath == "" {
		exitf("missing -config")
	}

	cfg, err := readConfig(configPath)
	if err != nil {
		exitf("%v", err)
	}
	if modeOverride != "" {
		cfg.Mode = modeOverride
	}
	if modelOverride != "" {
		cfg.Model = modelOverride
	}
	if imageOverride != "" {
		cfg.Image = imageOverride
	}

	if err := cfg.setDefaults(); err != nil {
		exitf("%v", err)
	}
	text, err := renderCompose(cfg)
	if err != nil {
		exitf("%v", err)
	}
	if err := writeOutput(outPath, text); err != nil {
		exitf("%v", err)
	}
}

func readConfig(path string) (Config, error) {
	f, err := os.Open(path)
	if err != nil {
		return Config{}, err
	}
	defer f.Close()

	var cfg Config
	dec := yaml.NewDecoder(f)
	dec.KnownFields(true)
	if err := dec.Decode(&cfg); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

func (c *Config) setDefaults() error {
	if err := c.applyRecipeFields(); err != nil {
		return err
	}
	c.Mode = strings.ToLower(strings.TrimSpace(c.Mode))
	switch c.Mode {
	case "head":
		c.Mode = "ray-head"
	case "worker":
		c.Mode = "ray-worker"
	}
	switch c.Mode {
	case "single", "ray-head", "ray-worker", "noray-head", "noray-worker":
	case "":
		return errors.New("mode is required: single, ray-head, ray-worker, noray-head, or noray-worker")
	default:
		return fmt.Errorf("unknown mode %q", c.Mode)
	}
	if c.Service == "" {
		switch c.Mode {
		case "ray-head", "noray-head":
			c.Service = "vllm-head"
		case "ray-worker", "noray-worker":
			c.Service = "vllm-worker"
		default:
			c.Service = "vllm"
		}
	}
	if c.ContainerName == "" {
		c.ContainerName = strings.ReplaceAll(c.Service, "_", "-")
	}
	if c.Image == "" {
		if c.Container != "" {
			c.Image = c.Container
		} else {
			c.Image = "${VLLM_IMAGE:-vllm-node-tf5:latest}"
		}
	}
	if c.HFCache == "" {
		c.HFCache = "${HOME}/.cache/huggingface"
	}
	if c.MasterPort == 0 {
		c.MasterPort = 29501
	}
	if c.RayPort == 0 {
		c.RayPort = c.MasterPort
	}
	if c.NumNodes == 0 {
		if c.Mode == "noray-head" || c.Mode == "noray-worker" {
			c.NumNodes = 2
		} else {
			c.NumNodes = 1
		}
	}
	if c.ShmSizeGB == 0 {
		c.ShmSizeGB = 64
	}
	if c.MemoryGB == 0 {
		c.MemoryGB = 110
	}
	if c.SwapGB == 0 {
		c.SwapGB = c.MemoryGB + 10
	}
	if c.PidsLimit == 0 {
		c.PidsLimit = 4096
	}
	if c.Env == nil {
		c.Env = map[string]any{}
	}
	addDefaultEnv(c.Env, "NCCL_IGNORE_CPU_AFFINITY", "1")
	addDefaultEnv(c.Env, "PYTHONUNBUFFERED", "1")
	addDefaultEnv(c.Env, "PYTHONWARNINGS", "ignore")
	addDefaultEnv(c.Env, "RAY_memory_monitor_refresh_ms", "0")
	addDefaultEnv(c.Env, "RAY_num_prestart_python_workers", "0")
	addDefaultEnv(c.Env, "RAY_object_store_memory", "1073741824")
	if c.NodeIP != "" {
		addDefaultEnv(c.Env, "VLLM_HOST_IP", c.NodeIP)
		addDefaultEnv(c.Env, "RAY_NODE_IP_ADDRESS", c.NodeIP)
		addDefaultEnv(c.Env, "RAY_OVERRIDE_NODE_IP_ADDRESS", c.NodeIP)
	}
	if c.ETHIF != "" {
		addDefaultEnv(c.Env, "MN_IF_NAME", c.ETHIF)
		addDefaultEnv(c.Env, "UCX_NET_DEVICES", c.ETHIF)
		addDefaultEnv(c.Env, "NCCL_SOCKET_IFNAME", c.ETHIF)
		addDefaultEnv(c.Env, "OMPI_MCA_btl_tcp_if_include", c.ETHIF)
		addDefaultEnv(c.Env, "GLOO_SOCKET_IFNAME", c.ETHIF)
		addDefaultEnv(c.Env, "TP_SOCKET_IFNAME", c.ETHIF)
	}
	if c.IBHCA != "" {
		addDefaultEnv(c.Env, "NCCL_IB_HCA", c.IBHCA)
		addDefaultEnv(c.Env, "NCCL_IB_DISABLE", "0")
	}
	if (c.Mode == "ray-head" || c.Mode == "noray-head") && c.HeadIP == "" {
		c.HeadIP = c.NodeIP
	}
	if (c.Mode == "ray-head" || c.Mode == "noray-head") && c.HeadIP == "" {
		return fmt.Errorf("head_ip or node_ip is required for %s mode", c.Mode)
	}
	if (c.Mode == "ray-worker" || c.Mode == "noray-worker") && c.HeadIP == "" {
		return fmt.Errorf("head_ip is required for %s mode", c.Mode)
	}
	if (c.Mode == "ray-worker" || c.Mode == "noray-worker") && c.NodeIP == "" {
		return fmt.Errorf("node_ip is required for %s mode", c.Mode)
	}
	if c.Mode == "noray-worker" && c.NodeRank == 0 {
		return errors.New("node_rank must be greater than 0 for noray-worker mode")
	}
	if (c.Mode == "noray-head" || c.Mode == "noray-worker") && c.NumNodes <= c.NodeRank {
		return fmt.Errorf("num_nodes (%d) must be greater than node_rank (%d)", c.NumNodes, c.NodeRank)
	}
	if c.ClusterOnly && c.Mode == "single" {
		return errors.New("recipe is cluster_only and cannot be rendered in single mode")
	}
	if c.SoloOnly && c.Mode != "single" {
		return fmt.Errorf("recipe is solo_only and cannot be rendered in %s mode", c.Mode)
	}
	if c.Mode != "ray-worker" && c.Model == "" {
		return fmt.Errorf("model is required for %s mode", c.Mode)
	}
	return nil
}

func (c *Config) applyRecipeFields() error {
	if c.Command == "" {
		return nil
	}
	command, err := substituteDefaults(c.Command, c.Defaults)
	if err != nil {
		return err
	}
	model, args, err := parseVLLMServe(command)
	if err != nil {
		return err
	}
	if c.Model == "" {
		c.Model = model
	}
	if len(c.ServeArgs) == 0 {
		c.ServeArgs = args
	}
	return nil
}

func addDefaultEnv(env map[string]any, key, value string) {
	if _, ok := env[key]; !ok {
		env[key] = value
	}
}

func substituteDefaults(command string, defaults map[string]any) (string, error) {
	out := command
	for key, value := range defaults {
		out = strings.ReplaceAll(out, "{"+key+"}", fmt.Sprint(value))
	}
	if placeholder := firstPlaceholder(out); placeholder != "" {
		return "", fmt.Errorf("command still contains unsubstituted placeholder %s: %q", placeholder, out)
	}
	return out, nil
}

func firstPlaceholder(s string) string {
	for i := 0; i < len(s); i++ {
		if s[i] != '{' {
			continue
		}
		j := strings.IndexByte(s[i+1:], '}')
		if j < 0 {
			continue
		}
		name := s[i+1 : i+1+j]
		if isPlaceholderName(name) {
			return "{" + name + "}"
		}
	}
	return ""
}

func isPlaceholderName(name string) bool {
	if name == "" {
		return false
	}
	for i, r := range name {
		ok := r == '_' || (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') || (i > 0 && r >= '0' && r <= '9')
		if !ok {
			return false
		}
	}
	return true
}

func parseVLLMServe(command string) (string, []string, error) {
	normalized := strings.ReplaceAll(command, "\\\n", " ")
	normalized = strings.ReplaceAll(normalized, "\n", " ")
	tokens, err := shellFields(normalized)
	if err != nil {
		return "", nil, err
	}
	for i := 0; i+2 < len(tokens); i++ {
		if tokens[i] == "vllm" && tokens[i+1] == "serve" {
			return tokens[i+2], tokens[i+3:], nil
		}
	}
	return "", nil, fmt.Errorf("command does not contain 'vllm serve <model>': %q", command)
}

func shellFields(s string) ([]string, error) {
	var fields []string
	var b strings.Builder
	var quote rune
	escaped := false
	inField := false

	flush := func() {
		if inField {
			fields = append(fields, b.String())
			b.Reset()
			inField = false
		}
	}

	for _, r := range s {
		if escaped {
			b.WriteRune(r)
			inField = true
			escaped = false
			continue
		}
		if r == '\\' {
			escaped = true
			inField = true
			continue
		}
		if quote != 0 {
			if r == quote {
				quote = 0
			} else {
				b.WriteRune(r)
				inField = true
			}
			continue
		}
		if r == '\'' || r == '"' {
			quote = r
			inField = true
			continue
		}
		if unicode.IsSpace(r) {
			flush()
			continue
		}
		b.WriteRune(r)
		inField = true
	}
	if escaped {
		b.WriteRune('\\')
	}
	if quote != 0 {
		return nil, errors.New("unterminated quote in command")
	}
	flush()
	return fields, nil
}

func renderCompose(c Config) (string, error) {
	var b strings.Builder
	b.WriteString("services:\n")
	fmt.Fprintf(&b, "  %s:\n", c.Service)
	fmt.Fprintf(&b, "    image: %s\n", yamlString(c.Image))
	fmt.Fprintf(&b, "    container_name: %s\n", yamlString(c.ContainerName))
	b.WriteString("    network_mode: \"host\"\n")
	if !c.NonPrivileged {
		b.WriteString("    ipc: \"host\"\n")
		if c.Privileged == nil || *c.Privileged {
			b.WriteString("    privileged: true\n")
		}
	} else {
		b.WriteString("    cap_add:\n")
		b.WriteString("      - IPC_LOCK\n")
		fmt.Fprintf(&b, "    shm_size: \"%dg\"\n", c.ShmSizeGB)
		b.WriteString("    devices:\n")
		b.WriteString("      - \"/dev/infiniband:/dev/infiniband\"\n")
		fmt.Fprintf(&b, "    mem_limit: \"%dg\"\n", c.MemoryGB)
		fmt.Fprintf(&b, "    memswap_limit: \"%dg\"\n", c.SwapGB)
		fmt.Fprintf(&b, "    pids_limit: %d\n", c.PidsLimit)
	}

	writeEnvironment(&b, c.Env)
	writeVolumes(&b, c)
	writeGPUReservation(&b)

	b.WriteString("    entrypoint:\n")
	b.WriteString("      - bash\n")
	b.WriteString("      - -lc\n")
	b.WriteString("    command: |-\n")
	for _, line := range commandLines(c) {
		fmt.Fprintf(&b, "      %s\n", line)
	}
	return b.String(), nil
}

func writeEnvironment(b *strings.Builder, env map[string]any) {
	if len(env) == 0 {
		return
	}
	keys := make([]string, 0, len(env))
	for k := range env {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	b.WriteString("    environment:\n")
	for _, k := range keys {
		fmt.Fprintf(b, "      - %s=%s\n", k, fmt.Sprint(env[k]))
	}
}

func writeVolumes(b *strings.Builder, c Config) {
	volumes := []string{fmt.Sprintf("%s:/root/.cache/huggingface", c.HFCache)}
	volumes = append(volumes, c.Volumes...)
	if len(volumes) == 0 {
		return
	}
	b.WriteString("    volumes:\n")
	for _, v := range volumes {
		fmt.Fprintf(b, "      - %s\n", yamlString(v))
	}
}

func writeGPUReservation(b *strings.Builder) {
	b.WriteString("    deploy:\n")
	b.WriteString("      resources:\n")
	b.WriteString("        reservations:\n")
	b.WriteString("          devices:\n")
	b.WriteString("            - driver: nvidia\n")
	b.WriteString("              count: all\n")
	b.WriteString("              capabilities: [gpu]\n")
}

func commandLines(c Config) []string {
	lines := []string{"set -e"}
	lines = append(lines, c.PreCommands...)

	switch c.Mode {
	case "ray-worker":
		lines = append(lines, fmt.Sprintf(
			"exec ray start --address=%s:%d --node-ip-address=%s --object-store-memory 1073741824 --num-cpus 2 --disable-usage-stats --block",
			c.HeadIP, c.RayPort, c.NodeIP,
		))
	case "ray-head":
		lines = append(lines, fmt.Sprintf(
			"ray start --head --port=%d --node-ip-address=%s --object-store-memory 1073741824 --num-cpus 2 --include-dashboard=false --disable-usage-stats",
			c.RayPort, c.HeadIP,
		))
		lines = append(lines, "sleep 5")
		lines = append(lines, renderServe("exec vllm serve", c.Model, ensureRayBackend(c.ServeArgs)))
	case "noray-head":
		lines = append(lines, renderServe("exec vllm serve", c.Model, noRayArgs(c, false)))
	case "noray-worker":
		lines = append(lines, renderServe("exec vllm serve", c.Model, noRayArgs(c, true)))
	case "single":
		lines = append(lines, renderServe("exec vllm serve", c.Model, stripFlagValue(c.ServeArgs, "--distributed-executor-backend")))
	}
	return lines
}

func ensureRayBackend(args []string) []string {
	for i := 0; i < len(args); i++ {
		if args[i] == "--distributed-executor-backend" {
			return args
		}
		if strings.HasPrefix(args[i], "--distributed-executor-backend=") {
			return args
		}
	}
	out := append([]string{}, args...)
	return append(out, "--distributed-executor-backend", "ray")
}

func noRayArgs(c Config, headless bool) []string {
	args := stripFlagValue(c.ServeArgs, "--distributed-executor-backend")
	args = append(args,
		"--nnodes", strconv.Itoa(c.NumNodes),
		"--node-rank", strconv.Itoa(c.NodeRank),
		"--master-addr", c.HeadIP,
		"--master-port", strconv.Itoa(c.MasterPort),
	)
	if headless {
		args = append(args, "--headless")
	}
	return args
}

func stripFlagValue(args []string, flag string) []string {
	out := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if arg == flag {
			if i+1 < len(args) {
				i++
			}
			continue
		}
		if strings.HasPrefix(arg, flag+"=") {
			continue
		}
		out = append(out, arg)
	}
	return out
}

func renderServe(prefix, model string, args []string) string {
	parts := []string{prefix, shellWord(model)}
	for _, arg := range args {
		parts = append(parts, shellWord(arg))
	}
	return strings.Join(parts, " ")
}

func shellWord(s string) string {
	if s == "" {
		return "''"
	}
	if strings.IndexFunc(s, func(r rune) bool {
		return !(r == '-' || r == '_' || r == '.' || r == '/' || r == ':' || r == '=' || r == ',' ||
			(r >= '0' && r <= '9') || (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z'))
	}) == -1 {
		return s
	}
	return strconv.Quote(s)
}

func yamlString(s string) string {
	return strconv.Quote(s)
}

func writeOutput(path, text string) error {
	if path == "-" {
		_, err := io.WriteString(os.Stdout, text)
		return err
	}
	return os.WriteFile(path, []byte(text), 0644)
}

func exitf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "composegen: "+format+"\n", args...)
	os.Exit(1)
}
