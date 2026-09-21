package orchestrator

import (
	"bytes"
	"context"
	"fmt"
	"strings"
)

func (c *Controller) prepareQwenQAD(ctx context.Context, component Component, action, token string) error {
	if action != "model" && action != "setup" {
		return fmt.Errorf("invalid QAD preparation action")
	}
	repo, revision, err := QwenQADCheckpoint(component.qwenQADVariant())
	if err != nil {
		return err
	}
	if err := c.prepareOrStartComponent(ctx, component, true); err != nil {
		return err
	}
	_, cache, err := c.runtimeHostPaths(component)
	if err != nil {
		return err
	}
	// Download on the component's configured host, with no GPU and no credential
	// in command arguments or persisted Compose. Snapshot revision is immutable.
	script := `import sys
from huggingface_hub import snapshot_download
token = sys.stdin.readline().strip()
print("QAD download: " + sys.argv[1] + " @ " + sys.argv[2], flush=True)
path = snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], cache_dir="/hf/hub", token=token or False, max_workers=4)
print("QAD ready: " + path, flush=True)
`
	if component.qwenQADVariant() == "huihui_lil" {
		script = qwenQADLocalVerificationScript()
	}
	cmd := hostCommand(ctx, c.host(component.Host), "docker", "run", "--rm", "-i", "-e", "HF_HUB_OFFLINE=0", "-e", "HF_HOME=/hf", "-v", cache+":/hf", "--entrypoint", "python3", "dgx-sglang-qwen38-qad:sm121-v5", "-u", "-c", script, repo, revision)
	cmd.Stdin = bytes.NewBufferString(strings.TrimSpace(token) + "\n")
	report := recipeReporter(ctx)
	if report == nil {
		report = func(string) {}
	}
	output := &recipeOutput{token: strings.TrimSpace(token), report: report}
	cmd.Stdout, cmd.Stderr = output, output
	err = cmd.Run()
	detail := output.finish()
	if err != nil {
		return fmt.Errorf("QAD model preparation failed: %w: %s", err, detail)
	}
	return nil
}

func qwenQADLocalVerificationScript() string {
	return `import hashlib,json,sys
from pathlib import Path
root = Path("/hf") / sys.argv[1]
if not (root / "transfer-manifest.json").is_file() or not (root / "runtime-qualification.json").is_file():
    raise RuntimeError("Local Huihui/LIL model is missing; complete weights_override conversion and runtime validation on this server first")
manifest = json.loads((root / "transfer-manifest.json").read_text())
qualification = json.loads((root / "runtime-qualification.json").read_text())
if manifest.get("status") != "candidate_verified" or qualification.get("status") != "passed":
    raise RuntimeError("Local Huihui/LIL model has not passed conversion and runtime validation")
if qualification.get("manifest_sha256") != hashlib.sha256((root / "transfer-manifest.json").read_bytes()).hexdigest():
    raise RuntimeError("Local Huihui/LIL qualification does not match the converted model")
index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
if not index or set(index.values()) != set(manifest["output_shard_hashes"]):
    raise RuntimeError("Local Huihui/LIL shard manifest is incomplete")
for name, expected in {**manifest["output_shard_hashes"], **manifest["metadata_sha256"]}.items():
    if Path(name).name != name:
        raise RuntimeError("Invalid shard name")
    with (root / name).open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
            raise RuntimeError("Local Huihui/LIL shard checksum mismatch: " + name)
print("Local Huihui/LIL verified: " + str(root), flush=True)
`
}
