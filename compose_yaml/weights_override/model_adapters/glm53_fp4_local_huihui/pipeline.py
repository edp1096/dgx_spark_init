"""Resume the worker-side CPU analysis/build after verified inputs arrive."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from download import atomic


NAME = "Huihui-GLM-5.3-Flash-abliterated-FP4-Local"


def prefix(row):
    fields = row["name"].split(".")
    projection = {"attn_output": "self_attn.o_proj",
                  "ffn_down_shexp": "mlp.shared_experts.down_proj"}[fields[2]]
    return "model.language_model.layers." + fields[1] + "." + projection


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def run(root):
    lock = (root / "pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    base = Path.home() / ".cache/huggingface/nvidia/GLM-5.3-Flash-FP4-Local"
    output = Path.home() / ".cache/huggingface/edp1096" / NAME
    source = root / "src"
    image = subprocess.check_output([
        "docker", "image", "inspect", "--format", "{{.Id}}",
        "dgx-sglang-qwen38-qad:sm121-v5-memory",
    ], text=True).strip()
    atomic(root / "pipeline-runtime.json", {
        "conversion_image": image, "pid": os.getpid(), "source": str(base),
        "destination": str(output), "started": time.time(),
    })

    def status(phase, **extra):
        atomic(root / "pipeline-status.json", {
            "phase": phase, "time": time.time(), "pid": os.getpid(), **extra,
        })

    def execute(mode):
        memory = "8g" if mode == "build" else "4g"
        command = [
            "docker", "run", "--rm", "--name", "glm53-fp4-local-" + mode + "-20261018",
            "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=1g",
            "--memory", memory, "--memory-swap", memory, "--cpus", "2",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "-e", "HOME=/tmp", "-e", "HF_HOME=/tmp/hf", "-e", "HF_HUB_OFFLINE=1",
            "-e", "TRANSFORMERS_OFFLINE=1", "-v", str(source) + ":/work:ro",
            "-v", str(root) + ":/job", "-v", str(base) + ":/model:ro",
            "-v", str(root / "ranges") + ":/job/ranges:ro",
            "-v", str(root / "metadata") + ":/job/metadata:ro",
        ]
        if mode == "build":
            (root / "outputs").mkdir(exist_ok=True)
            command += ["-v", str(root / "outputs") + ":/candidate"]
        command += [
            "--entrypoint", "python3", image, "-B", "-u",
            "/work/model_adapters/glm53_fp4_local_huihui/build.py",
            "--root", "/job", "--base", "/model", "--mode", mode,
        ]
        if mode == "build":
            command += ["--output", "/candidate/" + NAME]
        status(mode + "_running")
        with (root / (mode + ".log")).open("ab", buffering=0) as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{mode} container exited with code {result.returncode}")

    def wait_inputs():
        while True:
            tests = read(root / "test-numeric-result.json")
            if tests is not None and not tests["passed"]:
                raise ValueError("Numerical unit tests failed")
            delta = read(root / "delta-download-receipt.json")
            receipt = read(root / "base-download-receipt.json")
            index_file = base / "model.safetensors.index.json"
            if tests and delta and receipt and index_file.exists():
                audit = read(root / "metadata/gguf-audit.json")
                index = read(index_file)["weight_map"]
                wanted = {prefix(r) + suffix for r in audit["tensors"] if r["changed"]
                          for suffix in (".weight", ".weight_scale")}
                required = {index[n] for n in wanted}
                verified = {f["rfilename"] for f in receipt["files"]}
                missing = sorted(required - verified)
                if delta["status"] == "verified" and not missing:
                    return
                status("waiting_projection_shards", missing=missing)
            else:
                status("waiting_input_metadata")
            time.sleep(15)

    try:
        if output.exists():
            state = read(output / "transfer-manifest.json")
            if state is None or state["status"] != "candidate_verified":
                raise ValueError("An unexpected existing output must not be overwritten")
            status("candidate_verified_runtime_pending", output=str(output))
            return
        wait_inputs()
        dry = read(root / "numeric-dry-run.json")
        if dry is None:
            execute("analyze")
        elif dry["status"] != "verified":
            raise ValueError("Existing numerical dry run is not verified")
        while True:
            receipt = read(root / "base-download-receipt.json")
            if receipt and receipt["status"] == "verified":
                break
            status("waiting_full_source_download")
            time.sleep(15)
        execute("build")
        staged = root / "outputs" / NAME
        state = read(staged / "transfer-manifest.json")
        if not state or state["status"] != "candidate_verified":
            raise ValueError("Build did not produce a verified candidate")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError("Final output appeared during conversion")
        staged.rename(output)
        result = read(root / "conversion-result.json")
        result["container_output"] = result["output"]
        result["output"] = str(output)
        atomic(root / "conversion-result.json", result)
        status("candidate_verified_runtime_pending", output=str(output))
    except Exception as exc:
        status("failed", error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root.resolve(strict=True))
