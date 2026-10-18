"""Exercise a separately launched candidate server; never changes Talk settings."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import requests

from download import atomic, sha256


MODEL = "glm53-huihui-fp4-local-check"
SANDBOX_IMAGE = "eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595"
COMMON = {
    "model": MODEL, "temperature": 0, "top_p": 1, "seed": 20261018,
    "presence_penalty": 0, "frequency_penalty": 0,
    "chat_template_kwargs": {"reasoning_effort": "low", "clear_thinking": False},
}
SYSTEM = "Follow the user's instructions accurately."


def json_content(content):
    text = content.strip()
    if text.startswith("```json\n") and text.endswith("\n```"):
        text = text[8:-4]
    elif text.startswith("```\n") and text.endswith("\n```"):
        text = text[4:-4]
    return json.loads(text)


def check_code(content):
    source = content.strip()
    if source.startswith("```python\n") and source.endswith("\n```"):
        source = source[10:-4]
    checker = """import signal, pathlib
signal.alarm(5)
namespace = {}
source = pathlib.Path('/input/candidate.py').read_text()
exec(compile(source, '<candidate>', 'exec'), namespace)
function = namespace['sum_even']
for values, expected in [([], 0), ([1, 3, 5], 0), ([-2, 1, 4, 5], 2), ([0, 2, 2], 4), ([-2, -2, 6], 2)]:
    before = values.copy()
    assert function(values) == expected
    assert values == before
print('5 behavior checks passed')
"""
    with tempfile.TemporaryDirectory(prefix="glm53-code-check-") as directory:
        root = Path(directory)
        root.chmod(0o755)
        (root / "candidate.py").write_text(source)
        (root / "checker.py").write_text(checker)
        container_name = f"glm53-fp4-local-code-check-{os.getpid()}-{time.time_ns()}"
        command = [
            "docker", "run", "--rm", "--name", container_name, "--network", "none", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "32", "--cpus", "1", "--memory", "128m",
            "--memory-swap", "128m", "--user", "65534:65534",
            "-v", str(root) + ":/input:ro", "--entrypoint", "python3",
            SANDBOX_IMAGE, "-B", "/input/checker.py",
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        finally:
            subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=10)
    return {"passed": result.returncode == 0 and result.stdout.strip() == "5 behavior checks passed", "stdout": result.stdout[-1500:],
            "stderr": result.stderr[-1500:], "exit_code": result.returncode}


def run(args):
    args.root.mkdir(parents=True, exist_ok=True)
    fixture = "data:image/png;base64," + base64.b64encode(args.fixture.read_bytes()).decode()
    long_input = args.long_input.read_text()
    cases = [
        {"id": "arithmetic", "prompt": "17 * 19의 값을 정수 하나로만 답하세요.", "max_tokens": 512},
        {"id": "korean", "prompt": "비 오는 날 서점에서 만난 두 사람의 짧은 이야기를 한국어 세 문장으로 쓰세요.", "max_tokens": 1024},
        {"id": "code", "prompt": "정수 리스트 values에서 짝수만 합산하는 Python 함수 sum_even(values)를 작성하세요. 음수와 빈 리스트를 처리하고 입력을 바꾸지 마세요. 코드만 출력하세요.", "max_tokens": 1024},
        {"id": "structured_json", "prompt": 'Return the JSON object {"ok":true,"count":3}.', "max_tokens": 512,
         "response_format": {"type": "json_schema", "json_schema": {
             "name": "smoke", "strict": True, "schema": {"type": "object", "properties": {
                 "ok": {"type": "boolean"}, "count": {"type": "integer"}},
                 "required": ["ok", "count"], "additionalProperties": False}}}},
        {"id": "tool", "prompt": "Call get_weather exactly once with city='Seoul' and country='KR'.",
         "max_tokens": 768, "tools": [{"type": "function", "function": {
             "name": "get_weather", "description": "Get the current weather for a city.",
             "parameters": {"type": "object", "properties": {
                 "city": {"type": "string"}, "country": {"type": "string", "enum": ["KR"]}},
                 "required": ["city", "country"], "additionalProperties": False}}}],
         "tool_choice": "auto"},
        {"id": "vision", "prompt": [
            {"type": "image_url", "image_url": {"url": fixture}},
            {"type": "text", "text": 'Count the colored squares. Return a JSON object with keys red, green, blue, total; all values must be integers.'}],
         "max_tokens": 1024},
        {"id": "long_retrieval", "prompt": long_input, "max_tokens": 1536},
        {"id": "post_long", "prompt": "17 + 25의 값을 정수 하나로만 답하세요.", "max_tokens": 512},
    ]
    protocol = {
        "common": COMMON, "system": SYSTEM, "case_ids": [c["id"] for c in cases],
        "max_tokens": {c["id"]: c["max_tokens"] for c in cases},
        "fixture_sha256": sha256(args.fixture), "long_input_sha256": sha256(args.long_input),
        "long_input_source": str(args.long_input), "created": time.time(),
        "scope": "Candidate runtime smoke checks; not a general quality benchmark or proof of behavioral equivalence",
    }
    atomic(args.root / "runtime-protocol.json", protocol)
    report = {"status": "running", "model": MODEL, "started": time.time(), "cases": []}
    models = requests.get(args.endpoint + "/v1/models", timeout=20)
    models.raise_for_status()
    report["models"] = models.json()
    for case in cases:
        name = case["id"]
        atomic(args.root / "runtime-progress.json", {"phase": "testing", "case": name, "time": time.time()})
        request = {**COMMON, **{k: v for k, v in case.items() if k not in ("id", "prompt")},
                   "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": case["prompt"]}]}
        result = {"id": name, "request_sha256": hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()}
        started = time.monotonic()
        try:
            response = requests.post(args.endpoint + "/v1/chat/completions", json=request, timeout=(15, 1800))
            response.raise_for_status()
            raw = response.json()
            result.update(response=raw, seconds=time.monotonic() - started)
            choice = raw["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            complete = choice["finish_reason"] in ("stop", "tool_calls")
            checks = {"completed": complete}
            if name in ("arithmetic", "post_long"):
                checks["answer"] = content.strip() == ("323" if name == "arithmetic" else "42")
            elif name == "korean":
                checks["korean_nonempty"] = len(content.strip()) > 30 and sum("가" <= c <= "힣" for c in content) > 20
            elif name == "code":
                result["code_check"] = check_code(content)
                checks["behavior"] = result["code_check"]["passed"]
            elif name == "structured_json":
                parsed = json.loads(content)
                checks["exact_json"] = parsed == {"ok": True, "count": 3} and type(parsed["ok"]) is bool and type(parsed["count"]) is int
            elif name == "tool":
                calls = message.get("tool_calls") or []
                checks["single_call"] = len(calls) == 1
                checks["correct_call"] = bool(calls) and calls[0]["function"]["name"] == "get_weather" and json.loads(calls[0]["function"]["arguments"]) == {"city": "Seoul", "country": "KR"}
            elif name == "vision":
                checks["counts"] = json_content(content) == {"red": 2, "green": 1, "blue": 3, "total": 6}
            elif name == "long_retrieval":
                parsed = json_content(content)
                checks["retrieval_and_arithmetic"] = parsed == {"code": "MAPLE", "owner": "COMET", "remaining": 3711}
            result.update(checks=checks, passed=all(checks.values()))
        except Exception as exc:
            result.update(passed=False, seconds=time.monotonic() - started,
                          error_type=type(exc).__name__, error=str(exc)[:1000])
        report["cases"].append(result)
        atomic(args.root / "runtime-report.json", report)
        print("RUNTIME CHECK", name, "PASS" if result["passed"] else "FAIL", flush=True)
    report.update(status="passed" if all(c["passed"] for c in report["cases"]) else "failed",
                  finished=time.time())
    atomic(args.root / "runtime-report.json", report)
    atomic(args.root / "runtime-progress.json", {"phase": report["status"], "time": time.time()})
    return report["status"] == "passed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:19353")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--long-input", type=Path, required=True)
    raise SystemExit(0 if run(parser.parse_args()) else 1)
