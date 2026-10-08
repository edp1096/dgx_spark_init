#!/usr/bin/env python3
import json
import sys
import time
import urllib.error
import urllib.request


BASE_URL = "http://127.0.0.1:8000"
MODEL = "Huihui-Qwen3.8-27B-abliterated-FP8"


def request(path, payload=None, timeout=300):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read()
        return None if not body else json.loads(body)


def request_text(path, timeout=30):
    with urllib.request.urlopen(BASE_URL + path, timeout=timeout) as response:
        return response.read().decode()


deadline = time.monotonic() + 600
while True:
    try:
        request("/health", timeout=5)
        break
    except (OSError, urllib.error.URLError):
        if time.monotonic() >= deadline:
            sys.exit("vLLM health check timed out")
        print("waiting for vLLM...", flush=True)
        time.sleep(10)

models = request("/v1/models")
ids = [item["id"] for item in models.get("data", [])]
if MODEL not in ids:
    sys.exit(f"expected model is missing: {ids}")

result = request(
    "/v1/chat/completions",
    {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "한 문장으로 답해라. 지금 정상적으로 추론 중인가?",
            }
        ],
        "temperature": 0,
        "max_tokens": 256,
    },
)
print(json.dumps(result, ensure_ascii=False, indent=2))

metrics = request_text("/metrics")
spec_metrics = [
    line
    for line in metrics.splitlines()
    if line.startswith("vllm:spec_decode_") and not line.startswith("#")
]
if not spec_metrics:
    sys.exit("generation succeeded, but DSpark metrics are missing")

print("\nDSpark metrics:")
print("\n".join(spec_metrics))
