#!/usr/bin/env python3
import json
import sys
import time
import urllib.error
import urllib.request


BASE_URL = "http://127.0.0.1:8000"


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


deadline = time.monotonic() + 1200
while True:
    try:
        request("/health", timeout=5)
        break
    except (OSError, urllib.error.URLError):
        if time.monotonic() >= deadline:
            sys.exit("SGLang health check timed out")
        print("waiting for SGLang...", flush=True)
        time.sleep(10)

models = request("/v1/models")["data"]
if len(models) != 1:
    sys.exit(f"expected one served model, got: {models}")
model = models[0]["id"]

result = request(
    "/v1/chat/completions",
    {
        "model": model,
        "messages": [
            {"role": "user", "content": "한 문장으로 답해라. 지금 정상적으로 추론 중인가?"}
        ],
        "temperature": 0,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    },
    timeout=900,
)
print(json.dumps(result, ensure_ascii=False, indent=2))
