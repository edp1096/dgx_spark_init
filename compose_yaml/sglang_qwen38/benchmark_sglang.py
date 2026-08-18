#!/usr/bin/env python3
import argparse
import json
import re
import threading
import time
import urllib.request


BASE_URL = "http://127.0.0.1:8000"
METRICS = (
    "sglang:spec_accept_length",
    "sglang:spec_accept_rate",
    "sglang:spec_cap_length",
    "sglang:spec_block_accept_length",
)
CASES = (
    (
        "code_en",
        "Write a complete Python implementation of an async bounded worker pool using only the standard library. Include cancellation, exception propagation, type hints, and a short explanation. Do not omit code.",
    ),
    (
        "math_en",
        "Solve this carefully and explain each step: A box contains 4 red and 6 blue balls and another contains 7 red and 3 blue balls. A box is selected uniformly and a red ball is drawn. Find the probability that the second box was selected, then generalize the formula.",
    ),
    (
        "technical_ko",
        "HTTP/2와 HTTP/3의 차이를 전송 계층, 멀티플렉싱, 연결 설정, 패킷 손실 관점에서 표와 구체적인 예를 포함하여 설명해라.",
    ),
    (
        "prose_ko",
        "비가 그친 뒤의 조용한 항구 도시를 배경으로 약 800자 분량의 단편소설을 작성해라. 인물의 행동과 대화를 포함해라.",
    ),
)


def get_json(path):
    with urllib.request.urlopen(BASE_URL + path, timeout=30) as response:
        return json.loads(response.read())


def read_metrics():
    with urllib.request.urlopen(BASE_URL + "/metrics", timeout=5) as response:
        body = response.read().decode()
    result = {}
    for name in METRICS:
        match = re.search(r"^" + re.escape(name) + r"\{[^\n]*\}\s+([0-9.eE+-]+)$", body, re.MULTILINE)
        if match:
            result[name] = float(match.group(1))
    return result


def sample_metrics(stop, samples):
    while not stop.wait(0.05):
        try:
            values = read_metrics()
        except OSError:
            continue
        if any(values.values()):
            samples.append(values)


def completion(model, prompt, thinking):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 512,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    req = urllib.request.Request(
        BASE_URL + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1200) as response:
        return json.loads(response.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()

    model = get_json("/v1/models")["data"][0]["id"]
    results = []
    completion(model, "Reply with exactly: warmup complete", False)

    for name, prompt in CASES:
        stop = threading.Event()
        samples = []
        sampler = threading.Thread(target=sample_metrics, args=(stop, samples), daemon=True)
        sampler.start()
        started = time.monotonic()
        response = completion(model, prompt, args.thinking)
        elapsed = time.monotonic() - started
        stop.set()
        sampler.join(timeout=2)

        usage = response["usage"]
        tokens = usage["completion_tokens"]
        nonzero = {
            name.removeprefix("sglang:spec_"): round(max((s.get(name, 0) for s in samples), default=0), 4)
            for name in METRICS
        }
        result = {
            "case": name,
            "elapsed_seconds": round(elapsed, 3),
            "completion_tokens": tokens,
            "client_tokens_per_second": round(tokens / elapsed, 3),
            "finish_reason": response["choices"][0]["finish_reason"],
            **nonzero,
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    elapsed = sum(item["elapsed_seconds"] for item in results)
    tokens = sum(item["completion_tokens"] for item in results)
    summary = {
        "label": args.label,
        "model": model,
        "thinking": args.thinking,
        "cases": len(results),
        "completion_tokens": tokens,
        "elapsed_seconds": round(elapsed, 3),
        "client_tokens_per_second": round(tokens / elapsed, 3),
    }
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
