#!/usr/bin/env python3
import argparse
import json
import re
import time
import urllib.request


BASE_URL = "http://127.0.0.1:8000"
METRIC_NAMES = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
)
CASES = (
    (
        "knowledge",
        "광합성의 명반응과 캘빈 회로를 일반인이 이해할 수 있도록 한국어로 5문단 이내로 설명해라.",
    ),
    (
        "technology",
        "HTTP/2와 HTTP/3의 차이를 전송 계층, 멀티플렉싱, 연결 설정, 패킷 손실 관점에서 표와 짧은 결론으로 설명해라.",
    ),
    (
        "coding",
        "파이썬 표준 라이브러리만 사용해 제네릭 LRU 캐시 클래스를 작성하고 시간 복잡도와 동작 원리를 설명해라.",
    ),
    (
        "reasoning",
        "상자 A에는 빨간 공 4개와 파란 공 6개, 상자 B에는 빨간 공 7개와 파란 공 3개가 있다. 상자를 같은 확률로 하나 고른 뒤 공 하나를 뽑았더니 빨간색이었다. 선택한 상자가 B일 확률을 베이즈 정리로 단계별 설명해라.",
    ),
)


def get_json(path):
    with urllib.request.urlopen(BASE_URL + path, timeout=30) as response:
        return json.loads(response.read())


def get_metrics():
    with urllib.request.urlopen(BASE_URL + "/metrics", timeout=30) as response:
        text = response.read().decode()
    metrics = {}
    for name in METRIC_NAMES:
        match = re.search(
            r"^" + re.escape(name) + r"\{[^\n]*\}\s+([0-9.eE+-]+)$",
            text,
            re.MULTILINE,
        )
        if not match:
            raise RuntimeError(f"metric is missing: {name}")
        metrics[name] = float(match.group(1))
    return metrics


def completion(model, prompt):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 384,
        "seed": 42,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        BASE_URL + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.loads(response.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    models = get_json("/v1/models")["data"]
    if len(models) != 1:
        raise RuntimeError(f"expected one served model, got: {models}")
    model = models[0]["id"]
    results = []

    for name, prompt in CASES:
        before = get_metrics()
        started = time.monotonic()
        response = completion(model, prompt)
        elapsed = time.monotonic() - started
        after = get_metrics()
        delta = {key: after[key] - before[key] for key in METRIC_NAMES}
        drafts, proposed, accepted = (delta[key] for key in METRIC_NAMES)
        completion_tokens = response["usage"]["completion_tokens"]
        result = {
            "case": name,
            "elapsed_seconds": round(elapsed, 3),
            "completion_tokens": completion_tokens,
            "client_tokens_per_second": round(completion_tokens / elapsed, 3),
            "draft_rounds": int(drafts),
            "proposed_tokens": int(proposed),
            "accepted_tokens": int(accepted),
            "accepted_per_round": round(accepted / drafts, 4) if drafts else 0,
            "draft_acceptance_rate": round(accepted / proposed, 6)
            if proposed
            else 0,
            "finish_reason": response["choices"][0]["finish_reason"],
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    totals = {
        "elapsed": sum(item["elapsed_seconds"] for item in results),
        "completion_tokens": sum(item["completion_tokens"] for item in results),
        "draft_rounds": sum(item["draft_rounds"] for item in results),
        "proposed": sum(item["proposed_tokens"] for item in results),
        "accepted": sum(item["accepted_tokens"] for item in results),
    }
    summary = {
        "label": args.label,
        "model": model,
        "cases": len(results),
        "completion_tokens": totals["completion_tokens"],
        "elapsed_seconds": round(totals["elapsed"], 3),
        "client_tokens_per_second": round(
            totals["completion_tokens"] / totals["elapsed"], 3
        ),
        "accepted_per_round": round(
            totals["accepted"] / totals["draft_rounds"], 4
        ),
        "draft_acceptance_rate": round(
            totals["accepted"] / totals["proposed"], 6
        ),
    }
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
