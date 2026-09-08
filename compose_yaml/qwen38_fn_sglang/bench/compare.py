#!/usr/bin/env python3
"""Local TP1 comparison; synthetic prompts only, no user conversations.

Run with the server already healthy. Each timed prompt starts with a flushed
prefix cache. JSON records the actual outputs, usage and streaming timings.
"""
import argparse
import base64
import concurrent.futures
import json
import os
import pathlib
import statistics
import struct
import threading
import time
import zlib

import requests

PROMPTS = [
    ("ko_explain", "리눅스에서 프로세스와 스레드의 차이를 한국어로 설명해 주세요. 메모리 공유와 오류 격리, 웹 서버 예시를 포함해 충분히 설명하세요."),
    ("ko_plan", "작은 도서관의 도서 대출 프로그램을 설계하려고 합니다. 회원 등록, 대출, 반납, 연체 처리와 데이터베이스 테이블을 한국어로 설명해 주세요."),
    ("ko_story", "비 오는 날 오래된 서점에서 우연히 자신의 이름이 적힌 편지를 발견한 사람의 이야기를 한국어로 써 주세요. 대화와 구체적인 묘사를 포함하세요."),
    ("ko_compare", "집에서 소규모 서버를 운영할 때 유선 네트워크와 무선 네트워크의 차이를 한국어로 자세히 비교해 주세요. 지연, 안정성, 설치 비용을 다뤄 주세요."),
    ("code_python", "Write Python code for an LRU cache with get and put, capacity validation, and a small usage example. Use collections.OrderedDict. Return code only."),
    ("code_go", "Write Go code for a bounded worker pool processing integer jobs with context cancellation, a WaitGroup, and proper channel closure. Include a usage example. Return code only."),
    ("json", 'Return only a JSON object with a "books" array containing 6 invented books. Each book must have id, title, author, year, and available fields. Use valid JSON.'),
    ("en_prose", "Explain how a refrigerator moves heat, including the compressor, condenser, expansion valve and evaporator, for a curious teenager. Use complete paragraphs."),
]

CORPUS = [
    "한국어로 컴퓨터의 CPU, GPU, 메모리, 저장장치가 협력하는 원리를 설명하세요.",
    "한국어로 봄날 산책을 하며 만난 사람들과 풍경에 대한 짧은 이야기를 써 주세요.",
    "네트워크 설정을 점검하는 방법을 한국어로 설명하세요. IP 주소, DNS, 게이트웨이를 포함하세요.",
    "한국어로 파일 백업 계획과 복원 절차를 작성하세요. 매일, 매주, 매월 작업을 포함하세요.",
    "한국어로 프로그램 오류를 재현하고 원인을 찾아 수정하고 테스트하는 과정을 설명하세요.",
    "한국어로 여행 일정을 계획하는 두 사람의 자연스러운 대화를 써 주세요.",
    "Explain database transactions, indexes, joins, and query planning with examples.",
    "Write Python code that reads a CSV file, validates rows, groups by category and writes JSON.",
    "Write Go code for an HTTP JSON handler with validation, timeouts, and error handling.",
    "Create example JSON requests and responses for searching documents and checking server status.",
    "한국어로 bash, Docker, Git 명령어를 사용하는 개발 흐름을 예시와 함께 설명하세요.",
    "한국어로 수학 문제의 풀이를 설명하세요. 비율, 확률, 평균과 단위 환산 예시를 포함하세요.",
]


def chat(url, prompt, *, tools=None, thinking=False, max_tokens=256):
    body = dict(model="qwen3.8-flash-next", messages=[dict(role="user", content=prompt)],
                temperature=0, max_tokens=max_tokens, stream=True,
                stream_options={"include_usage": True},
                chat_template_kwargs={"enable_thinking": thinking})
    if tools:
        body.update(tools=tools, tool_choice="auto")
    start = time.monotonic()
    first = last = None
    content, reasoning, calls, usage = [], [], {}, {}
    finish = None
    with requests.post(url + "/v1/chat/completions", json=body, stream=True, timeout=600) as r:
        r.raise_for_status()
        for line in r.iter_lines(chunk_size=1):
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            data = json.loads(line[6:])
            if data.get("error"):
                raise RuntimeError(data["error"])
            usage = data.get("usage") or usage
            for choice in data.get("choices", []):
                delta = choice.get("delta", {})
                text = delta.get("content") or ""
                thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                if text or thought or delta.get("tool_calls"):
                    last = time.monotonic()
                    if first is None:
                        first = last
                content.append(text)
                reasoning.append(thought)
                for call in delta.get("tool_calls") or []:
                    dst = calls.setdefault(call["index"], {"name": "", "arguments": ""})
                    for k in dst:
                        dst[k] += call.get("function", {}).get(k) or ""
                finish = choice.get("finish_reason") or finish
    end = time.monotonic()
    tokens = usage.get("completion_tokens", 0)
    return dict(content="".join(content), reasoning="".join(reasoning),
                tool_calls=list(calls.values()), usage=usage, finish=finish,
                ttft_s=first - start if first else None, elapsed_s=end-start,
                decode_tok_s=(tokens-1)/(last-first) if first and last > first and tokens > 1 else None)


def flush(url):
    r = requests.post(url + "/flush_cache", timeout=30)
    if r.status_code == 405:
        r = requests.get(url + "/flush_cache", timeout=30)
    r.raise_for_status()


def image_prompt():
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind+data))
    raw = b"".join(b"\0" + b"\xff\0\0"*64 + b"\0\0\xff"*64 for _ in range(64))
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB",128,64,8,2,0,0,0)) + chunk(b"IDAT",zlib.compress(raw)) + chunk(b"IEND", b"")
    return [{"type":"text","text":"What color is on the left and what color is on the right? Answer in English."},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,"+base64.b64encode(png).decode()}}]


def valid_temperature_call(calls):
    for call in calls:
        try:
            if call["name"] == "get_temperature" and json.loads(call["arguments"]).get("city") == "Seoul":
                return True
        except (ValueError, AttributeError):
            pass
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--output", required=True)
    ap.add_argument("--corpus", action="store_true")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = {"url":args.url,"started":time.strftime("%FT%T%z"),"rows":[],"checks":[]}
    if args.resume and out.exists():
        results=json.loads(out.read_text())
    stop = threading.Event()
    available = []
    if results.get("min_mem_available_gib") is not None:
        available.append(results["min_mem_available_gib"])
    def monitor():
        while not stop.is_set():
            fields = dict(line.split(":",1) for line in pathlib.Path("/proc/meminfo").read_text().splitlines())
            available.append(int(fields["MemAvailable"].split()[0]) / 1048576)
            stop.wait(1)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    def save():
        results["min_mem_available_gib"] = min(available) if available else None
        temporary = out.with_suffix(out.suffix + ".tmp")
        with temporary.open("w") as handle:
            handle.write(json.dumps(results, ensure_ascii=False, indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(out)
    try:
        if args.corpus:
            for i, prompt in enumerate(CORPUS):
                if any(r["name"] == f"corpus_{i}" for r in results["rows"]):
                    continue
                row=chat(args.url,prompt,max_tokens=384)
                results["rows"].append(dict(name=f"corpus_{i}",prompt=prompt,**row));save()
                print(f"corpus {i+1}/{len(CORPUS)}",flush=True)
            return
        chat(args.url,"Say ready.",max_tokens=16)
        for repeat in range(args.rounds):
            for name, prompt in PROMPTS:
                if any(r["name"] == name and r["repeat"] == repeat for r in results["rows"]):
                    continue
                flush(args.url)
                row=chat(args.url,prompt,max_tokens=384 if name == "json" else 256)
                results["rows"].append(dict(name=name,repeat=repeat,prompt=prompt,**row));save()
                print(name,repeat,round(row["decode_tok_s"] or 0,2),row["usage"],flush=True)
        tool=[dict(type="function",function=dict(name="get_temperature",description="Get a city's current temperature",parameters=dict(type="object",properties={"city":{"type":"string"}},required=["city"])))]
        for thinking in [False,True]:
            if any(r["name"] == f"tools_thinking_{thinking}" for r in results["checks"]):
                continue
            flush(args.url)
            row=chat(args.url,'서울의 현재 기온을 get_temperature 도구로 확인하세요. city에는 "Seoul"을 넣으세요.',tools=tool,thinking=thinking,max_tokens=512)
            valid=valid_temperature_call(row["tool_calls"])
            results["checks"].append(dict(name=f"tools_thinking_{thinking}",passed=valid,**row));save()
        flush(args.url)
        row=chat(args.url,image_prompt(),max_tokens=96)
        results["checks"].append(dict(name="vision",passed="red" in row["content"].lower() and "blue" in row["content"].lower(),**row));save()
        flush(args.url)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            rows=list(pool.map(lambda p:chat(args.url,p,max_tokens=128),[PROMPTS[0][1],PROMPTS[4][1]]))
        results["checks"].append(dict(name="concurrency_2",passed=all(r["content"] and "!!!!" not in r["content"] for r in rows),responses=rows))
        flush(args.url)
        filler="The archive contains ordinary records about rain, books, trees, roads and clocks. No secret code appears in these records.\n"
        prompt=filler*700 + "\nThe secret access code is MAPLE-7392. Remember this exact code.\n" + filler*700 + "\nWhat is the secret access code? Reply with the code only."
        row=chat(args.url,prompt,max_tokens=64)
        results["checks"].append(dict(name="long_context",passed="MAPLE-7392" in row["content"],**row))
        for row in results["rows"]:
            if row["name"] == "json":
                try:
                    books=json.loads(row["content"])["books"]
                    valid=len(books)==6 and all({"id","title","author","year","available"} <= set(book) for book in books)
                except (ValueError, KeyError, TypeError):
                    valid=False
                results["checks"].append(dict(name=f"json_round_{row['repeat']}",passed=valid))
        results["summary"]={name:statistics.median(r["decode_tok_s"] for r in results["rows"] if r["name"]==name and r["decode_tok_s"]) for name,_ in PROMPTS}
        save()
        print(json.dumps({"summary":results["summary"],"checks":[(x['name'],x['passed']) for x in results['checks']],"min_mem_available_gib":results['min_mem_available_gib']},ensure_ascii=False),flush=True)
    finally:
        stop.set();thread.join();save()


if __name__ == "__main__":
    main()
