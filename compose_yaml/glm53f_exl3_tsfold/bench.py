"""Matched OpenAI API experiment; retains requests and complete unscored replies."""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

SEARCH = {"type": "function", "function": {"name": "search_news", "description": "Search current news for a country. Required for current-news questions.", "parameters": {"type": "object", "properties": {"country": {"type": "string"}, "query": {"type": "string"}}, "required": ["country", "query"]}}}
WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Get current weather for a location.", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}

def cases():
    rows = [
        ("prose-ko", "한국어로 메모리와 SSD의 역할, 속도 차이, 용량 부족 현상을 구체적인 예를 들어 12문장으로 설명하세요. 검색은 필요하지 않습니다.", None),
        ("code-en", "Write a Python function merge_intervals(intervals) that returns sorted merged closed integer intervals. Reject intervals whose start exceeds their end, do not mutate the input, and explain the algorithm and its complexity with three examples.", None),
        ("arithmetic-net-price", "120000원 상품을 15% 할인한 뒤, 할인 후 상품 가격의 10%를 부가세로 붙이고 배송비 3000원을 더한다. 최종 금액은? JSON 객체 {\"won\": 정수}만 출력.", {"json": {"won": 115200}}),
        ("probability", "상자에 빨간 공 3개와 파란 공 2개가 있다. 복원 없이 두 개를 뽑았을 때 색이 같을 확률은? 기약분수 하나만 답해라.", {"text": "2/5"}),
        ("ordering", "A,B,C,D 네 작업을 순서대로 하나씩 실행한다. B는 A 바로 뒤이며 C는 A보다 앞이고 D는 C보다 앞이다. 작업 순서를 쉼표로 구분해 출력하라.", {"text": "D,C,A,B"}),
        ("extract-json", "다음 명단에서 서울 거주이며 나이가 30 이상인 사람의 이름만 원래 순서대로 JSON 배열로 출력하라: 민수/서울/29, 지연/부산/35, 하나/서울/30, 준/서울/42, 서연/서울/19.", {"json": ["하나", "준"]}),
        ("grounded-unknown", "주어진 자료만 사용하라. 자료: '프로젝트 누리는 월요일 시작했고 예산은 420만원이다.' 프로젝트 책임자는 누구인가? 자료에 없으면 '알 수 없음'만 답하라.", {"text": "알 수 없음"}),
        ("korean-format", "개발팀에 보낼 정중한 일정 변경 안내를 한국어로 작성하라. 첫 줄은 '일정 변경 안내'이고, 본문은 정확히 세 문장이다. 금요일 배포를 다음 주 화요일로 연기한다는 사실, 추가 검증이 이유라는 사실, 양해 요청을 모두 담아라.", {"manual": "Title, three sentences, all three facts, natural Korean"}),
        ("news-japan", "오늘 일본 주요 뉴스를 검색해서 알려줘.", {"tool": "search_news", "argument_contains": "일본"}),
        ("weather-seoul", "지금 서울 신길동 날씨를 확인해 줘.", {"tool": "get_weather", "argument_contains": "신길동"}),
        ("no-tool", "현재 정보나 검색 없이 물이 얼면 부피가 커지는 이유를 한국어로 두 문장으로 설명해 줘.", {"no_tool": True}),
    ]
    records = "\n".join(f"보관함 {i:04d}: 담당팀은 팀{i % 23:02d}이고 확인 코드는 K{i * 7919 % 1000000:06d}이다." for i in range(250))
    rows.append(("retrieval", records + "\n보관함 0187의 확인 코드만 답하라.", {"text": f"K{187 * 7919 % 1000000:06d}"}))
    out = []
    for name, prompt, expected in rows:
        row = {"id": name, "prompt": prompt, "expected": expected}
        if name in ("news-japan", "weather-seoul", "no-tool"):
            row["tools"] = [SEARCH, WEATHER]
        out.append(row)
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--only", default="")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--modes", default="off,medium")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suite = cases()
    (out / "cases.json").write_text(json.dumps(suite, ensure_ascii=False, indent=2) + "\n")
    for mode in args.modes.split(","):
        for case in suite:
            if args.only and case["id"] not in args.only.split(","):
                continue
            for rep in range(args.reps):
                dest = out / f'{case["id"]}-{mode}-{rep}.json'
                if dest.exists():
                    continue
                system = f'Benchmark case {case["id"]}, mode {mode}, repetition {rep}. Follow the user\'s instructions carefully. Use available tools for current information. Do not invent tool results.'
                body = {"model": args.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": case["prompt"]}], "temperature": 0.7, "top_k": 20, "top_p": 0.95, "seed": 99 + rep, "max_tokens": 1536 if mode == "medium" else 768, "stream": False, "reasoning_effort": "none" if mode == "off" else mode, "chat_template_kwargs": {"enable_thinking": mode != "off"}}
                if case["id"] == "code-en" and mode == "medium":
                    body["max_tokens"] = 4096
                if case.get("tools"):
                    body.update(tools=case["tools"], tool_choice="auto")
                req = urllib.request.Request(args.url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body, ensure_ascii=False).encode(), headers={"Content-Type": "application/json"})
                start = time.monotonic()
                try:
                    with urllib.request.urlopen(req, timeout=900) as response:
                        result = json.load(response)
                    error = None
                except (urllib.error.URLError, TimeoutError) as exc:
                    result = None
                    error = str(exc)
                    if isinstance(exc, urllib.error.HTTPError):
                        error += " " + exc.read().decode(errors="replace")
                wall = time.monotonic() - start
                row = {"case": case["id"], "mode": mode, "rep": rep, "request": body, "expected": case["expected"], "wall_seconds": wall, "response": result, "error": error}
                dest.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
                msg = ((result or {}).get("choices") or [{}])[0].get("message", {})
                print(json.dumps({"case": case["id"], "mode": mode, "rep": rep, "wall_s": round(wall, 2), "usage": (result or {}).get("usage"), "native": (result or {}).get("tensorfold"), "chars": len(msg.get("content") or ""), "tool_calls": len(msg.get("tool_calls") or []), "error": error}, ensure_ascii=False), flush=True)
                if error:
                    raise RuntimeError(error)

if __name__ == "__main__":
    main()
