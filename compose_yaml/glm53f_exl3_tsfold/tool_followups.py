"""Replay deterministic tool results through the same model after its tool call."""
import argparse
import json
import time
from pathlib import Path
from fidelity import post

NEWS = {"benchmark_fixture": True, "articles": [
    {"title": "사쿠라 철도 야간 운행 확대", "summary": "사쿠라 철도는 10월 25일부터 막차를 23시 40분까지 운행한다.", "url": "https://example.test/news/rail"},
    {"title": "아오바 도서관 임시 휴관", "summary": "아오바 도서관은 설비 점검 때문에 10월 26일 하루 휴관한다.", "url": "https://example.test/news/library"}
]}
WEATHER = {"benchmark_fixture": True, "location": "서울 신길동", "temperature_c": 17.5, "condition": "맑음", "rain_probability_percent": 10}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--directory", required=True)
    args = p.parse_args()
    root = Path(args.directory)
    rows = []
    for mode in ["off", "medium"]:
        for name in ["news-japan", "weather-seoul"]:
            original = json.loads((root / f"{name}-{mode}-0.json").read_text())
            message = original["response"]["choices"][0]["message"]
            calls = message.get("tool_calls") or []
            if not calls:
                rows.append({"case": name, "mode": mode, "passed": False, "error": "No initial tool call"})
                continue
            body = dict(original["request"])
            body["messages"] = list(body["messages"]) + [message]
            for call in calls:
                fixture = NEWS if call["function"]["name"] == "search_news" else WEATHER
                body["messages"].append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(fixture, ensure_ascii=False)})
            start = time.monotonic()
            response = post(args.url, "/v1/chat/completions", body)
            answer = response["choices"][0]["message"]
            text = answer.get("content") or ""
            markers = ["사쿠라", "아오바"] if name == "news-japan" else ["17.5", "맑"]
            passed = all(v in text for v in markers) and not answer.get("tool_calls") and response["choices"][0]["finish_reason"] != "length"
            row = {"case": name, "mode": mode, "request": body, "response": response, "passed": passed, "seconds": time.monotonic() - start}
            rows.append(row)
            (root / "tool-followups.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({"case": name, "mode": mode, "passed": passed, "answer": text}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
