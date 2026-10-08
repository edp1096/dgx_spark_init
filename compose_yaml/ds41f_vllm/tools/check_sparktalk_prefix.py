"""Check two real SparkTalk turns, recorded metrics and current-user overrides.

Uses the running app's settings and tools. Deletes only its own test session.
No memory or model configuration is changed.
"""

# Resolve serving modules when invoked as python3 tools/<script>.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8585"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=(Path(__file__).resolve().parents[1] / 'results') / "sparktalk-prefix-validation.json")
OUTPUT = parser.parse_args().output


def api(path, body=None, method=None):
    request = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"}, method=method)
    return urllib.request.urlopen(request, timeout=600)


config = json.load(api("/api/config"))
assert config["context"]["output_reserve"] == 8192
session = json.load(api("/api/sessions", {"title": "Temporary prefix cache validation"}))
result = {"session_id": session["id"], "turns": [], "temporary_session_deleted": False,
          "reasoning_effort": config["model"].get("reasoning_effort"),
          "context_tokens": config["context"]["window_tokens"],
          "output_allowance": config["context"]["output_reserve"]}
try:
    for prompt, marker in [
        ("성능 점검용 임시 대화다. 검증 표식 cobalt731 만 출력해.", "cobalt731"),
        ("검증 표식을 amber962로 바꾼다. 변경한 표식만 출력해.", "amber962"),
    ]:
        start = time.monotonic()
        row = {"expected": marker, "text": "", "done": False, "performance": None,
               "first_generated_event_seconds": None}
        kind = None
        with api("/api/chat", {"session_id": session["id"], "content": prompt,
                              "tools_enabled": True}) as response:
            for line in response:
                if line.startswith(b"event: "):
                    kind = line[7:].strip().decode()
                elif line.startswith(b"data: "):
                    data = json.loads(line[6:])
                    if kind in ("reasoning", "delta") and data.get("delta"):
                        if row["first_generated_event_seconds"] is None:
                            row["first_generated_event_seconds"] = time.monotonic() - start
                    if kind == "delta":
                        row["text"] += data.get("delta", "")
                    elif kind == "performance":
                        row["performance"] = data
                    elif kind == "done":
                        row["done"] = True
                    elif kind == "error":
                        raise AssertionError(data)
        row["elapsed_seconds"] = time.monotonic() - start
        result["turns"].append(row)
        OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
        assert row["done"] and row["text"].strip().strip('`". *') == marker, row
        assert row["performance"] and not row["performance"].get("live"), row
    messages = json.load(api("/api/sessions/" + session["id"] + "/messages"))
    answers = [m for m in messages if m["role"] == "assistant"]
    assert len(answers) == 2, answers
    for answer, row in zip(answers, result["turns"]):
        assert answer["performance"] == row["performance"], (answer, row)
    result["stored_metrics_match"] = True
    assert result["turns"][1]["performance"]["cached_tokens"] > 5000, result["turns"]
finally:
    with api("/api/sessions/" + session["id"], method="DELETE") as response:
        result["temporary_session_deleted"] = response.status == 204
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print("SPARKTALK_PREFIX_VALIDATION_PASS", flush=True)
