"""Small task checks, kept separate from throughput and broad model-quality claims."""
import argparse
import json
import re
import statistics
import subprocess
from pathlib import Path
from bench import cases


def assess(expected, message):
    text = (message.get("content") or "").strip()
    calls = message.get("tool_calls") or []
    if "tool" in expected:
        valid = False
        for call in calls:
            f = call.get("function", {})
            try:
                a = json.loads(f.get("arguments", "{}"))
            except (TypeError, ValueError):
                continue
            if f.get("name") != expected["tool"]:
                continue
            if expected["tool"] == "search_news":
                valid |= str(a.get("country", "")).lower() in ["japan", "일본", "jp"] and bool(a.get("query"))
            else:
                valid |= "신길동" in str(a.get("location", ""))
        return {"task_correct": valid, "format_correct": valid}
    if "no_tool" in expected:
        return {"task_correct": bool(text) and not calls, "format_correct": not calls, "note": "Checks absence of unnecessary calls; scientific explanation is reviewed separately."}
    if "json" in expected:
        try:
            strict = json.loads(text) == expected["json"]
        except ValueError:
            strict = False
        bare = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        try:
            semantic = json.loads(bare) == expected["json"]
        except ValueError:
            semantic = False
        return {"task_correct": semantic, "format_correct": strict}
    if "text" in expected:
        want = expected["text"]
        normalized = re.sub(r"\s+", "", text)
        return {"task_correct": want.replace(" ", "") in normalized, "format_correct": normalized == want.replace(" ", "")}
    if "manual" in expected:
        lines = text.splitlines()
        body = " ".join(lines[1:]).strip()
        sentences = [v for v in re.split(r"[.!?。！？]+", body) if v.strip()]
        facts = all(v in body for v in ["금요일", "화요일", "검증", "양해"])
        valid = bool(lines) and lines[0] == "일정 변경 안내" and len(sentences) == 3 and facts
        return {"task_correct": facts, "format_correct": valid, "sentences": len(sentences), "note": "Naturalness and correctness require reading the retained answer."}
    return {}


def code_check(text, image):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if not blocks:
        return {"passed": False, "error": "No code block"}
    code = next((b for b in blocks if "def merge_intervals(" in b), blocks[0])
    test = '''
import copy, json
checks = [([], []), ([[1,3],[2,6],[8,10]], [[1,6],[8,10]]), ([[5,7],[1,3],[2,6]], [[1,7]]), ([[1,9],[2,3]], [[1,9]]), ([[1,2],[2,4]], [[1,4]]), ([[-5,-2],[-3,1]], [[-5,1]]), ([[1,1],[1,1]], [[1,1]]), ([[7,9],[1,2]], [[1,2],[7,9]])]
passed = 0
for data, expected in checks:
    original = copy.deepcopy(data)
    result = merge_intervals(data)
    assert [list(x) for x in result] == expected, (data, result, expected)
    assert data == original, 'input mutated'
    passed += 1
try:
    merge_intervals([[4,1]])
except ValueError:
    passed += 1
else:
    raise AssertionError('invalid interval was not rejected')
print(json.dumps({'passed': True, 'checks': passed}))
'''
    try:
        r = subprocess.run(["docker", "run", "--rm", "-i", "--network", "none", "--read-only", "--memory", "128m", "--cpus", "1", "--pids-limit", "32", "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--entrypoint", "python3", image, "-I", "-"], input=code + test, text=True, capture_output=True, timeout=30)
        if r.returncode == 0:
            return json.loads(r.stdout.splitlines()[-1])
        return {"passed": False, "error": r.stderr[-1000:]}
    except (subprocess.TimeoutExpired, ValueError) as exc:
        return {"passed": False, "error": str(exc)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("directory")
    p.add_argument("--code-image")
    args = p.parse_args()
    root = Path(args.directory)
    wanted = {c["id"]: c for c in cases()}
    scores, speed = [], {}
    for f in sorted(root.glob("*.json")):
        row = json.loads(f.read_text())
        if not isinstance(row, dict) or row.get("case") not in wanted or not row.get("response"):
            continue
        result = row["response"]
        choice = result["choices"][0]
        message = choice["message"]
        usage = result["usage"]
        perf = result["tensorfold"]
        prompt = usage["prompt_tokens"] - usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        timing = {"tg": perf["tokens_per_second"], "pp": prompt / perf["prefill_seconds"] if perf["prefill_seconds"] else None, "ttft": perf["time_to_first_token"], "wall_s": row["wall_seconds"], "answer_chars": len(message.get("content") or ""), "finish_reason": choice["finish_reason"], "output_tokens": usage["completion_tokens"], "reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)}
        speed.setdefault(row["case"] + "/" + row["mode"], []).append(timing)
        if row["rep"] == 0 and wanted[row["case"]]["expected"]:
            scores.append({"case": row["case"], "mode": row["mode"], **assess(wanted[row["case"]]["expected"], message), "complete": choice["finish_reason"] != "length"})
        if row["case"] == "code-en" and args.code_image:
            scores.append({"case": "code-en", "mode": row["mode"], "rep": row["rep"], "complete": choice["finish_reason"] != "length", "code": code_check(message.get("content") or "", args.code_image)})
    out = {"scope": "Small synthetic checks; not a standardized benchmark or evidence of general quality superiority.", "scores": scores, "speed": speed}
    (root / "scored.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    for k, vals in speed.items():
        if k.startswith(("prose-ko/", "code-en/", "retrieval/")):
            print(k, "TG median", round(statistics.median(v["tg"] for v in vals), 2), "PP median", round(statistics.median(v["pp"] for v in vals), 2))
    for mode in ["off", "medium"]:
        rows = [r for r in scores if r["mode"] == mode and "task_correct" in r]
        print(mode, "task", sum(r["task_correct"] for r in rows), "/", len(rows), "format", sum(r["format_correct"] for r in rows), "/", len(rows))


if __name__ == "__main__":
    main()
