#!/usr/bin/env python3
import argparse
import json
import re
import statistics
import subprocess
import time
import unicodedata
from pathlib import Path


TAG_PATTERN = re.compile(r"<\|[^|]+\|>")
INFERENCE_PATTERN = re.compile(r"\[sensevoice\] done ([0-9.]+)s")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(character for character in text if character.isalnum())


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, reference_character in enumerate(reference, 1):
        current = [row]
        for column, hypothesis_character in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1]
                    + (reference_character != hypothesis_character),
                )
            )
        previous = current
    return previous[-1]


def transcribe(command: list[str], audio_path: Path) -> tuple[str, float, float | None]:
    start = time.perf_counter()
    result = subprocess.run(
        [part.replace("{audio}", str(audio_path)) for part in command],
        check=True,
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed = time.perf_counter() - start
    text = TAG_PATTERN.sub("", result.stdout).strip()
    inference_match = INFERENCE_PATTERN.search(result.stderr)
    inference_seconds = (
        float(inference_match.group(1)) if inference_match is not None else None
    )
    return text, elapsed, inference_seconds


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="command after --; use {audio} for the input path",
    )
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not any("{audio}" in part for part in command):
        parser.error("command must contain an {audio} placeholder")

    records = [
        json.loads(line)
        for line in (args.corpus / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    total_edits = 0
    total_characters = 0
    for record in records:
        text, latency, inference_seconds = transcribe(
            command, args.corpus / record["file"]
        )
        reference = normalize(record["reference"])
        hypothesis = normalize(text)
        edits = edit_distance(reference, hypothesis)
        total_edits += edits
        total_characters += len(reference)
        result = {
            **record,
            "text": text,
            "latency_seconds": latency,
            "inference_seconds": inference_seconds,
            "edits": edits,
            "reference_characters": len(reference),
        }
        results.append(result)
        print(
            f"{record['file']} latency={latency:.3f}s "
            f"inference={inference_seconds if inference_seconds is not None else 'n/a'} "
            f"cer={edits / max(1, len(reference)):.4f}"
        )

    with args.output.open("w", encoding="utf-8") as output:
        for result in results:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")

    latencies = [result["latency_seconds"] for result in results]
    inference_times = [
        result["inference_seconds"]
        for result in results
        if result["inference_seconds"] is not None
    ]
    audio_seconds = sum(result["duration_seconds"] for result in results)
    print(f"samples={len(results)}")
    print(f"cer={total_edits / max(1, total_characters):.6f}")
    print(f"latency_mean_seconds={statistics.mean(latencies):.6f}")
    print(f"latency_median_seconds={statistics.median(latencies):.6f}")
    print(f"latency_p95_seconds={percentile(latencies, 0.95):.6f}")
    if inference_times:
        inference_total = sum(inference_times)
        print(f"inference_mean_seconds={statistics.mean(inference_times):.6f}")
        print(f"inference_p95_seconds={percentile(inference_times, 0.95):.6f}")
        print(f"inference_realtime_factor_x={audio_seconds / inference_total:.3f}")
    print(f"audio_seconds={audio_seconds:.3f}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
