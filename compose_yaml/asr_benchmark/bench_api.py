#!/usr/bin/env python3
import argparse
import json
import statistics
import time
import unicodedata
from pathlib import Path

import requests


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


def transcribe(endpoint: str, model: str, audio_path: Path, language: str) -> tuple[str, float]:
    start = time.perf_counter()
    with audio_path.open("rb") as audio:
        response = requests.post(
            endpoint,
            data={"model": model, "language": language},
            files={"file": (audio_path.name, audio, "audio/wav")},
            timeout=600,
        )
    elapsed = time.perf_counter() - start
    response.raise_for_status()
    return response.json()["text"], elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8694/v1/audio/transcriptions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", default="ko")
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in (args.corpus / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Warm up kernels and caches without counting the request.
    transcribe(
        args.endpoint,
        args.model,
        args.corpus / records[0]["file"],
        args.language,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    total_edits = 0
    total_characters = 0
    for record in records:
        text, latency = transcribe(
            args.endpoint,
            args.model,
            args.corpus / record["file"],
            args.language,
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
            "edits": edits,
            "reference_characters": len(reference),
        }
        results.append(result)
        print(
            f"{record['file']} latency={latency:.3f}s "
            f"cer={edits / max(1, len(reference)):.4f}"
        )

    with args.output.open("w", encoding="utf-8") as output:
        for result in results:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")

    latencies = [result["latency_seconds"] for result in results]
    audio_seconds = sum(result["duration_seconds"] for result in results)
    inference_seconds = sum(latencies)
    sorted_latencies = sorted(latencies)
    p95_index = min(len(sorted_latencies) - 1, round(0.95 * (len(sorted_latencies) - 1)))
    print(f"samples={len(results)}")
    print(f"cer={total_edits / max(1, total_characters):.6f}")
    print(f"latency_mean_seconds={statistics.mean(latencies):.6f}")
    print(f"latency_median_seconds={statistics.median(latencies):.6f}")
    print(f"latency_p95_seconds={sorted_latencies[p95_index]:.6f}")
    print(f"audio_seconds={audio_seconds:.3f}")
    print(f"inference_seconds={inference_seconds:.3f}")
    print(f"rtf={inference_seconds / audio_seconds:.6f}")
    print(f"realtime_factor_x={audio_seconds / inference_seconds:.3f}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
