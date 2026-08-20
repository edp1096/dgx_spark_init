#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=20)
    args = parser.parse_args()

    table = pq.read_table(
        args.parquet,
        columns=["id", "num_samples", "audio", "transcription", "gender"],
    )
    rows = table.to_pylist()
    candidates = [
        row
        for row in rows
        if 3.0 <= row["num_samples"] / 16000.0 <= 20.0
        and row["audio"]["bytes"]
        and row["transcription"]
    ]
    candidates.sort(key=lambda row: (row["num_samples"], row["id"]))

    if len(candidates) < args.count:
        raise RuntimeError(f"only {len(candidates)} usable rows")

    # Pick deterministic duration quantiles instead of only short, easy clips.
    indexes = [
        round(i * (len(candidates) - 1) / (args.count - 1))
        for i in range(args.count)
    ]
    selected = [candidates[index] for index in indexes]

    audio_dir = args.output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, row in enumerate(selected, 1):
            filename = f"sample-{index:03d}.wav"
            (audio_dir / filename).write_bytes(row["audio"]["bytes"])
            record = {
                "id": row["id"],
                "file": f"audio/{filename}",
                "duration_seconds": row["num_samples"] / 16000.0,
                "gender": row["gender"],
                "reference": row["transcription"],
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")

    total_seconds = sum(row["num_samples"] for row in selected) / 16000.0
    print(f"samples={len(selected)}")
    print(f"audio_seconds={total_seconds:.3f}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
